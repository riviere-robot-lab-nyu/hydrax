"""ROS 2 node that runs the AtmosM3 MPPI planner.

Subscribes to:
  - BASE_STATE_TOPIC  (nav_msgs/Odometry)
        pose.position.x/y, pose.orientation -> yaw, twist.linear.x/y in the
        child_frame_id (body frame) rotated to world, twist.angular.z as wz.
  - ARM_STATE_TOPIC   (sensor_msgs/JointState)
        arm joint positions and velocities, keyed by joint name.

Publishes (at `plan_rate_hz`, default 30 Hz):
  - <plan_topic>  (trajectory_msgs/JointTrajectory)
        Dense per-sim-step control sequence over the planning horizon. Each
        point carries 11 values matching AtmosM3's ctrl layout:
          0..2  base velocity command (vx_body, vy_body, wz_body)
          3..8  arm joint command (absolute or delta, see AtmosM3.arm_mode)
          9     gripper command (0=closed, 1=open)
          10    dead (always 0)
        time_from_start = (i+1) * sim_dt.

Run:
  ros2 run <pkg> mppi_ros_node      # if you wire it into a package
or:
  python3 mppi_ros_node.py
"""

import math
import time as _time

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

# Persistent on-disk XLA cache so the heavy MPPI JIT only pays its full
# cost once per (machine, jax/jaxlib, source-hash). Subsequent runs reuse
# the compiled program and `JIT done in X.XXs` collapses to ~1s.
jax.config.update("jax_compilation_cache_dir", "/workspace/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

from hydrax.algs import MPPI
from hydrax.tasks.atmos_m3 import AtmosM3


CONTROL_NAMES = [
    "vx_body_cmd", "vy_body_cmd", "wz_body_cmd",
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper_cmd", "_dead",
]

# ── Hard-coded topic / joint wiring (TODO: replace with real names) ───────
PLAN_TOPIC = "/mppi/plan"
ARM_STATE_TOPIC = "/TODO/arm/joint_states"
BASE_STATE_TOPIC = "/TODO/base/odom"   # nav_msgs/Odometry from the PX4 bridge
ARM_JOINT_NAMES = [
    "TODO_joint_0",
    "TODO_joint_1",
    "TODO_joint_2",
    "TODO_joint_3",
    "TODO_joint_4",
    "TODO_joint_5",
]

ODOM_LIN_VEL_START = 10
ODOM_ANG_VEL_START = 13

LOCAL_POS_HEADING_INDEX= 14
BASE_VEL_LIN_X_INDEX = 3
BASE_VEL_LIN_Y_INDEX = 4
BASE_VEL_YAW_RATE_INDEX = 15


class MppiPlannerNode(Node):
    def __init__(self):
        super().__init__("mppi_planner")

        # ── MPPI tuning parameters (overridable via ros2 args / YAML) ─────
        self.declare_parameter("plan_rate_hz", 20.0)
        self.declare_parameter("num_samples", 256)
        self.declare_parameter("plan_horizon", 0.25)
        self.declare_parameter("num_knots", 4)
        self.declare_parameter("temperature", 0.2)
        # If true, loads atmos_robot.xml only (skips wall_scene.xml +
        # carabiners + handles). Quick A/B knob for diagnosing per-step
        # cost regressions caused by added scene geometry.
        self.declare_parameter("use_robot_only", False)

        plan_rate = float(self.get_parameter("plan_rate_hz").value)
        num_samples = int(self.get_parameter("num_samples").value)
        plan_horizon = float(self.get_parameter("plan_horizon").value)
        num_knots = int(self.get_parameter("num_knots").value)
        temperature = float(self.get_parameter("temperature").value)
        use_robot_only = bool(self.get_parameter("use_robot_only").value)

        # Wiring is hard-coded at module top: PLAN_TOPIC, ARM_STATE_TOPIC,
        # BASE_STATE_TOPIC, ARM_JOINT_NAMES.
        self.arm_joint_names = list(ARM_JOINT_NAMES)

        # ── Task + controller ─────────────────────────────────────────────
        self.task = AtmosM3(use_robot_only=use_robot_only)
        self.ctrl = MPPI(
            self.task,
            num_samples=num_samples,
            noise_level=jnp.array(
                [0.1, 0.1, 0.1] + [0.01] * 8
            ),
            plan_horizon=plan_horizon,
            num_knots=num_knots,
            temperature=temperature,
            spline_type="zero",
        )
        self.mj_model = self.task.mj_model
        self.sim_dt = float(self.mj_model.opt.timestep)
        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)
        self.horizon_steps = max(int(round(plan_horizon / self.sim_dt)), 1)

        # Model-default qpos. Used as the base for _assemble_state so that
        # unfilled arm / gripper slots inherit a valid default pose instead
        # of zero (zeroing them puts the arm in an out-of-keyframe config,
        # which can diverge under mjx.step and feed inf -> NaN into MPPI).
        self.qpos0 = np.asarray(self.mj_model.qpos0, dtype=np.float32)

        self.mjx_data = mjx.make_data(self.task.model)
        self.policy_params = self.ctrl.init_params()
        self.integral = jnp.zeros(1)

        self.jit_optimize = jax.jit(self.ctrl.optimize)
        self.jit_interp = jax.jit(self.ctrl.interp_func)

        # Warm-up jit before any subscriber/timer can fire.
        # IMPORTANT: prime mjx_data with the exact dtypes / shapes that
        # _plan_and_publish will use at runtime (float32 qpos/qvel/time),
        # otherwise the first real call hits a fresh JIT trace and stalls
        # for tens of seconds on the Jetson.
        self.get_logger().info("Jitting MPPI controller...")
        t0 = _time.time()
        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.asarray(self.qpos0, dtype=jnp.float32),
            qvel=jnp.zeros(self.nv, dtype=jnp.float32),
            time=jnp.array(0.0, dtype=jnp.float32),
        )
        self.policy_params, _ = self.jit_optimize(
            self.mjx_data, self.policy_params, self.integral
        )
        jax.block_until_ready(self.policy_params)
        _tq_warm = (
            jnp.arange(0, self.horizon_steps) * self.sim_dt
            + self.mjx_data.time
        )
        _warm_interp = self.jit_interp(
            _tq_warm, self.policy_params.tk,
            self.policy_params.mean[None, ...],
        )
        jax.block_until_ready(_warm_interp)
        self.get_logger().info(f"JIT done in {_time.time() - t0:.2f}s")

        # ── State caches ──────────────────────────────────────────────────
        self.base_qpos = np.zeros(3, dtype=np.float32)   # x, y, yaw
        self.base_qvel = np.zeros(3, dtype=np.float32)   # vx_w, vy_w, wz
        self.arm_qpos = np.zeros(6, dtype=np.float32)
        self.arm_qvel = np.zeros(6, dtype=np.float32)
        # No source for gripper carriages yet — leave at zero (closed).
        self.gripper_qpos = np.zeros(2, dtype=np.float32)
        self.gripper_qvel = np.zeros(2, dtype=np.float32)


        #self.have_base = False
        #TODO: ACTUALY IMPLEMENT ARM
        #self.have_arm = True
        self.have_odom = False
        self.have_pos = False
        # Timestamp of the most recent base-state message (used as the
        # plan's header.stamp so consumers can compute end-to-end latency).
        self.last_base_stamp = None

        self.last_odom = None
        self.last_local_pos = None
        # Sensor-style QoS: keep only the latest sample, best-effort delivery.
        # Bounds staleness of last_base_stamp to one message instead of the
        # default depth=10 queue (which can hold 200ms of stale state at 50Hz).
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        #self.create_subscription(
            #Odometry, BASE_STATE_TOPIC, self._base_cb, sensor_qos,
        #)
        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_odometry', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_local_position_v1', self.local_pos_cb, qos_profile_sensor_data)

        #TODO: ACTUALLY IMPLEMENT ARM
        #self.create_subscription(
        #    JointState, ARM_STATE_TOPIC, self._arm_cb, sensor_qos,
        #)

        # ── Publisher ─────────────────────────────────────────────────────
        self.plan_pub = self.create_publisher(JointTrajectory, PLAN_TOPIC, 10)

        # ── Timer ─────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / plan_rate, self._plan_and_publish)

        self.get_logger().info(
            f"MPPI planner ready. Replanning at {plan_rate:.1f} Hz, "
            f"publishing on {PLAN_TOPIC}, horizon "
            f"{self.horizon_steps} steps ({plan_horizon:.2f}s)."
        )

    # ── Callbacks ─────────────────────────────────────────────────────────
   # def _base_cb(self, msg: Odometry):
   #     # Pose is in header.frame_id (world / odom). Twist is in
   #     # child_frame_id (body). The MuJoCo task expects qvel[:3] in world
   #     # frame, so rotate the linear twist by the current yaw.
   #     self.base_qpos[0] = msg.pose.pose.position.x
   #     self.base_qpos[1] = msg.pose.pose.position.y

   #     q = msg.pose.pose.orientation
   #     yaw = math.atan2(
   #         2.0 * (q.w * q.z + q.x * q.y),
   #         1.0 - 2.0 * (q.y * q.y + q.z * q.z),
   #     )
   #     self.base_qpos[2] = yaw

   #     vx_b = msg.twist.twist.linear.x
   #     vy_b = msg.twist.twist.linear.y
   #     c, s = math.cos(yaw), math.sin(yaw)
   #     self.base_qvel[0] = vx_b * c - vy_b * s
   #     self.base_qvel[1] = vx_b * s + vy_b * c
   #     self.base_qvel[2] = msg.twist.twist.angular.z

   #     self.last_base_stamp = msg.header.stamp
   #     self.have_base = True
# TODO: ACTUALLY DO ARM_CB
    #def _arm_cb(self, msg: JointState):
        #name_to_idx = {n: i for i, n in enumerate(msg.name)}
        #for j, name in enumerate(self.arm_joint_names):
            #i = name_to_idx.get(name)
            #if i is None:
                #continue
            #if i < len(msg.position):
                #self.arm_qpos[j] = float(msg.position[i])
            #if i < len(msg.velocity):
                #self.arm_qvel[j] = float(msg.velocity[i])
        #self.have_arm = True

    def odom_cb(self, msg: Float64MultiArray):
        self.last_odom = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_odom = True

    def local_pos_cb(self, msg: Float64MultiArray):
        self.last_local_pos = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_pos = True

    # ── Plan + publish ────────────────────────────────────────────────────
    def _assemble_state(self):

        # Start from the model default pose so arm/gripper qpos aren't zero
        # (which can be an unstable configuration). Only the base slots
        # [0:3] are overwritten from real sensor data here; once the arm
        # is wired, the arm callback will overwrite [3:9].
        qpos = self.qpos0.copy()
        qvel = np.zeros(self.nv, dtype=np.float32)
        #qpos[:3] = self.base_qpos
        #qpos[3:9] = self.arm_qpos
        #qpos[9:11] = self.gripper_qpos
        #qvel[:3] = self.base_qvel
        #qvel[3:9] = self.arm_qvel
        #qvel[9:11] = self.gripper_qvel
        #return qpos, qvel
        odom_msg, odom_t = self.last_odom
        local_pos_msg, _ = self.last_local_pos
        #qpos[0] = float(odom_msg.data[3])
        #qpos[1] = float(odom_msg.data[4])
        qpos[2] = float(local_pos_msg.data[LOCAL_POS_HEADING_INDEX]-np.pi/2)
        qpos[0] = float(odom_msg.data[4])
        qpos[1] = float(odom_msg.data[3])
        #qvel[0] = float(odom_msg.data[ODOM_LIN_VEL_START])
        #qvel[1] = float(odom_msg.data[ODOM_LIN_VEL_START + 1])
        qvel[2] = float(odom_msg.data[BASE_VEL_YAW_RATE_INDEX])
        qvel[0] = float(odom_msg.data[ODOM_LIN_VEL_START + 1])
        qvel[1] = float(odom_msg.data[ODOM_LIN_VEL_START])
        self.last_base_stamp = RclTime(nanoseconds=int(odom_t * 1e9)).to_msg()
        self.get_logger().info(f'Received pos_x={float(odom_msg.data[3]):.3f} pos_y={float(odom_msg.data[4]):.3f} yaw!={float(local_pos_msg.data[LOCAL_POS_HEADING_INDEX]-np.pi/2)}')
        return qpos, qvel
       # qpos[0] = body_x
       # qpos[1] = body_y



    def _plan_and_publish(self):
        #if not (self.have_base and self.have_arm):
        if not (self.have_odom and self.have_pos):
            self.get_logger().warn(
                f"Waiting for state... odom={self.have_odom} pos={self.have_pos}",
                throttle_duration_sec=2.0,
            )
            return

        qpos, qvel = self._assemble_state()
        # Use seconds-since-node-start, not Unix epoch seconds, as the time
        # fed to the cost function. The circle cost evaluates cos(B*t) /
        # sin(B*t); with epoch seconds (~1.78e9) and float32, that's noise.
        if not hasattr(self, "_t0_wall"):
            self._t0_wall = self.get_clock().now().nanoseconds * 1e-9
        now_t = self.get_clock().now().nanoseconds * 1e-9 - self._t0_wall

        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=jnp.array(now_t, dtype=jnp.float32),
        )

        self.policy_params, _ = self.jit_optimize(
            self.mjx_data, self.policy_params, self.integral
        )

        tq = jnp.arange(0, self.horizon_steps) * self.sim_dt + self.mjx_data.time
        us = np.asarray(
            self.jit_interp(
                tq, self.policy_params.tk, self.policy_params.mean[None, ...]
            )
        )[0]

        msg = JointTrajectory()
        # Use the timestamp of the state this plan was computed from, so
        # downstream consumers can measure full state-to-plan latency.
        msg.header.stamp = (
            self.last_base_stamp
            if self.last_base_stamp is not None
            else self.get_clock().now().to_msg()
        )
        msg.header.frame_id = "atmos_m3_mppi"
        msg.joint_names = list(CONTROL_NAMES[: self.nu])

        for i in range(self.horizon_steps):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in us[i]]
            t = (i + 1) * self.sim_dt
            sec = int(t)
            nsec = int(round((t - sec) * 1e9))
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(pt)

        self.plan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MppiPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
