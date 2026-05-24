"""ROS 2 trajectory-follower node for the AtmosM3 MPPI loop.

Sits between `mppi_ros_node.py` and the real drone:
  - Subscribes to the MPPI plan (JointTrajectory).
  - At a configurable high rate, picks the right slice of the latest plan
    based on `now - plan.header.stamp` and publishes the resulting command.
  - Subscribes to robot state purely for staleness watchdogs (it does not
    feed state back to the planner — the planner reads state directly).

Safety layer (the whole point of this node):
  - Plan-staleness watchdog: no new plan within `plan_timeout_s` → safe stop.
  - State-staleness watchdog: same for Odometry + JointState inputs.
  - Hard saturation on every output channel.
  - Slew-rate limit on base velocity to absorb plan-to-plan setpoint jumps.

Topic / joint names are placeholders (replace with the real wiring).

Run:
  python3 trajectory_follower.py
"""

import threading
import time as _time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ── Topic placeholders ──────────────────────────────────────────────────
# Inbound
PLAN_TOPIC       = "/mppi/plan"
#BASE_STATE_TOPIC = "/TODO/base/odom"
#ARM_STATE_TOPIC  = "/TODO/arm/joint_states"
# Outbound (to drone / arm vendor controllers)
#BASE_CMD_TOPIC    = "/TODO/base/cmd_vel"
#ARM_CMD_TOPIC     = "/TODO/arm/cmd"
#GRIPPER_CMD_TOPIC = "/TODO/gripper/cmd"

ARM_JOINT_NAMES = [
    "TODO_joint_0", "TODO_joint_1", "TODO_joint_2",
    "TODO_joint_3", "TODO_joint_4", "TODO_joint_5",
]

# AtmosM3 ctrl layout:
#   0..2  base velocity (vx_body, vy_body, wz_body)
#   3..8  arm joint commands
#   9     gripper command (0=closed, 1=open)
#   10    dead channel (ignored)
NU = 11
#NOTE THAT MPPI DOES BODY VELOCITY
# FROM TROSSEN_CLIENT_ROS.py
ODOM_LIN_VEL_START = 10
ODOM_ANG_VEL_START = 13
LOCAL_POS_HEADING_INDEX = 14
BASE_VEL_LIN_X_INDEX = 3
BASE_VEL_LIN_Y_INDEX = 4
BASE_VEL_YAW_RATE_INDEX = 15



class MPPIDriver(Node):
    def __init__(self):
        super().__init__("trajectory_follower")

        # Output rate
        self.declare_parameter("publish_rate_hz", 50.0)
        # Watchdog timeouts
        self.declare_parameter("plan_timeout_s", 0.2)
        self.declare_parameter("state_timeout_s", 0.2)
        # Slew limits on base velocity command
        self.declare_parameter("max_linear_accel", 2.0)    # m/s^2
        self.declare_parameter("max_angular_accel", 4.0)   # rad/s^2
        # Hard saturation (defense in depth on top of planner u_min/u_max)
        self.declare_parameter("max_linear_vel", 0.3)      # m/s
        self.declare_parameter("max_angular_vel", 0.4)     # rad/s
        # Arm joint limits (placeholder — set per-joint values for real robot)
        self.declare_parameter("max_arm_pos", 3.14)
        self.declare_parameter("min_arm_pos", -3.14)

        self.publish_rate  = float(self.get_parameter("publish_rate_hz").value)
        self.plan_timeout  = float(self.get_parameter("plan_timeout_s").value)
        self.state_timeout = float(self.get_parameter("state_timeout_s").value)
        self.max_lin_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_ang_accel = float(self.get_parameter("max_angular_accel").value)
        self.max_lin_vel   = float(self.get_parameter("max_linear_vel").value)
        self.max_ang_vel   = float(self.get_parameter("max_angular_vel").value)
        self.max_arm       = float(self.get_parameter("max_arm_pos").value)
        self.min_arm       = float(self.get_parameter("min_arm_pos").value)

        # ── Shared state, lock-protected ──────────────────────────────────
        self._lock = threading.Lock()
        self.plan_times    = None        # (N,) seconds-from-state-stamp
        self.plan_ctrls    = None        # (N, NU)
        self.plan_state_t  = None        # ros sec (the planner's input stamp)
        self.last_plan_recv = None       # ros sec
        self.last_base_recv = None
        self.last_arm_recv  = None
        # Slew + hold state
        self.prev_base_cmd    = np.zeros(3, dtype=np.float64)
        self.prev_arm_cmd     = np.zeros(6, dtype=np.float64)
        self.prev_gripper_cmd = 0.0
        # Counters
        self._n_plans      = 0
        self._n_safe_stops = 0
        self._n_ticks      = 0

        self.last_odom = None
        self.last_local_pos = None

        # ── ROS plumbing ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointTrajectory, PLAN_TOPIC, self._plan_cb, sensor_qos,
        )
        #self.create_subscription(
        #    Odometry, BASE_STATE_TOPIC, self._base_cb, sensor_qos,
        #)
        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_odometry', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_local_position_v1', self.local_pos_cb, qos_profile_sensor_data)

        #TODO: Real arm subscription
        #self.create_subscription(
            #JointState, ARM_STATE_TOPIC, self._arm_cb, sensor_qos,
        #)

        #self.base_cmd_pub = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)
        self.base_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/vla_base/traj_setpoint',
            qos_profile_sensor_data
        )
        #TODO: DO ARM
        #self.arm_cmd_pub  = self.create_publisher(JointTrajectory, ARM_CMD_TOPIC, 10)
        #self.grip_cmd_pub = self.create_publisher(Float32, GRIPPER_CMD_TOPIC, 10)

        # ── High-rate tick thread ─────────────────────────────────────────
        self._stop = threading.Event()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="follower_tick", daemon=True,
        )
        self._tick_thread.start()

        # Stats log every second
        self.create_timer(1.0, self._log_stats)

        self.get_logger().info(
            f"Trajectory follower ready. publish={self.publish_rate:.0f}Hz, "
            f"plan_timeout={self.plan_timeout*1000:.0f}ms, "
            f"state_timeout={self.state_timeout*1000:.0f}ms"
        )

    # ── Helpers ──────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ── Subscriber callbacks ─────────────────────────────────────────────
    def _plan_cb(self, msg: JointTrajectory):
        n = len(msg.points)
        if n == 0:
            return
        times = np.empty(n, dtype=np.float64)
        ctrls = np.empty((n, NU), dtype=np.float64)
        for i, pt in enumerate(msg.points):
            times[i] = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            ctrls[i, :] = pt.positions[:NU]
        state_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        recv_t = self._now()
        with self._lock:
            self.plan_times    = times
            self.plan_ctrls    = ctrls
            self.plan_state_t  = state_t
            self.last_plan_recv = recv_t
            self._n_plans += 1

  #  def _base_cb(self, msg: Odometry):
  #      with self._lock:
  #          self.last_base_recv = self._now()

    def odom_cb(self, msg: Float64MultiArray):
        t = self._now()
        self.last_odom = (msg, t)
        # Feeds the base-staleness watchdog in _tick — odom is the more
        # frequent of the two state streams, so use it as the heartbeat.
        with self._lock:
            self.last_base_recv = t

    def local_pos_cb(self, msg: Float64MultiArray):
        self.last_local_pos = (msg, self._now())

    def publish_base_cmd(self, base_action: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(base_action[0]), float(base_action[1]), float(base_action[2])]
        self.base_cmd_pub.publish(msg)

    def _arm_cb(self, msg: JointState):
        with self._lock:
            self.last_arm_recv = self._now()

    # ── High-rate loop (separate thread) ─────────────────────────────────
    def _tick_loop(self):
        period = 1.0 / self.publish_rate
        next_t = _time.monotonic()
        while not self._stop.is_set():
            self._tick()
            next_t += period
            sleep = next_t - _time.monotonic()
            if sleep > 0:
                _time.sleep(sleep)
            else:
                # Behind schedule — reset reference instead of accumulating lag.
                next_t = _time.monotonic()

    def _tick(self):
        now = self._now()
        with self._lock:
            plan_age = (
                now - self.last_plan_recv
                if self.last_plan_recv is not None else float("inf")
            )
            base_age = (
                now - self.last_base_recv
                if self.last_base_recv is not None else float("inf")
            )
            plan_times   = self.plan_times
            plan_ctrls   = self.plan_ctrls
            plan_state_t = self.plan_state_t

        self._n_ticks += 1

        # Cheap watchdogs first — these handle the "no plan yet" / "no state
        # yet" cases before we try to subtract None below. Arm watchdog is
        # disabled until the arm subscription is wired up; re-add when it is.
        if (
            plan_ctrls is None
            or plan_age > self.plan_timeout
            or base_age > self.state_timeout
        ):
            self._publish_safe_stop()
            self._n_safe_stops += 1
            return

        # Past here, plan_state_t and plan_times are guaranteed non-None
        # (they're set atomically with plan_ctrls under _lock).
        elapsed = now - plan_state_t
        if elapsed > plan_times[-1]:
            self._publish_safe_stop()
            self._n_safe_stops += 1
            return

        # Pick the plan slice that corresponds to *right now* in the state's
        # timeline (header.stamp == the Odometry stamp the planner used).
        idx = int(np.searchsorted(plan_times, elapsed))
        idx = min(idx, len(plan_ctrls) - 1)
        u = plan_ctrls[idx]
        self._publish_cmd(u)

    # ── Output: apply limits + publish ───────────────────────────────────
    def _publish_cmd(self, u: np.ndarray):
        dt = 1.0 / self.publish_rate

        # Base velocity: hard-clamp then slew-limit.
        #target_base = np.array([
            #np.clip(u[0], -self.max_lin_vel, self.max_lin_vel),
            #np.clip(u[1], -self.max_lin_vel, self.max_lin_vel),
            #np.clip(u[2], -self.max_ang_vel, self.max_ang_vel),
        #], dtype=np.float64)
        target_base = np.array([
            np.clip(u[0], -self.max_lin_vel, self.max_lin_vel),
            np.clip(-1 * u[1], -self.max_lin_vel, self.max_lin_vel),
            np.clip(u[2], -self.max_ang_vel, self.max_ang_vel),
        ], dtype=np.float64)
        max_lin_step = self.max_lin_accel * dt
        max_ang_step = self.max_ang_accel * dt
        base = np.empty(3, dtype=np.float64)
        base[0] = self.prev_base_cmd[0] + np.clip(
            target_base[0] - self.prev_base_cmd[0], -max_lin_step, max_lin_step,
        )
        base[1] = self.prev_base_cmd[1] + np.clip(
            target_base[1] - self.prev_base_cmd[1], -max_lin_step, max_lin_step,
        )
        base[2] = self.prev_base_cmd[2] + np.clip(
            target_base[2] - self.prev_base_cmd[2], -max_ang_step, max_ang_step,
        )
        self.prev_base_cmd = base

        #twist = Twist()
        #twist.linear.x  = float(base[0])
        #twist.linear.y  = float(base[1])
        #twist.linear.z  = 0.0
        #twist.angular.z = float(base[2])
        #self.base_cmd_pub.publish(twist)
        self.publish_base_cmd(base)

        #TODO: ARM!
        # Arm: clamp to joint limits, publish as single-point JointTrajectory.
       # arm = np.clip(
       #     np.array(u[3:9], dtype=np.float64), self.min_arm, self.max_arm,
       # )
       # self.prev_arm_cmd = arm
       # traj = JointTrajectory()
       # traj.header.stamp = self.get_clock().now().to_msg()
       # traj.joint_names = list(ARM_JOINT_NAMES)
       # pt = JointTrajectoryPoint()
       # pt.positions = [float(v) for v in arm]
       # traj.points.append(pt)
       # self.arm_cmd_pub.publish(traj)

       # # Gripper: clamp to [0, 1].
       # grip = float(np.clip(u[9], 0.0, 1.0))
       # self.prev_gripper_cmd = grip
       # self.grip_cmd_pub.publish(Float32(data=grip))

    def _publish_safe_stop(self):
        """Zero base velocity (slew-limited from current), hold arm/gripper."""
        safe_u = np.zeros(NU, dtype=np.float64)
        safe_u[3:9] = self.prev_arm_cmd
        safe_u[9]   = self.prev_gripper_cmd
        self._publish_cmd(safe_u)

    def _log_stats(self):
        with self._lock:
            n_plans = self._n_plans
            n_safe  = self._n_safe_stops
            n_ticks = self._n_ticks
            plan_age = (
                self._now() - self.last_plan_recv
                if self.last_plan_recv is not None else float("inf")
            )
        self.get_logger().info(
            f"ticks={n_ticks} | plans={n_plans} | safe_stops={n_safe} | "
            f"plan_age={plan_age*1000:6.1f}ms"
        )

    def shutdown(self):
        self._stop.set()
        if self._tick_thread.is_alive():
            self._tick_thread.join(timeout=2.0)
        # Final safe-stop so the robot doesn't ride the last cmd.
        self._publish_safe_stop()

    def _keyboard_listener(self):
        logger.info("Emergency stop triggered - moving to sleep poistion...")
        self.is_


def main(args=None):
    rclpy.init(args=args)
    node = MPPIDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
