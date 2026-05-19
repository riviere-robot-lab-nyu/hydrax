"""ROS 2 node that closes the MPPI control loop in MuJoCo, headless.

Pairs with `mppi_ros_node.py` (the planner). This node:
  - Steps a MuJoCo simulation of AtmosM3 in its own thread, paced to wallclock.
  - Subscribes to `PLAN_TOPIC` and applies the latest control trajectory.
  - Publishes Odometry on `BASE_STATE_TOPIC` and JointState on `ARM_STATE_TOPIC`
    so the planner sees fresh feedback.
  - Snapshots qpos/qvel during the run, then re-renders them to mp4 *after*
    the sim stops (rendering live on the Jetson is 10-30ms/frame and would
    starve physics + the plan subscriber).

Open-loop mode (`openloop:=true`): the sim does NOT publish state — the
planner closes the loop with the real robot via the PX4 bridge instead.
The sim still consumes the planner's plans and rolls forward in MuJoCo so
the recorded video shows the model's predicted trajectory under the same
commands. To make the comparison meaningful, the sim seeds its initial
qpos/qvel from the first real-bridge messages it sees (origin + identity
if none arrive within ~1s).

Stops automatically after `duration_sec` (default 60s) and writes the video
on shutdown. Set `duration_sec <= 0` to run until Ctrl-C.

Headless rendering: set `MUJOCO_GL=egl` (or `osmesa`) before launching.

Run alongside the planner:
  Terminal 1:  python3 mppi_ros_node.py
  Terminal 2:  MUJOCO_GL=egl python3 mppi_ros_sim_node.py
"""

import math
import threading
import time as _time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.utils.video import VideoRecorder
from std_msgs.msg import Float64MultiArray
# Single source of truth for topic / joint names — pulled from the planner
#from mppi_ros_node import (
    #ARM_STATE_TOPIC,
    #ARM_JOINT_NAMES,
    #BASE_STATE_TOPIC,
    #PLAN_TOPIC,
#)
PLAN_TOPIC = "/mppi/plan"
ARM_STATE_TOPIC = "/TODO/arm/joint_states"
ARM_JOINT_NAMES = [
    "TODO_joint_0",
    "TODO_joint_1",
    "TODO_joint_2",
    "TODO_joint_3",
    "TODO_joint_4",
    "TODO_joint_5",
]
class MppiSimNode(Node):
    def __init__(self):
        super().__init__("mppi_sim")

        self.declare_parameter("state_pub_rate_hz", 50.0)
        self.declare_parameter("video_fps", 30.0)
        self.declare_parameter("video_width", 720)
        self.declare_parameter("video_height", 480)
        self.declare_parameter("video_dir", "/workspace/hydrax/mppi_videos/")
        self.declare_parameter("record_video", True)
        self.declare_parameter("realtime", True)
        self.declare_parameter("camera_id", -1)        # -1 = follow robot
        self.declare_parameter("cam_distance", 4.0)
        self.declare_parameter("cam_azimuth", 90.0)
        self.declare_parameter("cam_elevation", -20.0)
        self.declare_parameter("cam_lookat_z", 0.5)
        self.declare_parameter("duration_sec", 60.0)   # 0 or negative = run forever
        self.declare_parameter("openloop", False)

        state_pub_rate = float(self.get_parameter("state_pub_rate_hz").value)
        video_fps = float(self.get_parameter("video_fps").value)
        video_w = int(self.get_parameter("video_width").value)
        video_h = int(self.get_parameter("video_height").value)
        video_dir = str(self.get_parameter("video_dir").value)
        record_video = bool(self.get_parameter("record_video").value)
        self.realtime = bool(self.get_parameter("realtime").value)
        camera_id = int(self.get_parameter("camera_id").value)
        self.cam_distance = float(self.get_parameter("cam_distance").value)
        self.cam_azimuth = float(self.get_parameter("cam_azimuth").value)
        self.cam_elevation = float(self.get_parameter("cam_elevation").value)
        self.cam_lookat_z = float(self.get_parameter("cam_lookat_z").value)
        duration = float(self.get_parameter("duration_sec").value)
        self.duration_sec = duration if duration > 0 else None
        self.openloop = bool(self.get_parameter("openloop").value)

        # ── Model + sim state ─────────────────────────────────────────────
        self.task = AtmosM3()
        self.mj_model = self.task.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sim_dt = float(self.mj_model.opt.timestep)
        self.nu = int(self.mj_model.nu)
        # Expand the model's offscreen framebuffer if the requested video
        # size is larger than what the XML declares (AtmosM3 ships with
        # 640x480). Has to happen before the Renderer is constructed.
        if record_video:
            self.mj_model.vis.global_.offwidth = max(
                int(self.mj_model.vis.global_.offwidth), video_w,
            )
            self.mj_model.vis.global_.offheight = max(
                int(self.mj_model.vis.global_.offheight), video_h,
            )

        # mjx mirror, used to call ctrl_transform_with_integral on the real
        # state at every sim step (matches deterministic_headless.py).
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        self.integral = jnp.zeros(1)
        self.jit_ctrl_transform = jax.jit(self.task.ctrl_transform_with_integral)
        self.jit_update_integral = jax.jit(self.task.update_integral)

        self.get_logger().info("Jitting ctrl_transform / update_integral...")
        t0 = _time.time()
        _ac = self.jit_ctrl_transform(
            self.mjx_data, jnp.zeros(self.nu), self.integral,
        )
        _ = self.jit_update_integral(
            self.mjx_data, jnp.zeros(self.nu), self.integral, _ac,
        )
        jax.block_until_ready(_ac)
        self.get_logger().info(f"JIT done in {_time.time() - t0:.2f}s")

        # ── Latest plan, protected by a lock ──────────────────────────────
        self._plan_lock = threading.Lock()
        self.plan_times = None      # (N,) seconds from plan_state_t (the
                                    # state timestamp the planner used)
        self.plan_ctrls = None      # (N, nu)
        self.plan_state_t = None    # ros wallclock (sec) of the state this
                                    # plan was computed from
        self.last_ctrl = np.zeros(self.nu, dtype=np.float64)

        # ── ROS plumbing ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointTrajectory, PLAN_TOPIC, self._plan_cb, sensor_qos,
        )
        #self.base_pub = self.create_publisher(Odometry, BASE_STATE_TOPIC, 10)
        # Initial-state seeding (openloop mode only). Stashed under a lock
        # because the spin executor delivers the callbacks on a different
        # thread than the sim loop that reads them.
        self._init_lock = threading.Lock()
        self._init_odom_msg = None
        self._init_local_pos_msg = None
        self._initialized = False

        if self.openloop:
            self.arm_pub = None
            self.odom_pub = None
            self.pos_pub = None
            self.create_subscription(
                Float64MultiArray, '/px4/bridge/vehicle_odometry',
                self._init_odom_cb, qos_profile_sensor_data,
            )
            self.create_subscription(
                Float64MultiArray, '/px4/bridge/vehicle_local_position_v1',
                self._init_local_pos_cb, qos_profile_sensor_data,
            )
        else:
            self.arm_pub = self.create_publisher(JointState, ARM_STATE_TOPIC, 10)

            self.odom_pub = self.create_publisher(Float64MultiArray, '/px4/bridge/vehicle_odometry',  10)
            self.pos_pub = self.create_publisher(Float64MultiArray, '/px4/bridge/vehicle_local_position_v1', 10)

        # ── Offline video config (render after the run) ──────────────────
        # We deliberately do NOT render during the sim — rendering on the
        # Jetson is ~10-30ms/frame and would steal time from physics and
        # the plan subscriber. Instead, we snapshot qpos/qvel at video_fps
        # cadence and re-render the run to mp4 when the sim stops.
        self.record_video = record_video
        self.video_w = video_w
        self.video_h = video_h
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.render_cam = camera_id if camera_id >= 0 else None
        # State snapshots: list of (sim_time, qpos, qvel) tuples.
        self._snapshots: list = []

        # ── Sim thread ────────────────────────────────────────────────────
        self.state_period = 1.0 / state_pub_rate
        self.snap_period = 1.0 / video_fps if record_video else None
        self._stop = threading.Event()
        self._wall_t0 = _time.monotonic()
        self._sim_t0 = float(self.mj_data.time)
        self._n_steps = 0
        self._n_plans = 0
        self._sim_thread = threading.Thread(
            target=self._sim_loop, name="mppi_sim_loop", daemon=True,
        )
        self._sim_thread.start()

        self.get_logger().info(
            f"MPPI sim ready. mj_dt={self.sim_dt*1000:.2f}ms ({1.0/self.sim_dt:.0f}Hz), "
            f"nu={self.nu}, state_pub={state_pub_rate:.0f}Hz"
            + (
                f", snapshots @ {video_fps:.0f}fps → mp4 at end ({video_dir})"
                if record_video else ""
            )
            + (
                f", duration={self.duration_sec:.0f}s"
                if self.duration_sec is not None else ", run until Ctrl-C"
            )
        )

    # ── Plan ─────────────────────────────────────────────────────────────
    def _plan_cb(self, msg: JointTrajectory):
        n = len(msg.points)
        if n == 0:
            return
        times = np.empty(n, dtype=np.float64)
        ctrls = np.empty((n, self.nu), dtype=np.float64)
        for i, pt in enumerate(msg.points):
            times[i] = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            # Planner publishes nu+1 entries (last one is the "_dead" channel);
            # the model's actuators only consume the first nu.
            ctrls[i, :] = pt.positions[: self.nu]
        # The plan's time_from_start values are relative to the state timestamp
        # the planner stamped into header.stamp (= the Odometry stamp it
        # planned from). Indexing from there — rather than from plan_recv_t —
        # absorbs the sim→planner→sim round-trip latency into the lookup.
        state_t = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        with self._plan_lock:
            self.plan_times = times
            self.plan_ctrls = ctrls
            self.plan_state_t = state_t
            self._n_plans += 1

    def _current_ctrl(self) -> np.ndarray:
        with self._plan_lock:
            if self.plan_ctrls is None:
                return self.last_ctrl
            elapsed = (
                self.get_clock().now().nanoseconds * 1e-9 - self.plan_state_t
            )
            idx = int(np.searchsorted(self.plan_times, elapsed))
            idx = min(idx, len(self.plan_ctrls) - 1)
            self.last_ctrl = self.plan_ctrls[idx]
            return self.last_ctrl

    # ── Sim loop (own thread) ────────────────────────────────────────────
    def _sim_loop(self):
        # In openloop mode, wait briefly for the first real-bridge messages
        # so we start the rollout from the robot's actual pose. Anything
        # past ~1s and we fall through with model defaults — comparison
        # will be muddled by the initial-state offset, but better than
        # blocking indefinitely on a topic that may never appear.
        if self.openloop:
            seed_deadline = _time.monotonic() + 1.0
            while _time.monotonic() < seed_deadline and not self._stop.is_set():
                if self._try_seed_initial_state():
                    break
                _time.sleep(0.05)
            if not self._initialized:
                self.get_logger().warn(
                    "Openloop: no initial state from /px4/bridge within 1s — "
                    "starting from model defaults (origin, identity, zero vel)."
                )
            # Reset the wallclock anchor so the seeding wait doesn't show up
            # as a real-time deficit the sim then tries to "catch up" on.
            self._wall_t0 = _time.monotonic()

        next_state_t = 0.0
        next_snap_t = 0.0
        next_log_t = 1.0
        while not self._stop.is_set():
            sim_elapsed = float(self.mj_data.time) - self._sim_t0
            wall_elapsed = _time.monotonic() - self._wall_t0

            # Stop when the configured duration has elapsed.
            if self.duration_sec is not None and sim_elapsed >= self.duration_sec:
                self.get_logger().info(
                    f"Reached duration {self.duration_sec:.1f}s — stopping sim."
                )
                self._stop.set()
                # Signal the main thread to break out of rclpy.spin().
                try:
                    rclpy.try_shutdown()
                except Exception:
                    pass
                break

            # Real-time pacing: if we're ahead of wallclock, sleep briefly.
            if self.realtime and sim_elapsed > wall_elapsed:
                _time.sleep(min(sim_elapsed - wall_elapsed, 0.005))
                continue

            # Pull the latest plan command and transform it for the actuators.
            u = jnp.asarray(self._current_ctrl())
            self.mjx_data = self.mjx_data.replace(
                qpos=jnp.asarray(self.mj_data.qpos),
                qvel=jnp.asarray(self.mj_data.qvel),
            )
            actual_ctrl = self.jit_ctrl_transform(self.mjx_data, u, self.integral)
            self.integral = self.jit_update_integral(
                self.mjx_data, u, self.integral, actual_ctrl,
            )
            self.mj_data.ctrl[:] = np.asarray(actual_ctrl)

            mujoco.mj_step(self.mj_model, self.mj_data)
            self._n_steps += 1

            # Periodic outputs (sim-time-driven so they slow down if sim does)
            if sim_elapsed >= next_state_t:
                self._publish_state()
                next_state_t += self.state_period
            if self.snap_period is not None and sim_elapsed >= next_snap_t:
                self._snapshots.append(
                    (sim_elapsed, self.mj_data.qpos.copy(), self.mj_data.qvel.copy())
                )
                next_snap_t += self.snap_period
            if sim_elapsed >= next_log_t:
                self._log_stats(sim_elapsed, wall_elapsed)
                next_log_t += 1.0

    # ── State pub ────────────────────────────────────────────────────────
    def _publish_state(self):
        """
        /px4/bridge/vehicle_local_position_v1  (16 floats):
    [timestamp_us,
     xy_valid, z_valid, v_xy_valid, v_z_valid,
     x, y, z,
     vx, vy, vz,
     ax, ay, az,
     heading, heading_var]

  /px4/bridge/vehicle_odometry  (25 floats):
    [timestamp_us, pose_frame, velocity_frame,
     px, py, pz,
     qw, qx, qy, qz,
     vx, vy, vz,
     wx, wy, wz,
     pos_var_x, pos_var_y, pos_var_z,
     ori_var_r, ori_var_p, ori_var_y,
     vel_var_x, vel_var_y, vel_var_z]"""
        if self.openloop:
            return
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        x, y, yaw = float(qpos[0]), float(qpos[1]), float(qpos[2])

        odom = Float64MultiArray()
        odom.data = [float(0.0), float(0.0), float(0.0),
                     x, y, float(0.0), float(0.0), float(0.0), float(0.0), float(0.0), 
                     float(qvel[0]), float(qvel[1]), float(0.0), 
                     float(0.0), float(0.0), float(qvel[2])]
        local_pos = Float64MultiArray()
        local_pos.data = [
            float(0.0),                                     # 0: timestamp_us
            float(0.0), float(0.0), float(0.0), float(0.0), # 1-4: xy/z/v_xy/v_z valid
            float(0.0), float(0.0), float(0.0),             # 5-7: x, y, z
            float(0.0), float(0.0), float(0.0),             # 8-10: vx, vy, vz
            float(0.0), float(0.0), float(0.0),             # 11-13: ax, ay, az
            yaw, float(0.0)                                 # 14-15: heading, heading_var
        ]

        #odom.header.stamp = self.get_clock().now().to_msg()
        #odom.header.frame_id = "odom"
        #odom.child_frame_id = "base_link"
        #odom.pose.pose.position.x = x
        #odom.pose.pose.position.y = y
        #odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        #odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        # qvel[:2] is world-frame in the planner's convention; the planner
        # rotates Odometry's body-frame twist by yaw to get back to world,
        # so we apply the inverse rotation here.
        #vx_w, vy_w = float(qvel[0]), float(qvel[1])
        #c, s = math.cos(yaw), math.sin(yaw)
        #odom.twist.twist.linear.x = vx_w * c + vy_w * s
        #odom.twist.twist.linear.y = -vx_w * s + vy_w * c
        #odom.twist.twist.angular.z = float(qvel[2])
        #self.base_pub.publish(odom)
        self.odom_pub.publish(odom)
        self.pos_pub.publish(local_pos)

        arm = JointState()
        arm.header.stamp = self.get_clock().now().to_msg()
        arm.name = list(ARM_JOINT_NAMES)
        arm.position = [float(v) for v in qpos[3:9]]
        arm.velocity = [float(v) for v in qvel[3:9]]
        self.arm_pub.publish(arm)

    # ── Openloop initial-state seeding ───────────────────────────────────
    # Callback layout must match what the planner reads in _assemble_state:
    #   odom.data[3] = px, data[4] = py, data[10:13] = vx,vy,vz,
    #   data[13:16] = wx,wy,wz
    #   local_pos.data[14] = heading (yaw)
    def _init_odom_cb(self, msg: Float64MultiArray):
        with self._init_lock:
            if self._init_odom_msg is None:
                self._init_odom_msg = msg

    def _init_local_pos_cb(self, msg: Float64MultiArray):
        with self._init_lock:
            if self._init_local_pos_msg is None:
                self._init_local_pos_msg = msg

    def _try_seed_initial_state(self) -> bool:
        if self._initialized:
            return True
        with self._init_lock:
            odom = self._init_odom_msg
            pos = self._init_local_pos_msg
        if odom is None or pos is None:
            return False
        self.mj_data.qpos[0] = float(odom.data[3])
        self.mj_data.qpos[1] = float(odom.data[4])
        self.mj_data.qpos[2] = float(pos.data[14])
        self.mj_data.qvel[0] = float(odom.data[10])
        self.mj_data.qvel[1] = float(odom.data[11])
        self.mj_data.qvel[2] = float(odom.data[15])
        # Propagate qpos/qvel through dependent state (xpos, contacts, etc.)
        # before the first mj_step reads them.
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._initialized = True
        self.get_logger().info(
            f"Openloop seeded from real bridge: "
            f"x={self.mj_data.qpos[0]:.2f}, y={self.mj_data.qpos[1]:.2f}, "
            f"yaw={self.mj_data.qpos[2]:.2f}"
        )
        return True

    def _log_stats(self, sim_elapsed: float, wall_elapsed: float):
        rtr = sim_elapsed / max(wall_elapsed, 1e-6)
        self.get_logger().info(
            f"sim_t={sim_elapsed:6.2f}s | rtr={rtr:.2f}x | "
            f"plans={self._n_plans} | steps={self._n_steps} | "
            f"snaps={len(self._snapshots)}"
        )

    # ── Offline render: turn saved (qpos, qvel) into mp4 ─────────────────
    def _render_video(self):
        if not self.record_video or not self._snapshots:
            return
        try:
            renderer = mujoco.Renderer(
                self.mj_model, height=self.video_h, width=self.video_w,
            )
        except Exception as e:
            self.get_logger().warn(
                f"Skipping video — renderer init failed: {e}. "
                f"Try `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`."
            )
            return

        recorder = VideoRecorder(
            self.video_dir,
            width=self.video_w,
            height=self.video_h,
            fps=self.video_fps,
        )
        if not recorder.start():
            return

        # Free tracking camera, lookat updated to the robot base each frame.
        # If the user specified an XML-defined camera (camera_id >= 0), use
        # that instead and leave its behavior to the XML.
        track_cam = None
        if self.render_cam is None:
            track_cam = mujoco.MjvCamera()
            track_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            track_cam.distance = self.cam_distance
            track_cam.azimuth = self.cam_azimuth
            track_cam.elevation = self.cam_elevation

        n = len(self._snapshots)
        self.get_logger().info(f"Rendering {n} frames to mp4 ...")
        t0 = _time.time()
        scratch = mujoco.MjData(self.mj_model)
        for i, (_, qpos, qvel) in enumerate(self._snapshots):
            scratch.qpos[:] = qpos
            scratch.qvel[:] = qvel
            # mj_forward updates kinematics + derived quantities used by the
            # scene; cheaper than mj_step and doesn't advance time.
            mujoco.mj_forward(self.mj_model, scratch)
            if track_cam is not None:
                track_cam.lookat[0] = float(qpos[0])
                track_cam.lookat[1] = float(qpos[1])
                track_cam.lookat[2] = self.cam_lookat_z
                renderer.update_scene(scratch, camera=track_cam)
            else:
                renderer.update_scene(scratch, camera=self.render_cam)
            frame = renderer.render()
            recorder.add_frame(frame.tobytes())
            if (i + 1) % max(n // 10, 1) == 0:
                self.get_logger().info(
                    f"  ... {i + 1}/{n} frames ({100.0 * (i + 1) / n:.0f}%)"
                )
        recorder.stop()
        self.get_logger().info(
            f"Wrote {n} frames in {_time.time() - t0:.1f}s"
        )

    def shutdown(self):
        self._stop.set()
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=2.0)
        self._render_video()


def main(args=None):
    rclpy.init(args=args)
    node = MppiSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        # Sim thread may already have called try_shutdown() when duration
        # elapsed; use try_shutdown() here to avoid the double-shutdown error.
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
