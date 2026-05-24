"""Unified mock bridge for end-to-end testing of the planners.

Publishes everything the planners (mppi_ros_node.py / mppi_vla_node.py)
need to come out of "Waiting for state..." gating, and listens to /mppi/plan
to print one-line summaries of what comes back.

Topics published:
  - /px4/bridge/vehicle_odometry        std_msgs/Float64MultiArray (25 floats)
  - /px4/bridge/vehicle_local_position_v1  std_msgs/Float64MultiArray (16 floats)
  - /vla/chunk                          trajectory_msgs/JointTrajectory

Topics subscribed:
  - /mppi/plan                          trajectory_msgs/JointTrajectory

The fake base motion (`state_mode`) and the VLA chunk content (`vla_mode`)
are chosen via ROS params at launch — no code edits needed to flip between
tests. Set `publish_vla:=false` to test the vanilla planner without VLA.

Run:
  Terminal 1:  python3 mppi_vla_node.py --ros-args -p cost_mode:=vla_track
  Terminal 2:  python3 mock_bridge.py --ros-args -p state_mode:=circle \
                                                 -p vla_mode:=constant_forward
"""

import math
import time as _time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Must match the indices the planners read out of the Float64MultiArray
# topics — single source of truth here so a layout change shows up in one
# place. Cross-check against mppi_vla_node.py constants of the same name.
ODOM_LEN = 25
ODOM_POS_X_IDX = 3            # planner reads qpos[1] = odom[3]
ODOM_POS_Y_IDX = 4            # planner reads qpos[0] = odom[4]
ODOM_LIN_VEL_START = 10       # planner reads qvel[1] = odom[10], qvel[0] = odom[11]
ODOM_YAW_RATE_IDX = 15        # planner reads qvel[2] = odom[15]

LOCAL_POS_LEN = 16
LOCAL_POS_HEADING_IDX = 14    # planner reads qpos[2] = local_pos[14] - pi/2

# 11-D control vector layout (see README §4).
NU = 11
VX_IDX, VY_IDX, WZ_IDX = 0, 1, 2


class MockBridge(Node):
    def __init__(self):
        super().__init__("mock_bridge")

        # ── State-side params ─────────────────────────────────────────────
        # State publish rate (matches PX4 bridge nominal).
        self.declare_parameter("state_pub_rate_hz", 50.0)
        # How the fake base moves.
        #   "static": parked at origin, zero velocity.
        #   "circle": orbits at (radius, omega).
        #   "drift" : straight-line drift at +x_dot_drift.
        self.declare_parameter("state_mode", "static")
        self.declare_parameter("circle_radius_m", 0.5)
        self.declare_parameter("circle_omega_rad_s", 0.2)
        self.declare_parameter("drift_vx_m_s", 0.05)

        # ── VLA-side params ───────────────────────────────────────────────
        self.declare_parameter("publish_vla", True)
        self.declare_parameter("vla_topic", "/vla/chunk")
        self.declare_parameter("vla_pub_rate_hz", 1.5)
        self.declare_parameter("vla_chunk_horizon_s", 2.0)
        self.declare_parameter("vla_chunk_dt_s", 0.05)        # 20 Hz inside chunk
        # What VLA controls to publish in each chunk:
        #   "zero"             : all zeros.
        #   "constant_forward" : vx = 0.1 m/s for the entire chunk.
        #   "step"             : vx = 0 for first half, vx = 0.3 for second half.
        #   "sinusoid"         : vx = vla_sin_amp * sin(2π t / chunk_horizon).
        #   "yaw_spin"         : wz = 0.3 rad/s constant.
        self.declare_parameter("vla_mode", "constant_forward")
        self.declare_parameter("vla_sin_amp", 0.2)

        # ── Plan listener params ──────────────────────────────────────────
        self.declare_parameter("plan_topic", "/mppi/plan")
        self.declare_parameter("print_plans", True)

        state_rate = float(self.get_parameter("state_pub_rate_hz").value)
        self.state_mode = str(self.get_parameter("state_mode").value)
        self.R = float(self.get_parameter("circle_radius_m").value)
        self.omega = float(self.get_parameter("circle_omega_rad_s").value)
        self.drift_vx = float(self.get_parameter("drift_vx_m_s").value)

        self.publish_vla = bool(self.get_parameter("publish_vla").value)
        vla_topic = str(self.get_parameter("vla_topic").value)
        vla_rate = float(self.get_parameter("vla_pub_rate_hz").value)
        self.vla_horizon = float(self.get_parameter("vla_chunk_horizon_s").value)
        self.vla_dt = float(self.get_parameter("vla_chunk_dt_s").value)
        self.vla_mode = str(self.get_parameter("vla_mode").value)
        self.vla_sin_amp = float(self.get_parameter("vla_sin_amp").value)
        self.vla_n_pts = max(int(round(self.vla_horizon / self.vla_dt)), 1)

        plan_topic = str(self.get_parameter("plan_topic").value)
        self.print_plans = bool(self.get_parameter("print_plans").value)

        # ── Publishers / subscribers ──────────────────────────────────────
        self.odom_pub = self.create_publisher(
            Float64MultiArray, "/px4/bridge/vehicle_odometry", 10
        )
        self.pos_pub = self.create_publisher(
            Float64MultiArray, "/px4/bridge/vehicle_local_position_v1", 10
        )
        if self.publish_vla:
            self.vla_pub = self.create_publisher(
                JointTrajectory, vla_topic, 10
            )

        self.create_subscription(
            JointTrajectory, plan_topic, self._plan_cb, qos_profile_sensor_data
        )

        # ── Timers ────────────────────────────────────────────────────────
        self._t0 = _time.time()
        self._last_plan_wall = None
        self._plan_count = 0
        self.create_timer(1.0 / state_rate, self._tick_state)
        if self.publish_vla:
            self.create_timer(1.0 / vla_rate, self._tick_vla)

        self.get_logger().info(
            f"mock_bridge ready. state_mode={self.state_mode} "
            f"vla={'on' if self.publish_vla else 'off'}"
            + (f" vla_mode={self.vla_mode} chunk={self.vla_n_pts}pts@{self.vla_dt}s"
               if self.publish_vla else "")
        )

    # ── Fake state ────────────────────────────────────────────────────────
    def _fake_base(self, t: float):
        """Return (x, y, yaw, vx_world, vy_world, wz) for time t (seconds)."""
        if self.state_mode == "circle":
            c, s = math.cos(self.omega * t), math.sin(self.omega * t)
            x = self.R * c
            y = self.R * s
            yaw = self.omega * t
            vx = -self.R * self.omega * s
            vy =  self.R * self.omega * c
            wz = self.omega
            return x, y, yaw, vx, vy, wz
        if self.state_mode == "drift":
            return self.drift_vx * t, 0.0, 0.0, self.drift_vx, 0.0, 0.0
        # "static" / unknown
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _tick_state(self):
        t = _time.time() - self._t0
        x, y, yaw, vx, vy, wz = self._fake_base(t)
        # Wrap yaw to (-pi, pi] like a real PX4 stream so downstream logs
        # don't show unbounded values after long runs.
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        odom = np.zeros(ODOM_LEN, dtype=np.float64)
        odom[ODOM_POS_X_IDX] = y    # NB: planner does qpos[1] = odom[3]
        odom[ODOM_POS_Y_IDX] = x    # and qpos[0] = odom[4]
        odom[ODOM_LIN_VEL_START]     = vy
        odom[ODOM_LIN_VEL_START + 1] = vx
        odom[ODOM_YAW_RATE_IDX] = wz

        local_pos = np.zeros(LOCAL_POS_LEN, dtype=np.float64)
        # Planner does qpos[2] = local_pos[14] - pi/2, so add pi/2 here.
        local_pos[LOCAL_POS_HEADING_IDX] = yaw + math.pi / 2.0

        odom_msg = Float64MultiArray(); odom_msg.data = odom.tolist()
        pos_msg  = Float64MultiArray(); pos_msg.data  = local_pos.tolist()
        self.odom_pub.publish(odom_msg)
        self.pos_pub.publish(pos_msg)

    # ── Fake VLA chunk ────────────────────────────────────────────────────
    def _vla_chunk_controls(self) -> np.ndarray:
        """(vla_n_pts, NU) chunk of controls for the configured mode."""
        u = np.zeros((self.vla_n_pts, NU), dtype=np.float32)
        if self.vla_mode == "zero":
            return u
        if self.vla_mode == "constant_forward":
            u[:, VX_IDX] = 0.1
            return u
        if self.vla_mode == "step":
            half = self.vla_n_pts // 2
            u[half:, VX_IDX] = 0.3
            return u
        if self.vla_mode == "sinusoid":
            tau = np.arange(self.vla_n_pts) * self.vla_dt
            u[:, VX_IDX] = self.vla_sin_amp * np.sin(
                2.0 * math.pi * tau / max(self.vla_horizon, 1e-3)
            )
            return u
        if self.vla_mode == "yaw_spin":
            u[:, WZ_IDX] = 0.3
            return u
        self.get_logger().warn(
            f"Unknown vla_mode={self.vla_mode!r}; falling back to zero."
        )
        return u

    def _tick_vla(self):
        ctrls = self._vla_chunk_controls()
        now = self.get_clock().now().to_msg()

        msg = JointTrajectory()
        msg.header.stamp = now
        msg.header.frame_id = "mock_vla"
        msg.joint_names = [f"u{i}" for i in range(NU)]
        for i in range(self.vla_n_pts):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in ctrls[i]]
            t = (i + 1) * self.vla_dt
            sec = int(t)
            nsec = int(round((t - sec) * 1e9))
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(pt)
        self.vla_pub.publish(msg)

    # ── Plan listener ─────────────────────────────────────────────────────
    def _plan_cb(self, msg: JointTrajectory):
        self._plan_count += 1
        wall = _time.time()
        if self._last_plan_wall is not None:
            dt = wall - self._last_plan_wall
            rate = 1.0 / dt if dt > 0 else float("inf")
        else:
            rate = float("nan")
        self._last_plan_wall = wall

        if not self.print_plans or not msg.points:
            return
        n = len(msg.points)
        first = msg.points[0].positions
        last = msg.points[-1].positions
        # Just print first 3 dims of first / last point (base commands) so
        # the line stays readable.
        first_str = " ".join(f"{first[i]:+.3f}" for i in range(min(3, len(first))))
        last_str = " ".join(f"{last[i]:+.3f}" for i in range(min(3, len(last))))
        self.get_logger().info(
            f"plan #{self._plan_count} @ {rate:5.1f}Hz n_pts={n} "
            f"first[base]=[{first_str}] last[base]=[{last_str}]"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MockBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
