"""Standalone VLA chunk publisher (no state, no clobbering).

Why this exists alongside `mock_bridge.py`:
  mock_bridge.py publishes BOTH fake state AND VLA chunks. When you chain
  the planner with `mppi_ros_sim_node.py` (which also publishes state on
  `/px4/bridge/*` in non-openloop mode), running mock_bridge alongside the
  sim would have two publishers fighting over the same topics. This script
  is the VLA-only subset: publishes /vla/chunk and listens to /mppi/plan,
  nothing else.

Topics published:
  - /vla/chunk   trajectory_msgs/JointTrajectory  (configurable via vla_topic)

Topics subscribed:
  - /mppi/plan   trajectory_msgs/JointTrajectory  (for one-line summary log)

Run alongside the sim recipe in README_VLA_SIM_TEST.md.
"""

import math
import time as _time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# 11-D control vector layout — must match mppi_vla_node.py.
NU = 11
VX_IDX, VY_IDX, WZ_IDX = 0, 1, 2


class VlaChunkPublisher(Node):
    def __init__(self):
        super().__init__("vla_chunk_publisher")

        self.declare_parameter("vla_topic", "/vla/chunk")
        self.declare_parameter("vla_pub_rate_hz", 1.5)
        self.declare_parameter("vla_chunk_horizon_s", 2.0)
        self.declare_parameter("vla_chunk_dt_s", 0.05)
        # vla_mode options (same set mock_bridge.py supports):
        #   "zero"             : all zeros.
        #   "constant_forward" : vx = vla_const_vx for the whole chunk.
        #   "step"             : vx = 0 for first half, vx = vla_step_vx after.
        #   "sinusoid"         : vx = vla_sin_amp * sin(2π t / chunk_horizon).
        #   "yaw_spin"         : wz = vla_yaw_rate constant.
        #   "circle_track"     : (vx, vy, wz) drives a body-frame circle of
        #                        radius vla_circle_r at omega vla_circle_w.
        #                        Useful for visually comparing against the
        #                        AtmosM3 default circle cost.
        self.declare_parameter("vla_mode", "constant_forward")
        self.declare_parameter("vla_const_vx", 0.1)
        self.declare_parameter("vla_step_vx", 0.3)
        self.declare_parameter("vla_sin_amp", 0.2)
        self.declare_parameter("vla_yaw_rate", 0.3)
        self.declare_parameter("vla_circle_r", 1.0)
        self.declare_parameter("vla_circle_w", 0.3)

        self.declare_parameter("plan_topic", "/mppi/plan")
        self.declare_parameter("print_plans", True)
        self.declare_parameter("print_every_n", 1)

        vla_topic = str(self.get_parameter("vla_topic").value)
        self.vla_rate = float(self.get_parameter("vla_pub_rate_hz").value)
        self.vla_horizon = float(self.get_parameter("vla_chunk_horizon_s").value)
        self.vla_dt = float(self.get_parameter("vla_chunk_dt_s").value)
        self.vla_mode = str(self.get_parameter("vla_mode").value)
        self.vla_const_vx = float(self.get_parameter("vla_const_vx").value)
        self.vla_step_vx = float(self.get_parameter("vla_step_vx").value)
        self.vla_sin_amp = float(self.get_parameter("vla_sin_amp").value)
        self.vla_yaw_rate = float(self.get_parameter("vla_yaw_rate").value)
        self.vla_circle_r = float(self.get_parameter("vla_circle_r").value)
        self.vla_circle_w = float(self.get_parameter("vla_circle_w").value)

        self.vla_n_pts = max(int(round(self.vla_horizon / self.vla_dt)), 1)

        plan_topic = str(self.get_parameter("plan_topic").value)
        self.print_plans = bool(self.get_parameter("print_plans").value)
        self.print_every_n = max(int(self.get_parameter("print_every_n").value), 1)

        self.vla_pub = self.create_publisher(JointTrajectory, vla_topic, 10)
        self.create_subscription(
            JointTrajectory, plan_topic, self._plan_cb, qos_profile_sensor_data
        )

        self._t0 = _time.time()
        self._chunk_count = 0
        self._plan_count = 0
        self._last_plan_wall = None
        self.create_timer(1.0 / self.vla_rate, self._tick_vla)

        self.get_logger().info(
            f"vla_chunk_publisher ready. mode={self.vla_mode} "
            f"chunk={self.vla_n_pts}pts @ {self.vla_dt}s "
            f"(horizon {self.vla_horizon:.2f}s), pub_rate={self.vla_rate:.2f}Hz"
        )

    # ── Chunk builders ────────────────────────────────────────────────────
    def _build_controls(self, t_chunk_start: float) -> np.ndarray:
        """Return (vla_n_pts, NU). t_chunk_start is the chunk's anchor time."""
        u = np.zeros((self.vla_n_pts, NU), dtype=np.float32)
        if self.vla_mode == "zero":
            return u
        if self.vla_mode == "constant_forward":
            u[:, VX_IDX] = self.vla_const_vx
            return u
        if self.vla_mode == "step":
            half = self.vla_n_pts // 2
            u[half:, VX_IDX] = self.vla_step_vx
            return u
        if self.vla_mode == "sinusoid":
            tau = np.arange(self.vla_n_pts) * self.vla_dt
            u[:, VX_IDX] = self.vla_sin_amp * np.sin(
                2.0 * math.pi * tau / max(self.vla_horizon, 1e-3)
            )
            return u
        if self.vla_mode == "yaw_spin":
            u[:, WZ_IDX] = self.vla_yaw_rate
            return u
        if self.vla_mode == "circle_track":
            # Body-frame command for a constant-radius constant-omega circle:
            # vx_body = r*omega tangent, wz = omega. Keeps vy = 0.
            u[:, VX_IDX] = self.vla_circle_r * self.vla_circle_w
            u[:, WZ_IDX] = self.vla_circle_w
            return u
        self.get_logger().warn(
            f"Unknown vla_mode={self.vla_mode!r}; falling back to zero."
        )
        return u

    def _tick_vla(self):
        now_msg = self.get_clock().now().to_msg()
        t_chunk = _time.time() - self._t0
        ctrls = self._build_controls(t_chunk)

        msg = JointTrajectory()
        msg.header.stamp = now_msg
        msg.header.frame_id = "vla_chunk_publisher"
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
        self._chunk_count += 1

    # ── Plan listener (eyeball only) ──────────────────────────────────────
    def _plan_cb(self, msg: JointTrajectory):
        self._plan_count += 1
        wall = _time.time()
        if self._last_plan_wall is not None:
            dt = wall - self._last_plan_wall
            rate = 1.0 / dt if dt > 0 else float("inf")
        else:
            rate = float("nan")
        self._last_plan_wall = wall

        if (not self.print_plans or not msg.points
                or self._plan_count % self.print_every_n):
            return
        n = len(msg.points)
        first = msg.points[0].positions
        last = msg.points[-1].positions
        first_str = " ".join(f"{first[i]:+.3f}" for i in range(min(3, len(first))))
        last_str = " ".join(f"{last[i]:+.3f}" for i in range(min(3, len(last))))
        self.get_logger().info(
            f"plan #{self._plan_count} @ {rate:5.1f}Hz n_pts={n} "
            f"first[base]=[{first_str}] last[base]=[{last_str}] "
            f"chunks_sent={self._chunk_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VlaChunkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
