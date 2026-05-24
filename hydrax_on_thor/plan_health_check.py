"""Subscribe to /mppi/plan and diagnose plan-content health.

Per-plan output (one line each, throttled by print_every_n):

  plan #N  base_dims_nan=[X X X] arm_dims_nan=[X X X X X X] grip_nan=X dead=X
           first_nan_knot=K (-1 if none)
           per-knot-nan-count first 5: [..]
           base finite range: vx[-a,+b] vy[-c,+d] wz[-e,+f]
           arm  finite range: ...
           gripper finite range: ...

Quick mental model:
  - first_nan_knot=0 means the very first knot of the plan is NaN -> MPPI's
    optimized mean is bad at the head; bug is in optimize/cost/state init.
  - first_nan_knot=K (mid-plan) means MPPI rolled forward and diverged at
    horizon step K -> dynamics divergence during rollouts.
  - per-dim NaN counts tell you which control channels go bad first.

Run alongside mppi_ros_node.py / mppi_vla_node.py + any state source.
"""

import math
import time as _time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from trajectory_msgs.msg import JointTrajectory


NU = 11
BASE_SLICE = slice(0, 3)
ARM_SLICE = slice(3, 9)
GRIP_IDX = 9
DEAD_IDX = 10


class PlanHealthCheck(Node):
    def __init__(self):
        super().__init__("plan_health_check")

        self.declare_parameter("plan_topic", "/mppi/plan")
        self.declare_parameter("print_every_n", 10)
        # If True, the first NaN plan (and the first GOOD plan after a NaN
        # streak) is always logged, in addition to the throttled cadence.
        self.declare_parameter("always_log_transitions", True)

        plan_topic = str(self.get_parameter("plan_topic").value)
        self.print_every_n = max(int(self.get_parameter("print_every_n").value), 1)
        self.always_log_transitions = bool(
            self.get_parameter("always_log_transitions").value
        )

        self.create_subscription(
            JointTrajectory, plan_topic, self._cb, qos_profile_sensor_data
        )

        self._count = 0
        self._last_was_nan = None        # None=no plans yet, True/False otherwise
        self._last_wall = None
        self.get_logger().info(
            f"plan_health_check ready. Listening on {plan_topic} "
            f"(printing every {self.print_every_n} plans)."
        )

    def _cb(self, msg: JointTrajectory):
        self._count += 1
        wall = _time.time()
        if self._last_wall is not None:
            dt = wall - self._last_wall
            rate = 1.0 / dt if dt > 0 else float("inf")
        else:
            rate = float("nan")
        self._last_wall = wall

        n_knots = len(msg.points)
        if n_knots == 0:
            self.get_logger().warn(f"plan #{self._count}: empty (no points)")
            return

        # Stack into (n_knots, nu) array. Pad/truncate to NU if mismatched.
        arr = np.full((n_knots, NU), np.nan, dtype=np.float64)
        for i, pt in enumerate(msg.points):
            vals = np.asarray(pt.positions, dtype=np.float64)
            k = min(vals.size, NU)
            arr[i, :k] = vals[:k]

        nan_mask = ~np.isfinite(arr)             # True where NaN/inf
        any_nan = bool(nan_mask.any())
        all_nan = bool(nan_mask.all())

        # Decide whether to log this one.
        is_transition = (
            self.always_log_transitions
            and self._last_was_nan is not None
            and any_nan != self._last_was_nan
        )
        is_first_nan = (
            self.always_log_transitions
            and self._last_was_nan is None
            and any_nan
        )
        throttled = (self._count % self.print_every_n) == 0
        do_log = throttled or is_transition or is_first_nan

        self._last_was_nan = any_nan

        if not do_log:
            return

        # Compose detailed report.
        per_dim_nan = nan_mask.sum(axis=0)                       # (NU,)
        per_knot_nan = nan_mask.sum(axis=1)                      # (n_knots,)
        first_nan_knot = int(np.argmax(per_knot_nan > 0)) if any_nan else -1
        if any_nan and per_knot_nan[first_nan_knot] == 0:
            first_nan_knot = -1                                  # argmax of all-zero

        def _rng(slice_) -> str:
            sub = arr[:, slice_] if isinstance(slice_, slice) else arr[:, slice_:slice_+1]
            finite = sub[np.isfinite(sub)]
            if finite.size == 0:
                return "ALL-NAN"
            lo, hi = float(finite.min()), float(finite.max())
            return f"[{lo:+.3f},{hi:+.3f}]"

        header = (
            f"plan #{self._count} @ {rate:5.1f}Hz "
            f"n_knots={n_knots} "
            + ("ALL-NAN " if all_nan else ("SOME-NAN " if any_nan else "OK "))
            + f"first_nan_knot={first_nan_knot}"
        )
        nan_counts = (
            f"  nan_per_dim base={list(per_dim_nan[BASE_SLICE])} "
            f"arm={list(per_dim_nan[ARM_SLICE])} "
            f"grip={int(per_dim_nan[GRIP_IDX])} dead={int(per_dim_nan[DEAD_IDX])}"
        )
        first_knot_nans = (
            f"  per_knot_nan first 6: {list(per_knot_nan[:6])}"
        )
        ranges = (
            f"  finite ranges: base_vx={_rng(0)} vy={_rng(1)} wz={_rng(2)}; "
            f"arm_j0={_rng(3)} j5={_rng(8)}; grip={_rng(GRIP_IDX)}"
        )
        self.get_logger().info("\n".join([header, nan_counts, first_knot_nans, ranges]))


def main(args=None):
    rclpy.init(args=args)
    node = PlanHealthCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
