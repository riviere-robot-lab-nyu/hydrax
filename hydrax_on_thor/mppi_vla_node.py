"""ROS 2 node that runs the AtmosM3 MPPI planner with a VLA warm-start.

Same wiring as `mppi_ros_node.py` for state inputs and plan output, plus:

  - Subscribes to a VLA action-chunk topic (trajectory_msgs/JointTrajectory).
    Each chunk is a sequence of nu-vectors with per-point `time_from_start`,
    anchored at `header.stamp`.

Warm-start behavior:

  1. On every plan tick: run `jit_optimize`, then overwrite the trailing
     `vla_tail_knots` entries of `policy_params.mean` with the VLA chunk
     resampled at the corresponding knot times. A scalar `vla_tail_alpha`
     blends between MPPI's optimized tail (0.0) and the VLA prior (1.0).

  2. When a new VLA chunk arrives: set a "full warmstart" flag. On the next
     tick, *before* `jit_optimize`, resample the chunk at all knot times and
     overwrite all of `policy_params.mean`.

  3. Cold start: wait (no publish) until base state AND a first VLA chunk
     have been received.

Net effect: MPPI is hard-snapped to the VLA plan whenever a fresh chunk
arrives (~1-2 Hz), is free to wander between chunks while still being pulled
toward the VLA action at the end of its horizon, and the publish format /
state pipeline are identical to `mppi_ros_node.py`.
"""

import time as _time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

jax.config.update("jax_compilation_cache_dir", "/workspace/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

from hydrax.algs import MPPI, MPPI_WithCtx
from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.tasks.atmos_m3_vla_track import AtmosM3VlaTrack
from hydrax.tasks.atmos_m3_value import AtmosM3ValueShaped
from hydrax.tasks.atmos_m3_vla_track_value import AtmosM3VlaTrackValue


CONTROL_NAMES = [
    "vx_body_cmd", "vy_body_cmd", "wz_body_cmd",
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper_cmd", "_dead",
]

PLAN_TOPIC = "/mppi/plan"
VLA_CHUNK_TOPIC = "/vla/chunk"
ARM_STATE_TOPIC = "/TODO/arm/joint_states"
BASE_STATE_TOPIC = "/TODO/base/odom"
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

LOCAL_POS_HEADING_INDEX = 14
BASE_VEL_LIN_X_INDEX = 3
BASE_VEL_LIN_Y_INDEX = 4
BASE_VEL_YAW_RATE_INDEX = 15


class MppiVlaPlannerNode(Node):
    def __init__(self):
        super().__init__("mppi_vla_planner")

        self.declare_parameter("plan_rate_hz", 20.0)
        self.declare_parameter("num_samples", 256)
        self.declare_parameter("plan_horizon", 0.25)
        self.declare_parameter("num_knots", 8)
        self.declare_parameter("temperature", 0.2)

        # VLA warm-start tuning.
        self.declare_parameter("vla_chunk_topic", VLA_CHUNK_TOPIC)
        self.declare_parameter("vla_tail_knots", 1)
        self.declare_parameter("vla_tail_alpha", 1.0)
        self.declare_parameter("vla_full_warmstart_on_new_chunk", True)
        self.declare_parameter("vla_max_chunk_age_s", 2.0)

        # Cost-mode selection.
        #   "default"         : stock AtmosM3 cost. Same as mppi_ros_node.py.
        #   "vla_track"       : track a state trajectory rolled out from the VLA
        #                       chunk once per chunk arrival.
        #   "value_shaped"    : -V(s) (or -Q(s,u)) per-step from a flax MLP.
        #   "vla_track_value" : blend of vla_track and value_shaped, with
        #                       independent scalar weights (w_track, w_value).
        #                       Requires a value_ckpt_path.
        self.declare_parameter("cost_mode", "default")

        # vla_track-specific.
        # Reference only needs to cover one chunk period + the plan horizon
        # (~0.67 s + 0.25 s at the default 1.5 Hz chunk rate). 1.0 s is plenty;
        # longer just makes the per-chunk mjx rollout slower without buying
        # MPPI any new lookups (the cost clamps past the end of the ref).
        self.declare_parameter("vla_track_ref_horizon_s", 1.0)
        self.declare_parameter("vla_track_base_weights",
                               [10.0, 10.0, 5.0, 1.0, 1.0, 0.1])
        self.declare_parameter("vla_track_arm_weights",
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("vla_track_ctrl_reg", 0.1)

        # value_shaped-specific.
        self.declare_parameter("value_kind", "v")          # "v" or "q"
        self.declare_parameter("value_hidden", 256)
        self.declare_parameter("value_out_scale", 1.0)
        self.declare_parameter("value_ckpt_path", "")
        self.declare_parameter("value_ctrl_reg", 0.1)

        # vla_track_value-specific (the other two-mode params are reused).
        self.declare_parameter("vla_track_value_w_track", 1.0)
        self.declare_parameter("vla_track_value_w_value", 1.0)

        plan_rate = float(self.get_parameter("plan_rate_hz").value)
        num_samples = int(self.get_parameter("num_samples").value)
        plan_horizon = float(self.get_parameter("plan_horizon").value)
        num_knots = int(self.get_parameter("num_knots").value)
        temperature = float(self.get_parameter("temperature").value)

        vla_chunk_topic = str(self.get_parameter("vla_chunk_topic").value)
        self.vla_tail_knots = max(int(self.get_parameter("vla_tail_knots").value), 0)
        self.vla_tail_alpha = float(self.get_parameter("vla_tail_alpha").value)
        self.vla_full_warmstart_on_new_chunk = bool(
            self.get_parameter("vla_full_warmstart_on_new_chunk").value
        )
        self.vla_max_chunk_age_s = float(
            self.get_parameter("vla_max_chunk_age_s").value
        )

        self.cost_mode = str(self.get_parameter("cost_mode").value)
        assert self.cost_mode in (
            "default", "vla_track", "value_shaped", "vla_track_value",
        ), f"Unknown cost_mode={self.cost_mode!r}"
        # Modes that need the chunk → reference-state mjx rollout.
        self._uses_vla_track = self.cost_mode in ("vla_track", "vla_track_value")

        self.arm_joint_names = list(ARM_JOINT_NAMES)

        noise_level = jnp.array([0.1, 0.1, 0.1] + [0.01] * 8)

        if self.cost_mode == "default":
            self.task = AtmosM3()
            ctrl_cls = MPPI
        elif self.cost_mode == "vla_track":
            self.task = AtmosM3VlaTrack(
                base_track_weights=jnp.asarray(
                    list(self.get_parameter("vla_track_base_weights").value),
                    dtype=jnp.float32,
                ),
                arm_track_weights=jnp.asarray(
                    list(self.get_parameter("vla_track_arm_weights").value),
                    dtype=jnp.float32,
                ),
                ctrl_reg=float(
                    self.get_parameter("vla_track_ctrl_reg").value
                ),
            )
            ctrl_cls = MPPI_WithCtx
        elif self.cost_mode == "value_shaped":
            self.task = AtmosM3ValueShaped(
                value_kind=str(self.get_parameter("value_kind").value),
                hidden=int(self.get_parameter("value_hidden").value),
                out_scale=float(self.get_parameter("value_out_scale").value),
                ctrl_reg=float(self.get_parameter("value_ctrl_reg").value),
            )
            ctrl_cls = MPPI_WithCtx
        else:  # vla_track_value
            self.task = AtmosM3VlaTrackValue(
                w_track=float(
                    self.get_parameter("vla_track_value_w_track").value
                ),
                w_value=float(
                    self.get_parameter("vla_track_value_w_value").value
                ),
                value_kind=str(self.get_parameter("value_kind").value),
                hidden=int(self.get_parameter("value_hidden").value),
                out_scale=float(self.get_parameter("value_out_scale").value),
                base_track_weights=jnp.asarray(
                    list(self.get_parameter("vla_track_base_weights").value),
                    dtype=jnp.float32,
                ),
                arm_track_weights=jnp.asarray(
                    list(self.get_parameter("vla_track_arm_weights").value),
                    dtype=jnp.float32,
                ),
                ctrl_reg=float(
                    self.get_parameter("vla_track_ctrl_reg").value
                ),
            )
            ctrl_cls = MPPI_WithCtx

        self.ctrl = ctrl_cls(
            self.task,
            num_samples=num_samples,
            noise_level=noise_level,
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
        self.num_knots = num_knots
        self.horizon_steps = max(int(round(plan_horizon / self.sim_dt)), 1)

        # Model-default qpos. See mppi_ros_node.py for rationale: arm /
        # gripper slots that aren't fed by sensors stay at the model's
        # valid keyframe instead of being zeroed (which can diverge under
        # mjx.step and produce NaN plans).
        self.qpos0 = np.asarray(self.mj_model.qpos0, dtype=np.float32)

        # Clamp tail size to valid range.
        self.vla_tail_knots = min(self.vla_tail_knots, self.num_knots)

        # Control bounds for clipping VLA values into the task's admissible set.
        self.u_min = np.asarray(self.task.u_min, dtype=np.float32)
        self.u_max = np.asarray(self.task.u_max, dtype=np.float32)

        self.mjx_data = mjx.make_data(self.task.model)
        self.policy_params = self.ctrl.init_params()
        self.integral = jnp.zeros(1)

        # ── ctx setup (cost-mode dependent) ───────────────────────────────
        # ctx is the pytree threaded into MPPI_WithCtx.optimize. Its shape
        # and dtype must be constant for JIT; only its values change.
        if self._uses_vla_track:
            self.vla_track_ref_horizon_s = float(
                self.get_parameter("vla_track_ref_horizon_s").value
            )
            self.H_ref = max(
                int(round(self.vla_track_ref_horizon_s / self.sim_dt)), 1
            )
            n_track = 12  # (x, y, yaw, vx, vy, wz, arm0..arm5_pos)
            # Initial vla half of ctx: zeros + ref_t0=0, ref_dt=sim_dt.
            init_vla_ctx = (
                jnp.zeros((self.H_ref, n_track), dtype=jnp.float32),
                jnp.float32(0.0),
                jnp.float32(self.sim_dt),
            )

        if self.cost_mode == "vla_track":
            self.ctx = init_vla_ctx
        elif self.cost_mode == "value_shaped":
            self.value_ctrl_reg = float(self.get_parameter("value_ctrl_reg").value)
            self.value_kind = str(self.get_parameter("value_kind").value)
            ckpt = str(self.get_parameter("value_ckpt_path").value)
            if ckpt:
                self.ctx = self._load_value_params(ckpt)
                self.get_logger().info(f"Loaded value-net params from {ckpt}")
            else:
                self.ctx = self.task.init_value_params(seed=0)
                self.get_logger().warn(
                    "value_shaped mode with no value_ckpt_path: using random "
                    "MLP init. Cost signal will be ~zero (tanh of tiny inputs)."
                )
        elif self.cost_mode == "vla_track_value":
            # Per user choice: refuse to start with no checkpoint, so we never
            # fly a value term initialized to random tanh-of-noise.
            ckpt = str(self.get_parameter("value_ckpt_path").value)
            if not ckpt:
                raise RuntimeError(
                    "cost_mode=vla_track_value requires value_ckpt_path. "
                    "If you want pure tracking while V is still training, "
                    "use cost_mode=vla_track instead."
                )
            value_params = self._load_value_params(ckpt)
            self.get_logger().info(f"Loaded value-net params from {ckpt}")
            self.ctx = (init_vla_ctx, value_params)
        else:
            self.ctx = None

        self.jit_interp = jax.jit(self.ctrl.interp_func)

        if self.cost_mode == "default":
            self.jit_optimize = jax.jit(self.ctrl.optimize)
        else:
            self.jit_optimize = jax.jit(self.ctrl.optimize)
            # Build the VLA chunk → reference-state rollout (vla_track and
            # vla_track_value both need it).
            if self._uses_vla_track:
                self.jit_vla_rollout = jax.jit(self._rollout_vla_chunk)
                # The H_ref-step single-trajectory mjx rollout is the slow path
                # on chunk arrival (~100 ms at H_ref=100). Run it on a worker
                # thread so the 20 Hz planner doesn't block; swap ctx in on the
                # first plan tick after the future resolves. One-tick staleness
                # (~50 ms) is negligible vs the chunk period (~667 ms).
                self._ref_pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="vla_ref"
                )
                self._pending_ref: Future | None = None
                self._first_ref_done = False

        # ── JIT warm-up ───────────────────────────────────────────────────
        self.get_logger().info(f"Jitting MPPI controller (cost_mode={self.cost_mode})...")
        t0 = _time.time()
        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.asarray(self.qpos0, dtype=jnp.float32),
            qvel=jnp.zeros(self.nv, dtype=jnp.float32),
            time=jnp.array(0.0, dtype=jnp.float32),
        )
        if self.cost_mode == "default":
            self.policy_params, _ = self.jit_optimize(
                self.mjx_data, self.policy_params, self.integral
            )
        else:
            self.policy_params, _ = self.jit_optimize(
                self.mjx_data, self.policy_params, self.integral, self.ctx
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
        if self._uses_vla_track:
            _dummy_ctrls = jnp.zeros((self.H_ref, self.nu), dtype=jnp.float32)
            _warm_ref = self.jit_vla_rollout(self.mjx_data, _dummy_ctrls)
            jax.block_until_ready(_warm_ref)
        self.get_logger().info(f"JIT done in {_time.time() - t0:.2f}s")

        # ── State caches ──────────────────────────────────────────────────
        self.base_qpos = np.zeros(3, dtype=np.float32)
        self.base_qvel = np.zeros(3, dtype=np.float32)
        self.arm_qpos = np.zeros(6, dtype=np.float32)
        self.arm_qvel = np.zeros(6, dtype=np.float32)
        self.gripper_qpos = np.zeros(2, dtype=np.float32)
        self.gripper_qvel = np.zeros(2, dtype=np.float32)

        self.have_odom = False
        self.have_pos = False
        self.last_base_stamp = None
        self.last_odom = None
        self.last_local_pos = None

        # ── VLA chunk cache ───────────────────────────────────────────────
        # Times are stored in the node's "seconds since _t0_wall" clock so
        # they're directly comparable to params.tk (which uses state.time).
        self.vla_times = None       # (chunk_len,) np.float32, ascending
        self.vla_controls = None    # (chunk_len, nu) np.float32
        self.vla_recv_node_t = None # node-clock seconds when the chunk arrived
        self.have_vla = False
        self.needs_full_warmstart = False

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Float64MultiArray, '/px4/bridge/vehicle_odometry',
            self.odom_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Float64MultiArray, '/px4/bridge/vehicle_local_position_v1',
            self.local_pos_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            JointTrajectory, vla_chunk_topic, self._vla_cb, 10,
        )

        self.plan_pub = self.create_publisher(JointTrajectory, PLAN_TOPIC, 10)

        self.timer = self.create_timer(1.0 / plan_rate, self._plan_and_publish)

        self.get_logger().info(
            f"MPPI+VLA planner ready (cost_mode={self.cost_mode}). "
            f"Replanning at {plan_rate:.1f} Hz, "
            f"publishing on {PLAN_TOPIC}, horizon "
            f"{self.horizon_steps} steps ({plan_horizon:.2f}s). "
            f"VLA chunk topic: {vla_chunk_topic}, "
            f"tail_knots={self.vla_tail_knots}, tail_alpha={self.vla_tail_alpha}, "
            f"full_warmstart_on_new_chunk={self.vla_full_warmstart_on_new_chunk}."
        )
        self.is_first_tick = True

    # ── Callbacks ─────────────────────────────────────────────────────────
    def odom_cb(self, msg: Float64MultiArray):
        self.last_odom = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_odom = True

    def local_pos_cb(self, msg: Float64MultiArray):
        self.last_local_pos = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_pos = True

    def _vla_cb(self, msg: JointTrajectory):
        """Cache a VLA action chunk in the node's wall-clock frame.

        We convert the chunk's (header.stamp + time_from_start) into the same
        "seconds since _t0_wall" timeline that params.tk lives in, so later
        resampling is a plain 1-D interp per control dim.
        """
        if not msg.points:
            self.get_logger().warn("VLA chunk had no points; ignoring.")
            return
        # _t0_wall is established on the first plan tick. If a chunk shows up
        # before that, anchor it now so the math still works.
        if not hasattr(self, "_t0_wall"):
            self._t0_wall = self.get_clock().now().nanoseconds * 1e-9

        stamp = msg.header.stamp
        chunk_t0_node = (
            stamp.sec + stamp.nanosec * 1e-9
        ) - self._t0_wall

        n_pts = len(msg.points)
        times = np.empty(n_pts, dtype=np.float32)
        ctrls = np.zeros((n_pts, self.nu), dtype=np.float32)
        for i, pt in enumerate(msg.points):
            tfs = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            times[i] = chunk_t0_node + tfs
            vals = pt.positions
            k = min(len(vals), self.nu)
            ctrls[i, :k] = np.asarray(vals[:k], dtype=np.float32)

        # Ensure strictly ascending time axis for np.interp.
        order = np.argsort(times, kind="stable")
        times = times[order]
        ctrls = ctrls[order]

        # Clip into task control bounds so the seeded mean is always feasible.
        np.clip(ctrls, self.u_min, self.u_max, out=ctrls)

        self.vla_times = times
        self.vla_controls = ctrls
        self.vla_recv_node_t = self.get_clock().now().nanoseconds * 1e-9 - self._t0_wall
        self.have_vla = True
        if self.vla_full_warmstart_on_new_chunk:
            self.needs_full_warmstart = True

    # ── Helpers ───────────────────────────────────────────────────────────
    def _vla_resample(self, query_times: np.ndarray) -> np.ndarray:
        """Resample the cached VLA chunk at query times (node clock).

        Returns an array of shape (len(query_times), nu). Out-of-range queries
        are clamped to the chunk endpoints (np.interp's default).
        """
        out = np.empty((len(query_times), self.nu), dtype=np.float32)
        for d in range(self.nu):
            out[:, d] = np.interp(
                query_times, self.vla_times, self.vla_controls[:, d]
            )
        return out

    def _chunk_age_s(self, now_node_t: float) -> float:
        if self.vla_recv_node_t is None:
            return float("inf")
        return now_node_t - self.vla_recv_node_t

    def _rollout_vla_chunk(self, state: mjx.Data, controls: jax.Array) -> jax.Array:
        """Roll a fixed-shape control sequence through the mjx model.

        Returns the tracked state slice at each step: (H_ref, n_track).
        Uses the same ctrl_transform / integral pipeline that MPPI uses, so
        the resulting reference is what *would* happen if the robot executed
        these controls under the same low-level controller.
        """
        def _step(carry, u):
            x, integral = carry
            actual = self.task.ctrl_transform_with_integral(x, u, integral)
            integral = self.task.update_integral(x, u, integral, actual)
            x = x.replace(ctrl=actual)
            x = mjx.step(self.task.model, x)
            return (x, integral), self.task.slice_state(x)

        integral0 = jnp.zeros(1)
        _, ref_states = jax.lax.scan(_step, (state, integral0), controls)
        return ref_states

    def _load_value_params(self, path: str):
        """Load value-net params from a pickled flax pytree."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def _install_vla_ctx(self, new_vla_ctx) -> None:
        """Insert a fresh vla_ctx (ref_states, ref_t0, ref_dt) into self.ctx.

        In `vla_track` mode the ctx *is* the vla tuple; in `vla_track_value`
        mode it's a (vla_ctx, value_params) pair and we only swap the first
        slot. Only called from the main thread.
        """
        if self.cost_mode == "vla_track":
            self.ctx = new_vla_ctx
        else:  # vla_track_value
            self.ctx = (new_vla_ctx, self.ctx[1])

    def _maybe_rebuild_vla_ref(self, now_t: float) -> None:
        """Kick off (or block on, the first time) a VLA-chunk → reference-state
        rollout. The rollout runs on a worker thread; the result is adopted on
        a later plan tick via _adopt_ref_if_ready, so the planner never blocks
        on the H_ref-step mjx scan.

        On the very first chunk we DO block, because the vla half of self.ctx
        is still zeros and running optimize against it would produce one
        garbage plan.
        """
        if not self._uses_vla_track:
            return
        # Drop new requests while a prior rebuild is still in flight. At 1.5 Hz
        # chunk rate and ~100 ms rebuilds this is essentially never hit; if it
        # ever does, the next chunk arrival will trigger another rebuild.
        if self._pending_ref is not None:
            return

        grid_times = (
            np.float32(now_t) + np.arange(self.H_ref, dtype=np.float32) * self.sim_dt
        )
        ctrls_grid = self._vla_resample(grid_times)            # (H_ref, nu)
        ctrls_grid = np.clip(ctrls_grid, self.u_min, self.u_max)
        ctrls_jax = jnp.asarray(ctrls_grid, dtype=jnp.float32)
        # Capture the current mjx.Data so the worker's view of state can't
        # race with the main thread's next-tick replace().
        state_snapshot = self.mjx_data

        def _run():
            ref_states = self.jit_vla_rollout(state_snapshot, ctrls_jax)
            jax.block_until_ready(ref_states)
            return (
                ref_states,
                jnp.float32(now_t),
                jnp.float32(self.sim_dt),
            )

        fut = self._ref_pool.submit(_run)
        if not self._first_ref_done:
            self._install_vla_ctx(fut.result())
            self._first_ref_done = True
        else:
            self._pending_ref = fut

    def _adopt_ref_if_ready(self) -> None:
        """If a backgrounded ref rollout has finished, swap it into self.ctx.

        Called once per plan tick before optimize, so the freshest available
        reference is used. Only mutates ctx from the main (planning) thread.
        """
        if not self._uses_vla_track:
            return
        if self._pending_ref is not None and self._pending_ref.done():
            self._install_vla_ctx(self._pending_ref.result())
            self._pending_ref = None

    # ── State assembly ────────────────────────────────────────────────────
    def _assemble_state(self):
        qpos = self.qpos0.copy()
        qvel = np.zeros(self.nv, dtype=np.float32)
        odom_msg, odom_t = self.last_odom
        local_pos_msg, _ = self.last_local_pos
        qpos[2] = float(local_pos_msg.data[LOCAL_POS_HEADING_INDEX] - np.pi / 2)
        qpos[0] = float(odom_msg.data[4])
        qpos[1] = float(odom_msg.data[3])
        qvel[2] = float(odom_msg.data[BASE_VEL_YAW_RATE_INDEX])
        qvel[0] = float(odom_msg.data[ODOM_LIN_VEL_START + 1])
        qvel[1] = float(odom_msg.data[ODOM_LIN_VEL_START])
        self.last_base_stamp = RclTime(nanoseconds=int(odom_t * 1e9)).to_msg()
        return qpos, qvel

    # ── Plan + publish ────────────────────────────────────────────────────
    def _plan_and_publish(self):
        if not (self.have_odom and self.have_pos and self.have_vla):
            self.get_logger().warn(
                f"Waiting for state... odom={self.have_odom} "
                f"pos={self.have_pos} vla={self.have_vla}",
                throttle_duration_sec=2.0,
            )
            return

        qpos, qvel = self._assemble_state()
        if not hasattr(self, "_t0_wall"):
            self._t0_wall = self.get_clock().now().nanoseconds * 1e-9
        now_t = self.get_clock().now().nanoseconds * 1e-9 - self._t0_wall

        if self.is_first_tick:
            self.vla_recv_node_t = now_t
            self.is_first_tick = False

        chunk_age = self._chunk_age_s(now_t)
        chunk_stale = chunk_age > self.vla_max_chunk_age_s
        if chunk_stale:
            self.get_logger().warn(
                f"VLA chunk is stale ({chunk_age:.2f}s > "
                f"{self.vla_max_chunk_age_s:.2f}s); skipping tail pinning.",
                throttle_duration_sec=2.0,
            )

        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(qpos),
            qvel=jnp.array(qvel),
            time=jnp.array(now_t, dtype=jnp.float32),
        )

        # Pre-optimize full warmstart: only on the tick after a new chunk.
        # We resample the chunk at the knot times the *next* optimize call
        # will use after its internal shift: tk_new = linspace(0, H, K) + now_t.
        if self.needs_full_warmstart and not chunk_stale:
            tk_pre = np.linspace(
                0.0, float(self.ctrl.plan_horizon), self.num_knots
            ).astype(np.float32) + np.float32(now_t)
            seeded = self._vla_resample(tk_pre)               # (K, nu)
            seeded = np.clip(seeded, self.u_min, self.u_max)
            new_mean = jnp.asarray(seeded, dtype=self.policy_params.mean.dtype)
            self.policy_params = self.policy_params.replace(mean=new_mean)
            # vla_track mode: kick off a fresh ref rollout anchored at the
            # current state. After the first chunk this runs on a worker
            # thread and the result is swapped into self.ctx on a later tick.
            self._maybe_rebuild_vla_ref(now_t)
            self.needs_full_warmstart = False

        # Adopt a finished background rollout, if one is ready, before optimize
        # — so the freshest ref is used. No-op outside vla_track.
        self._adopt_ref_if_ready()

        if self.cost_mode == "default":
            self.policy_params, _ = self.jit_optimize(
                self.mjx_data, self.policy_params, self.integral
            )
        else:
            self.policy_params, _ = self.jit_optimize(
                self.mjx_data, self.policy_params, self.integral, self.ctx
            )

        # Post-optimize tail pinning. params.tk is already in node-clock
        # absolute time (alg_base.optimize shifts it by state.time), so we
        # can use it directly to look up the VLA chunk.
        if (
            self.vla_tail_knots > 0
            and self.vla_tail_alpha > 0.0
            and not chunk_stale
        ):
            tk_abs = np.asarray(self.policy_params.tk, dtype=np.float32)
            tail_times = tk_abs[-self.vla_tail_knots:]
            vla_tail = self._vla_resample(tail_times)         # (vla_tail_knots, nu)
            mean_np = np.array(self.policy_params.mean, dtype=np.float32)
            alpha = float(np.clip(self.vla_tail_alpha, 0.0, 1.0))
            mean_np[-self.vla_tail_knots:] = (
                alpha * vla_tail
                + (1.0 - alpha) * mean_np[-self.vla_tail_knots:]
            )
            np.clip(mean_np, self.u_min, self.u_max, out=mean_np)
            self.policy_params = self.policy_params.replace(
                mean=jnp.asarray(mean_np, dtype=self.policy_params.mean.dtype)
            )

        tq = jnp.arange(0, self.horizon_steps) * self.sim_dt + self.mjx_data.time
        us = np.asarray(
            self.jit_interp(
                tq, self.policy_params.tk,
                self.policy_params.mean[None, ...],
            )
        )[0]

        msg = JointTrajectory()
        msg.header.stamp = (
            self.last_base_stamp
            if self.last_base_stamp is not None
            else self.get_clock().now().to_msg()
        )
        msg.header.frame_id = "atmos_m3_mppi_vla"
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
    node = MppiVlaPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
