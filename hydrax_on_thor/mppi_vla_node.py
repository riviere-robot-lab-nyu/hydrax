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
from geometry_msgs.msg import PoseStamped

from hydrax.algs import MPPI, MPPI_WithCtx
from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.tasks.atmos_m3_vla_track import AtmosM3VlaTrack
from hydrax.tasks.atmos_m3_value import AtmosM3ValueShaped
from hydrax.tasks.atmos_m3_vla_track_value import AtmosM3VlaTrackValue
from hydrax.tasks.iql_nets import load_dataset_norm, load_iql_bundle


CONTROL_NAMES = [
    "vx_body_cmd", "vy_body_cmd", "wz_body_cmd",
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper_cmd", "_dead",
]

PLAN_TOPIC = "/mppi/plan"
# Published by trossen_client_ros_chunk_publisher.py as a Float64MultiArray
# of shape (N, VLA_ACTION_DIM), row-major flattened. No header timestamp and
# no per-point time_from_start — we anchor at receive time and space samples
# by vla_chunk_dt_s.
VLA_CHUNK_TOPIC = "/vla/action_chunk"
# VLA action layout: [arm_0..arm_5, gripper, vx, vy, yaw_rate].
# MPPI ctrl layout:  [vx, vy, wz, arm_0..arm_5, gripper, dead] (see CONTROL_NAMES).
# We re-order in _vla_cb so the seeded mean lives in MPPI/task convention.
VLA_ACTION_DIM = 10
# /follower/joint_state: Float64MultiArray of positions(7) + velocities(7),
# published by trossen_client_ros_async.py from the Trossen driver.
# Layout: [arm_0..arm_5, gripper, qd_arm_0..qd_arm_5, qd_gripper].
ARM_STATE_TOPIC = "/follower/joint_state"
ARM_STATE_LEN = 7  # 6 arm joints + 1 gripper

ODOM_LIN_VEL_START = 10
ODOM_ANG_VEL_START = 13

BASE_VEL_LIN_X_INDEX = 3
BASE_VEL_LIN_Y_INDEX = 4
BASE_VEL_YAW_RATE_INDEX = 15

GLOBAL_POSE_TOPIC = "/global_pose"


# Per-iql_mode w_value defaults. Calibrated from iql_smoke_test.py cost
# spreads so that the IQL term's contribution lands in a workable band for
# MPPI's fixed temperature (target spread/T ≈ 3–5).
#
# Reference spreads (smoke test, default temperature=0.2):
#   track ≈ 0.109, v/q/adv ≈ 0.004, telescoping ≈ 0.003, logprob ≈ 244.7.
#
# TRACK_VALUE defaults give ~75% tracking / 25% IQL influence by setting
# w_value * spread(iql) ≈ (1/3) * w_track * spread(track).
W_VALUE_DEFAULTS_TRACK_VALUE = {
    "v":           9.0,
    "q":           9.0,
    "advantage":   9.0,
    "telescoping": 12.0,
    "logprob":     0.00015,
}

# VALUE_SHAPED defaults scale the IQL term to spread ≈ 0.6 (so spread/T ≈ 3
# at the default temperature=0.2). The pure-value mode has no tracking term,
# so the IQL signal is the entire cost and needs a bigger scale than in the
# blended case.
W_VALUE_DEFAULTS_VALUE_SHAPED = {
    "v":           150.0,
    "q":           150.0,
    "advantage":   150.0,
    "telescoping": 200.0,
    "logprob":     0.0025,
}

# L=0.16665
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
        # Blend factor for full warmstart: 1.0 = hard snap mean to VLA (legacy
        # behavior), 0.0 = ignore VLA, in-between = convex combo with the
        # previously optimized mean. Only applied on chunk arrival, not per-tick.
        self.declare_parameter("vla_full_warmstart_alpha", 1.0)
        self.declare_parameter("vla_max_chunk_age_s", 2.0)
        # Time between consecutive actions inside a single chunk. The publisher
        # is the OpenPI policy server output (~30 Hz control by default).
        self.declare_parameter("vla_chunk_dt_s", 1.0 / 30.0)
        # Passthrough: skip MPPI entirely; publish the VLA chunk resampled at
        # the plan's normal query times as /mppi/plan. Used for end-to-end
        # verification of the chunk -> driver chain without the optimizer.
        self.declare_parameter("passthrough", False)

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
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("vla_track_ctrl_reg", 0.1)

        # value_shaped / vla_track_value: shared IQL params. The IQL nets,
        # obs-feature builder, and action minmax all come from a trained
        # ikostrikov/implicit_q_learning bundle. See hydrax.tasks.iql_nets
        # for the exact net defs / obs layout.
        #   iql_mode = "v" | "q" | "telescoping" | "advantage" | "logprob"
        #     v           : -V(s)      per step (dense)
        #     q           : -min(Q1,Q2)(s,u) per step
        #     telescoping : 0 per step; terminal = -γ^H V(s_T) (avoids
        #                   double-counting that the dense V form bakes in)
        #     advantage   : V(s)-min(Q1,Q2)(s,u) per step (=  -A)
        #     logprob     : -log π(u|s) per step (proxy for advantage when
        #                   V/Q are noisy)
        self.declare_parameter("iql_mode", "v")
        # Folder containing value.ckpt / critic.ckpt / actor.ckpt
        # (e.g. .../checkpoints/step_1000000). Empty string = random init
        # (intended only for plumbing smoke tests; cost signal will be junk).
        self.declare_parameter("iql_ckpt_dir", "")
        # Full path to the dataset config JSON (carries obs_mean/std and
        # action_min/max — required for normalization to match training).
        self.declare_parameter("iql_dataset_config_path", "")
        # MLP hidden dims; must match the trained checkpoint.
        # IQL mujoco_config: (256, 256).
        self.declare_parameter("iql_hidden_dims", [256, 256])
        # Training-time discount γ. Used by "telescoping" mode to weight the
        # terminal value (γ^H V(s_T)).
        self.declare_parameter("iql_gamma", 0.99)
        # Mild ctrl regularizer on the base velocity dims, applied on top of
        # the IQL term so the optimizer doesn't run free when V/Q is flat.
        self.declare_parameter("value_ctrl_reg", 0.1)
        # Multiplier on the IQL term for the pure value_shaped mode. -1.0 =
        # use the per-iql_mode default from W_VALUE_DEFAULTS_VALUE_SHAPED.
        self.declare_parameter("value_shaped_w_value", -1.0)

        # vla_track_value-specific (the other two-mode params are reused).
        # w_value=-1.0 means "use the per-iql_mode default from
        # W_VALUE_DEFAULTS_TRACK_VALUE" (calibrated for 75% track / 25% value).
        # Set to any non-negative value to override.
        self.declare_parameter("vla_track_value_w_track", 1.0)
        self.declare_parameter("vla_track_value_w_value", -1.0)

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
        self.vla_full_warmstart_alpha = float(
            np.clip(self.get_parameter("vla_full_warmstart_alpha").value, 0.0, 1.0)
        )
        self.vla_max_chunk_age_s = float(
            self.get_parameter("vla_max_chunk_age_s").value
        )
        self.vla_chunk_dt = float(self.get_parameter("vla_chunk_dt_s").value)
        self.passthrough = bool(self.get_parameter("passthrough").value)

        self.cost_mode = str(self.get_parameter("cost_mode").value)
        assert self.cost_mode in (
            "default", "vla_track", "value_shaped", "vla_track_value",
        ), f"Unknown cost_mode={self.cost_mode!r}"
        # Modes that need the chunk → reference-state mjx rollout.
        self._uses_vla_track = self.cost_mode in ("vla_track", "vla_track_value")

        noise_level = jnp.array([0.05, 0.05, 0.05] + [0.005] * 8)

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
            self._iql_norm = self._load_iql_norm_or_fail()
            iql_mode_str = str(self.get_parameter("iql_mode").value)
            w_value_param = float(
                self.get_parameter("value_shaped_w_value").value
            )
            if w_value_param < 0.0:
                w_value_resolved = W_VALUE_DEFAULTS_VALUE_SHAPED.get(
                    iql_mode_str, 1.0
                )
                self.get_logger().info(
                    f"value_shaped_w_value=auto → "
                    f"using per-mode default {w_value_resolved:g} for "
                    f"iql_mode={iql_mode_str!r} (target spread/T ≈ 3)"
                )
            else:
                w_value_resolved = w_value_param
            self.w_value_resolved = w_value_resolved
            self.task = AtmosM3ValueShaped(
                iql_mode=iql_mode_str,
                norm=self._iql_norm,
                hidden_dims=tuple(int(x) for x in
                                  self.get_parameter("iql_hidden_dims").value),
                gamma=float(self.get_parameter("iql_gamma").value),
                ctrl_reg=float(self.get_parameter("value_ctrl_reg").value),
                w_value=w_value_resolved,
            )
            ctrl_cls = MPPI_WithCtx
        else:  # vla_track_value
            self._iql_norm = self._load_iql_norm_or_fail()
            iql_mode_str = str(self.get_parameter("iql_mode").value)
            w_track_resolved = float(
                self.get_parameter("vla_track_value_w_track").value
            )
            w_value_param = float(
                self.get_parameter("vla_track_value_w_value").value
            )
            if w_value_param < 0.0:
                w_value_resolved = W_VALUE_DEFAULTS_TRACK_VALUE.get(
                    iql_mode_str, 1.0
                )
                self.get_logger().info(
                    f"vla_track_value_w_value=auto → "
                    f"using per-mode default {w_value_resolved:g} for "
                    f"iql_mode={iql_mode_str!r} (target: ~75% track / 25% value)"
                )
            else:
                w_value_resolved = w_value_param
            self.w_track_resolved = w_track_resolved
            self.w_value_resolved = w_value_resolved
            self.task = AtmosM3VlaTrackValue(
                w_track=w_track_resolved,
                w_value=w_value_resolved,
                iql_mode=iql_mode_str,
                norm=self._iql_norm,
                hidden_dims=tuple(int(x) for x in
                                  self.get_parameter("iql_hidden_dims").value),
                gamma=float(self.get_parameter("iql_gamma").value),
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
            # Always pass a fully-shaped IQL bundle: jit bakes the shape.
            self.ctx = self._build_iql_ctx(strict=False)
        elif self.cost_mode == "vla_track_value":
            # Blend mode: refuse to start with no checkpoint, so we never
            # fly a value term initialized to random noise. Bundle is a dict
            # with "vla" + "value" + "critic" + "actor" keys.
            iql_dict = self._build_iql_ctx(strict=True)
            self.ctx = {"vla": init_vla_ctx, **iql_dict}
        else:
            self.ctx = None

        # Tell the value/blend task what the rollout horizon is, so its
        # "telescoping" mode can apply γ^H to the terminal V (no effect on
        # other modes — γ^H is a scalar multiplier).
        if self.cost_mode in ("value_shaped", "vla_track_value"):
            self.task.set_horizon_steps(self.horizon_steps)

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
        self.have_global_pose = False
        self.have_arm = False
        self.last_base_stamp = None
        self.last_odom = None
        self.last_global_pose = None
        self.last_arm = None

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
            PoseStamped, GLOBAL_POSE_TOPIC,
            self.global_pose_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Float64MultiArray, ARM_STATE_TOPIC,
            self.arm_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Float64MultiArray, vla_chunk_topic, self._vla_cb,
            qos_profile_sensor_data,
        )

        self.plan_pub = self.create_publisher(JointTrajectory, PLAN_TOPIC, 10)

        self.timer = self.create_timer(1.0 / plan_rate, self._plan_and_publish)

        if self.cost_mode == "vla_track_value":
            track_value_log = (
                f" w_track={self.w_track_resolved:g} "
                f"w_value={self.w_value_resolved:g}"
            )
        elif self.cost_mode == "value_shaped":
            track_value_log = f" w_value={self.w_value_resolved:g}"
        else:
            track_value_log = ""
        self.get_logger().info(
            f"MPPI+VLA planner ready (cost_mode={self.cost_mode}, "
            f"passthrough={self.passthrough}).{track_value_log} "
            f"Replanning at {plan_rate:.1f} Hz, "
            f"publishing on {PLAN_TOPIC}, horizon "
            f"{self.horizon_steps} steps ({plan_horizon:.2f}s). "
            f"VLA chunk topic: {vla_chunk_topic}, "
            f"tail_knots={self.vla_tail_knots}, tail_alpha={self.vla_tail_alpha}, "
            f"full_warmstart_on_new_chunk={self.vla_full_warmstart_on_new_chunk} "
            f"(alpha={self.vla_full_warmstart_alpha:.2f})."
        )
        self.is_first_tick = True

    # ── Callbacks ─────────────────────────────────────────────────────────
    def odom_cb(self, msg: Float64MultiArray):
        self.last_odom = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_odom = True

    def global_pose_cb(self, msg: PoseStamped):
        self.last_global_pose = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_global_pose = True

    def arm_cb(self, msg: Float64MultiArray):
        # Layout from trossen_client_ros_async.py: positions(7) + velocities(7).
        if len(msg.data) < ARM_STATE_LEN:
            self.get_logger().warn(
                f"Arm state msg too short ({len(msg.data)} < "
                f"{2 * ARM_STATE_LEN}); ignoring.",
                throttle_duration_sec=2.0,
            )
            return
        self.last_arm = (msg, self.get_clock().now().nanoseconds * 1e-9)
        self.have_arm = True

    def _vla_cb(self, msg: Float64MultiArray):
        """Cache a VLA action chunk (Float64MultiArray) in the node's clock.

        Wire format (see trossen_client_ros_chunk_publisher._publish_chunk):
          data    : flattened (N, action_dim) row-major
          layout  : MultiArrayLayout with dim[0]=N, dim[1]=action_dim
          no header, no per-point timestamps

        We anchor the chunk at the receive time and space samples by
        self.vla_chunk_dt, then re-order channels from the VLA layout
        [arm_0..arm_5, gripper, vx, vy, yaw_rate] into the MPPI ctrl layout
        [vx, vy, wz, arm_0..arm_5, gripper, dead] and invert the gripper to
        the task convention (0=closed, 1=open).
        """
        if not msg.data:
            self.get_logger().warn("VLA chunk empty; ignoring.")
            return

        # Recover (N, action_dim) shape from layout if present; fall back to
        # the publisher's documented action_dim=10 otherwise.
        action_dim = (
            int(msg.layout.dim[1].size)
            if len(msg.layout.dim) >= 2 and msg.layout.dim[1].size > 0
            else VLA_ACTION_DIM
        )
        total = len(msg.data)
        if action_dim <= 0 or total % action_dim != 0:
            self.get_logger().warn(
                f"VLA chunk size {total} not divisible by action_dim "
                f"{action_dim}; ignoring."
            )
            return
        n_pts = total // action_dim
        vla_chunk = np.asarray(msg.data, dtype=np.float32).reshape(
            n_pts, action_dim
        )

        # _t0_wall is established on the first plan tick. If a chunk shows up
        # before that, anchor it now so the math still works.
        if not hasattr(self, "_t0_wall"):
            self._t0_wall = self.get_clock().now().nanoseconds * 1e-9

        # Anchor at receive time (publisher has no header); space samples by
        # the policy's control period.
        now_node_t = self.get_clock().now().nanoseconds * 1e-9 - self._t0_wall
        times = (
            np.float32(now_node_t)
            + np.arange(n_pts, dtype=np.float32) * np.float32(self.vla_chunk_dt)
        )

        # Re-layout VLA -> MPPI ctrl. Guards on action_dim let us gracefully
        # accept odd publisher outputs (arm-only, no base, etc.).
        ctrls = np.zeros((n_pts, self.nu), dtype=np.float32)
        if action_dim >= 6:
            ctrls[:, 3:9] = vla_chunk[:, 0:6]                    # arm joints
        if action_dim >= 7:
            ctrls[:, 9] = vla_chunk[:, 6]                        # gripper (VLA convention; inverted below)
        if action_dim >= 10:
            ctrls[:, 0] = vla_chunk[:, 7]                        # vx_body
            ctrls[:, 1] = -vla_chunk[:, 8]                        # vy_body
            ctrls[:, 2] = vla_chunk[:, 9]                        # yaw_rate (wz)

        # VLA gripper convention (0=open, 1=close) -> MPPI task convention
        # (atmos_m3.py: ctrl[9] = 0=closed, 1=open). mppi_driver re-inverts
        # back to VLA convention before sending to the trossen helper.
        ctrls[:, 9] = 1.0 - ctrls[:, 9]

        # Clip into task control bounds so the seeded mean is always feasible.
        np.clip(ctrls, self.u_min, self.u_max, out=ctrls)

        self.vla_times = times
        self.vla_controls = ctrls
        self.vla_recv_node_t = now_node_t
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

    def _load_iql_norm_or_fail(self) -> dict:
        """Read the IQL dataset config JSON (obs/action normalization stats)."""
        cfg_path = str(self.get_parameter("iql_dataset_config_path").value)
        if not cfg_path:
            raise RuntimeError(
                f"cost_mode={self.cost_mode} needs iql_dataset_config_path "
                "(the JSON with observation_mean/std and action_min/max)."
            )
        return load_dataset_norm(cfg_path)

    def _build_iql_ctx(self, strict: bool) -> dict:
        """Build {value, critic, actor} param dict for the IQL-backed cost.

        If iql_ckpt_dir is set, loads the trained params from
        <dir>/{value,critic,actor}.ckpt and fills any missing entries with
        random init (so the dict always has consistent shape for JIT).
        If strict=True, refuses to start without iql_ckpt_dir.
        """
        ckpt_dir = str(self.get_parameter("iql_ckpt_dir").value)
        hidden_dims = tuple(int(x) for x in
                            self.get_parameter("iql_hidden_dims").value)
        # Random-init template for any missing entries — keeps the ctx
        # pytree shape stable across configs.
        ctx = self.task.init_iql_params(seed=0)
        if ckpt_dir:
            loaded = load_iql_bundle(
                ckpt_dir,
                action_dim=self.task.iql.action_dim,
                obs_dim=self.task.iql.obs_dim,
                hidden_dims=hidden_dims,
            )
            ctx.update(loaded)
            self.get_logger().info(
                f"Loaded IQL bundle from {ckpt_dir}: "
                f"{sorted(loaded.keys())}"
            )
        elif strict:
            raise RuntimeError(
                f"cost_mode={self.cost_mode} requires iql_ckpt_dir. "
                "If you want pure tracking while IQL is still training, "
                "use cost_mode=vla_track instead."
            )
        else:
            self.get_logger().warn(
                "value_shaped with no iql_ckpt_dir: using random IQL params. "
                "Cost signal will be untrained noise."
            )
        return ctx

    def _install_vla_ctx(self, new_vla_ctx) -> None:
        """Insert a fresh vla_ctx (ref_states, ref_t0, ref_dt) into self.ctx.

        In `vla_track` mode the ctx *is* the vla tuple; in `vla_track_value`
        mode it's a dict with a "vla" key (plus value/critic/actor IQL
        params) and we only swap the "vla" slot. Only called from the main
        thread.
        """
        if self.cost_mode == "vla_track":
            self.ctx = new_vla_ctx
        else:  # vla_track_value
            self.ctx = {**self.ctx, "vla": new_vla_ctx}

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

        # Base pose from /global_pose/ — assumed to be published in the same
        # world frame the planner has historically been driven in (the working
        # mppi_ros_node + driver setup). Read directly with no swap/offset.
        pose_msg, _ = self.last_global_pose
        pos = pose_msg.pose.position
        q = pose_msg.pose.orientation
        yaw = float(np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        ))

        qpos[0] = float(pos.x) - float(0.16665*np.cos(yaw))
        qpos[1] = float(pos.y) + float(0.16665*np.sin(yaw))
        qpos[2] = yaw

        # Base velocity: identical to the working mppi_ros_node mapping out of
        # /px4/bridge/vehicle_odometry. Do not change without re-validating on
        # hardware — the x/y swap matches the bridge's NED layout against the
        # model's world axes, and the unsigned yaw-rate read is what's been
        # flying.
        odom_msg, odom_t = self.last_odom
        qvel[0] = float(odom_msg.data[ODOM_LIN_VEL_START + 1])
        qvel[1] = float(odom_msg.data[ODOM_LIN_VEL_START])
        qvel[2] = float(odom_msg.data[BASE_VEL_YAW_RATE_INDEX])

        # Arm + gripper: positions(7) + velocities(7) from /follower/joint_state.
        # qpos[3:9] / qvel[3:9] = 6 arm joints; qpos[9:11] / qvel[9:11] = two
        # carriage joints, both held at the gripper opening (their axes are
        # mirrored so equal values open/close symmetrically).
        arm_msg, _ = self.last_arm
        arm_data = np.asarray(arm_msg.data, dtype=np.float32)
        arm_pos = arm_data[:ARM_STATE_LEN]
        arm_vel = arm_data[ARM_STATE_LEN:2 * ARM_STATE_LEN]
        qpos[3:9] = arm_pos[:6]
        qvel[3:9] = arm_vel[:6]
        qpos[9] = qpos[10] = float(arm_pos[6])
        qvel[9] = qvel[10] = float(arm_vel[6])

        self.last_base_stamp = RclTime(nanoseconds=int(odom_t * 1e9)).to_msg()
        return qpos, qvel

    # ── Plan + publish ────────────────────────────────────────────────────
    def _plan_and_publish(self):
        if not (self.have_odom and self.have_global_pose and self.have_arm and self.have_vla):
            self.get_logger().warn(
                f"Waiting for state... odom={self.have_odom} "
                f"global_pose={self.have_global_pose} arm={self.have_arm} "
                f"vla={self.have_vla}",
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

        # Passthrough: bypass MPPI; publish the VLA chunk directly resampled
        # at the plan's normal query times. If the chunk is stale we don't
        # publish, so the driver's plan_timeout watchdog safe-stops instead of
        # the arm tracking ancient setpoints.
        if self.passthrough:
            if chunk_stale:
                return
            tq_np = (
                np.float32(now_t)
                + np.arange(self.horizon_steps, dtype=np.float32) * self.sim_dt
            )
            us = self._vla_resample(tq_np)
            us = np.clip(us, self.u_min, self.u_max)
            self._publish_plan(us, frame_id="atmos_m3_vla_passthrough")
            return

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
            alpha = self.vla_full_warmstart_alpha
            if alpha >= 1.0:
                blended = seeded
            else:
                mean_np = np.array(self.policy_params.mean, dtype=np.float32)
                blended = np.clip(
                    alpha * seeded + (1.0 - alpha) * mean_np,
                    self.u_min, self.u_max,
                )
            new_mean = jnp.asarray(blended, dtype=self.policy_params.mean.dtype)
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

        self._publish_plan(us, frame_id="atmos_m3_mppi_vla")

    def _publish_plan(self, us: np.ndarray, frame_id: str) -> None:
        """Pack a (horizon_steps, nu) control sequence into JointTrajectory and publish.

        Shared by the optimized path and the passthrough branch; frame_id is
        the only thing that differs so consumers can tell them apart in logs.
        """
        msg = JointTrajectory()
        msg.header.stamp = (
            self.last_base_stamp
            if self.last_base_stamp is not None
            else self.get_clock().now().to_msg()
        )
        msg.header.frame_id = frame_id
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
