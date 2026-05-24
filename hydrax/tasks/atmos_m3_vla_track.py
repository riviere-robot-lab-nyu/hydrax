"""AtmosM3 variant that tracks a reference state trajectory from a VLA chunk.

Used together with `MPPI_WithCtx`. The `ctx` pytree expected by
`running_cost_ctx` / `terminal_cost_ctx` is:

    ctx = (ref_states, ref_t0, ref_dt)

    ref_states: (H_ref, n_track)  jax.float32  — uniformly-sampled reference
                                                 state slice (see _slice_state)
    ref_t0:     ()                jax.float32  — absolute node-clock time of
                                                 the FIRST reference sample
    ref_dt:     ()                jax.float32  — spacing between samples
                                                 (typically sim_dt)

Lookup uses nearest-index on the uniform grid, clamped to [0, H_ref-1].
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.atmos_m3 import AtmosM3


class AtmosM3VlaTrack(AtmosM3):
    """AtmosM3 with a state-tracking cost driven by a VLA-rollout reference.

    Args (in addition to AtmosM3's):
        base_track_weights: (6,) per-dim weight on (x, y, yaw, vx, vy, wz)
                            tracking error. Default emphasises pose over vel.
        arm_track_weights:  (6,) per-dim weight on arm-joint tracking error.
                            Default 0 → arm not tracked (flip on when arm is
                            wired up via ROS param).
        ctrl_reg:           Scalar weight on sum(control[:3]**2). Keeps the
                            base command modest. Set to 0 to remove.
        yaw_wrap:           If True (default), wrap the yaw tracking error to
                            [-pi, pi] so the cost doesn't blow up at the seam.
    """

    def __init__(
        self,
        base_track_weights: jax.Array = jnp.array([10.0, 10.0, 5.0, 1.0, 1.0, 0.1]),
        arm_track_weights: jax.Array = jnp.zeros(6),
        ctrl_reg: float = 0.1,
        yaw_wrap: bool = True,
        **atmos_kwargs,
    ) -> None:
        super().__init__(**atmos_kwargs)
        self.base_track_weights = jnp.asarray(base_track_weights, dtype=jnp.float32)
        self.arm_track_weights = jnp.asarray(arm_track_weights, dtype=jnp.float32)
        self.ctrl_reg = float(ctrl_reg)
        self.yaw_wrap = bool(yaw_wrap)

    @staticmethod
    def slice_state(state: mjx.Data) -> jax.Array:
        """The 12-D state slice that gets tracked.

        Order: [x, y, yaw, vx, vy, wz, arm0..arm5_pos].
        Gripper and dead dims are excluded (binary / unused).
        """
        return jnp.concatenate([
            state.qpos[0:3],   # x, y, yaw
            state.qvel[0:3],   # vx, vy, wz
            state.qpos[3:9],   # arm joints
        ])

    def _track_error(self, state: mjx.Data, s_ref: jax.Array) -> jax.Array:
        """Per-dim tracking error with optional yaw wrapping."""
        s = self.slice_state(state)
        err = s - s_ref
        if self.yaw_wrap:
            err = err.at[2].set((err[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        return err

    def _lookup_ref(self, ctx: Tuple[jax.Array, jax.Array, jax.Array],
                    t: jax.Array) -> jax.Array:
        """Nearest-index lookup on the uniform reference grid."""
        ref_states, ref_t0, ref_dt = ctx
        idx_f = (t - ref_t0) / ref_dt
        idx = jnp.clip(
            jnp.round(idx_f).astype(jnp.int32), 0, ref_states.shape[0] - 1
        )
        return ref_states[idx]

    def running_cost_ctx(
        self,
        state: mjx.Data,
        control: jax.Array,
        ctx: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        s_ref = self._lookup_ref(ctx, state.time)
        err = self._track_error(state, s_ref)
        weights = jnp.concatenate([self.base_track_weights, self.arm_track_weights])
        track = jnp.sum(weights * jnp.square(err))
        ctrl_pen = self.ctrl_reg * jnp.sum(jnp.square(control[:3]))
        return track + ctrl_pen

    def terminal_cost_ctx(
        self,
        state: mjx.Data,
        ctx: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        s_ref = self._lookup_ref(ctx, state.time)
        err = self._track_error(state, s_ref)
        weights = jnp.concatenate([self.base_track_weights, self.arm_track_weights])
        # Heavier terminal weight pulls the endpoint onto the reference.
        return 10.0 * jnp.sum(weights * jnp.square(err))
