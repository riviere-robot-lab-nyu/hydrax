"""AtmosM3 variant: VLA state-tracking + learned value-network cost.

Used together with `MPPI_WithCtx`. The `ctx` pytree is:

    ctx = (vla_ctx, value_params)

where `vla_ctx = (ref_states, ref_t0, ref_dt)` (see `AtmosM3VlaTrack`) and
`value_params` is a flax param tree for the value MLP (see `AtmosM3ValueShaped`).

Running cost per step:

    cost = w_track * tracking_term
         + w_value * value_term
         + ctrl_reg * sum(control[:3] ** 2)

with `tracking_term` from AtmosM3VlaTrack and `value_term = -V(s)` (or
`-Q(s,u)` if `value_kind == "q"`). Terminal cost mirrors AtmosM3VlaTrack's
10x weighting on the tracking term, plus the value term once.

The point of this mode is to keep the value signal honest while V is still
training: the VLA-rollout reference pulls MPPI toward sane motion even when
the value net is noisy or miscalibrated.
"""

from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.atmos_m3_vla_track import AtmosM3VlaTrack
from hydrax.tasks.atmos_m3_value import ValueMLP


VlaCtx = Tuple[jax.Array, jax.Array, jax.Array]


class AtmosM3VlaTrackValue(AtmosM3VlaTrack):
    """VLA-track + value-net cost, blended via two scalar weights.

    Args (in addition to AtmosM3VlaTrack's):
        w_track:    Multiplier on the VLA-tracking cost term.
        w_value:    Multiplier on the value-net cost term.
        value_kind: "v" -> cost = -V(s); "q" -> cost = -Q(s, u).
        hidden:     Width of the value MLP's hidden layers.
        out_scale:  Tanh-bound multiplier on value-net output.
    """

    def __init__(
        self,
        w_track: float = 1.0,
        w_value: float = 1.0,
        value_kind: Literal["v", "q"] = "v",
        hidden: int = 256,
        out_scale: float = 1.0,
        **vla_track_kwargs,
    ) -> None:
        super().__init__(**vla_track_kwargs)
        assert value_kind in ("v", "q"), value_kind
        self.w_track = float(w_track)
        self.w_value = float(w_value)
        self.value_kind = value_kind
        self.value_net = ValueMLP(hidden=hidden, out_scale=out_scale)
        nq = int(self.mj_model.nq)
        nv = int(self.mj_model.nv)
        nu = int(self.mj_model.nu)
        self.feature_dim_v = nq + nv
        self.feature_dim_q = nq + nv + nu

    def features_v(self, state: mjx.Data) -> jax.Array:
        return jnp.concatenate([state.qpos, state.qvel])

    def features_q(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        return jnp.concatenate([state.qpos, state.qvel, control])

    def init_value_params(self, seed: int = 0) -> Any:
        """Random init for the value-net param tree (shape only — node will
        usually overwrite with checkpoint weights)."""
        rng = jax.random.key(seed)
        if self.value_kind == "v":
            dummy = jnp.zeros(self.feature_dim_v, dtype=jnp.float32)
        else:
            dummy = jnp.zeros(self.feature_dim_q, dtype=jnp.float32)
        return self.value_net.init(rng, dummy)

    def _value_term(self, state: mjx.Data, control: jax.Array,
                    value_params: Any) -> jax.Array:
        if self.value_kind == "v":
            v = self.value_net.apply(value_params, self.features_v(state))
        else:
            v = self.value_net.apply(value_params, self.features_q(state, control))
        return -v

    def _track_term(self, state: mjx.Data, vla_ctx: VlaCtx) -> jax.Array:
        s_ref = self._lookup_ref(vla_ctx, state.time)
        err = self._track_error(state, s_ref)
        weights = jnp.concatenate([self.base_track_weights, self.arm_track_weights])
        return jnp.sum(weights * jnp.square(err))

    def running_cost_ctx(
        self,
        state: mjx.Data,
        control: jax.Array,
        ctx: Tuple[VlaCtx, Any],
    ) -> jax.Array:
        vla_ctx, value_params = ctx
        track = self._track_term(state, vla_ctx)
        val = self._value_term(state, control, value_params)
        ctrl_pen = self.ctrl_reg * jnp.sum(jnp.square(control[:3]))
        return self.w_track * track + self.w_value * val + ctrl_pen

    def terminal_cost_ctx(
        self,
        state: mjx.Data,
        ctx: Tuple[VlaCtx, Any],
    ) -> jax.Array:
        vla_ctx, value_params = ctx
        track = self._track_term(state, vla_ctx)
        # Q mode has no terminal control; stand in with zero control.
        u_zero = jnp.zeros(int(self.mj_model.nu), dtype=state.qpos.dtype)
        val = self._value_term(state, u_zero, value_params)
        return self.w_track * (10.0 * track) + self.w_value * val
