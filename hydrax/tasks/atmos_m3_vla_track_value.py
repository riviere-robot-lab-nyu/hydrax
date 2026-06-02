"""AtmosM3 variant: VLA state-tracking + IQL-shaped cost.

Used with ``MPPI_WithCtx``. The ``ctx`` pytree is:

    ctx = {"vla":     (ref_states, ref_t0, ref_dt),
           "value":   V params,
           "critic":  double-critic params,
           "actor":   tanh-Gaussian policy params}

The VLA half ("vla" key) is identical to ``AtmosM3VlaTrack``'s setup. The
three IQL param entries are mode-conditional — pass random init for the
unused ones (still required so JIT bakes in the shapes).

Running cost per step:

    cost = w_track * tracking_term
         + w_value * iql_running_term
         + ctrl_reg * sum(control[:3]**2)

Terminal cost mirrors ``AtmosM3VlaTrack``'s 10× weighting on the tracking
term plus the IQL terminal term once.

The ``iql_mode`` parameter has the same semantics as
``AtmosM3ValueShaped.iql_mode`` — see its docstring for the five settings.
The point of this mode is to keep MPPI sane when V/Q are noisy: the
VLA-rollout reference pulls the controller toward meaningful motion even
when the value signal is uninformative.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.atmos_m3_vla_track import AtmosM3VlaTrack
from hydrax.tasks.atmos_m3_value import IQLMode, _VALID_MODES
from hydrax.tasks.iql_nets import (
    AtmosM3IQLFeatures,
    IQLDoubleCritic,
    IQLNormalTanhPolicy,
    IQLValueNet,
    iql_actor_log_prob,
)


VlaCtx = Tuple[jax.Array, jax.Array, jax.Array]


class AtmosM3VlaTrackValue(AtmosM3VlaTrack):
    """VLA-track + IQL cost, blended via two scalar weights.

    Args (in addition to AtmosM3VlaTrack's):
        w_track:     Multiplier on the VLA-tracking cost term.
        w_value:     Multiplier on the IQL cost term.
        iql_mode:    Which IQL quantity drives the cost (see AtmosM3ValueShaped).
        norm:        Output of ``load_dataset_norm(...)``. Required.
        hidden_dims: MLP shape (must match checkpoints).
        gamma:       IQL training discount (used by "telescoping" mode).
    """

    def __init__(
        self,
        w_track: float = 1.0,
        w_value: float = 1.0,
        iql_mode: IQLMode = "v",
        norm: dict = None,
        hidden_dims: tuple = (256, 256),
        gamma: float = 0.99,
        **vla_track_kwargs,
    ) -> None:
        super().__init__(**vla_track_kwargs)
        assert iql_mode in _VALID_MODES, f"iql_mode must be one of {_VALID_MODES}"
        if norm is None:
            raise ValueError(
                "AtmosM3VlaTrackValue requires `norm` (from iql_nets.load_dataset_norm)."
            )
        self.w_track = float(w_track)
        self.w_value = float(w_value)
        self.iql_mode = iql_mode
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = float(gamma)

        # Shared feature/action plumbing.
        self.iql = AtmosM3IQLFeatures(self.mj_model, norm)
        self.value_net = IQLValueNet(hidden_dims=self.hidden_dims)
        self.critic_net = IQLDoubleCritic(hidden_dims=self.hidden_dims)
        self.actor_net = IQLNormalTanhPolicy(
            action_dim=self.iql.action_dim, hidden_dims=self.hidden_dims,
        )

        self._horizon_steps = None  # set by controller; γ^H for telescoping

    # ── Param init / horizon helpers ────────────────────────────────────
    def init_iql_params(self, seed: int = 0) -> dict:
        rng = jax.random.key(seed)
        rv, rc, ra = jax.random.split(rng, 3)
        obs_dummy = jnp.zeros((self.iql.obs_dim,), dtype=jnp.float32)
        act_dummy = jnp.zeros((self.iql.action_dim,), dtype=jnp.float32)
        return {
            "value": self.value_net.init(rv, obs_dummy)["params"],
            "critic": self.critic_net.init(rc, obs_dummy, act_dummy)["params"],
            "actor": self.actor_net.init(ra, obs_dummy)["params"],
        }

    def set_horizon_steps(self, H: int) -> None:
        self._horizon_steps = int(H)

    @property
    def gamma_pow_H(self) -> float:
        if self._horizon_steps is None:
            return 1.0
        return float(self.gamma ** self._horizon_steps)

    # ── Cost building blocks ────────────────────────────────────────────
    def _v(self, params: Any, obs: jax.Array) -> jax.Array:
        return self.value_net.apply({"params": params}, obs)

    def _q_min(self, params: Any, obs: jax.Array, a_norm: jax.Array) -> jax.Array:
        q1, q2 = self.critic_net.apply({"params": params}, obs, a_norm)
        return jnp.minimum(q1, q2)

    def _logp(self, params: Any, obs: jax.Array, a_norm: jax.Array) -> jax.Array:
        return iql_actor_log_prob(self.actor_net, params, obs, a_norm)

    def _iql_running(self, state: mjx.Data, control: jax.Array, ctx: dict) -> jax.Array:
        obs = self.iql.build_obs(state)
        if self.iql_mode == "v":
            return -self._v(ctx["value"], obs)
        if self.iql_mode == "telescoping":
            return jnp.float32(0.0)
        a_norm = self.iql.build_action_norm(control)
        if self.iql_mode == "q":
            return -self._q_min(ctx["critic"], obs, a_norm)
        if self.iql_mode == "advantage":
            v = self._v(ctx["value"], obs)
            q = self._q_min(ctx["critic"], obs, a_norm)
            return v - q
        if self.iql_mode == "logprob":
            return -self._logp(ctx["actor"], obs, a_norm)
        raise ValueError(self.iql_mode)

    def _iql_terminal(self, state: mjx.Data, ctx: dict) -> jax.Array:
        obs = self.iql.build_obs(state)
        if self.iql_mode == "telescoping":
            return -self.gamma_pow_H * self._v(ctx["value"], obs)
        return -self._v(ctx["value"], obs)

    def _track_term(self, state: mjx.Data, vla_ctx: VlaCtx) -> jax.Array:
        s_ref = self._lookup_ref(vla_ctx, state.time)
        err = self._track_error(state, s_ref)
        weights = jnp.concatenate([self.base_track_weights, self.arm_track_weights])
        return jnp.sum(weights * jnp.square(err))

    # ── MPPI_WithCtx interface ──────────────────────────────────────────
    def running_cost_ctx(
        self,
        state: mjx.Data,
        control: jax.Array,
        ctx: dict,
    ) -> jax.Array:
        track = self._track_term(state, ctx["vla"])
        iql = self._iql_running(state, control, ctx)
        ctrl_pen = self.ctrl_reg * jnp.sum(jnp.square(control[:3]))
        return self.w_track * track + self.w_value * iql + ctrl_pen

    def running_cost_batch_ctx(
        self,
        states: mjx.Data,
        controls: jax.Array,
        ctx: dict,
    ) -> jax.Array:
        """Batched running-cost over a (T,) stack of states + (T, nu) controls.

        Defers the IQL network forward until after the dynamics scan so the T
        per-step MLP launches fuse into a single batched forward (one launch
        per layer). Tracking term + ctrl_reg vmap along too — they're cheap,
        but co-locating keeps the cost signature clean.
        """
        return jax.vmap(self.running_cost_ctx, in_axes=(0, 0, None))(
            states, controls, ctx
        )

    def terminal_cost_ctx(
        self,
        state: mjx.Data,
        ctx: dict,
    ) -> jax.Array:
        track = self._track_term(state, ctx["vla"])
        iql = self._iql_terminal(state, ctx)
        return self.w_track * (10.0 * track) + self.w_value * iql
