"""AtmosM3 variant with a learned IQL value head (and optionally Q/actor) as
the running / terminal cost.

Backed by checkpoints from ikostrikov/implicit_q_learning. The flax param
trees live in ``ctx`` (a dict) so MPPI_WithCtx can jit the rollout — only
values flow through, not shapes.

Five ``iql_mode`` settings:

    "v"           cost_t = -V(s_t)                        (dense V — uses every step)
    "q"           cost_t = -min(Q1, Q2)(s_t, a_t)         (dense Q)
    "telescoping" running_cost = 0; terminal_cost = -γ^T V(s_T)
                  Mathematically Σγ^t r = V(s_0) - γ^T V(s_T); V(s_0) is
                  constant per plan tick so it drops out of MPPI ranking,
                  leaving only -γ^T V(s_T). Use when you trust V to capture
                  long-horizon value and want to avoid the double-counting
                  the dense V form bakes in.
    "advantage"   cost_t = V(s_t) - min(Q1, Q2)(s_t, a_t) (post-step A; see
                  note in mppi_vla_node design discussion).
    "logprob"     cost_t = -log π(a_t | s_t)              (proxy for advantage
                  when V/Q aren't reliable; uses the tanh-squashed Gaussian
                  actor and its analytic log-det).

The IQL networks were trained on a 33-D obs (see iql_datasets/README.md) and
10-D normalized actions. ``AtmosM3IQLFeatures`` (in ``iql_nets``) handles
the (mjx state, MPPI control) → (obs, action) translation.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.tasks.iql_nets import (
    AtmosM3IQLFeatures,
    IQLDoubleCritic,
    IQLNormalTanhPolicy,
    IQLValueNet,
    iql_actor_log_prob,
)


# Valid iql_mode values.
IQLMode = Literal["v", "q", "telescoping", "advantage", "logprob"]
_VALID_MODES = ("v", "q", "telescoping", "advantage", "logprob")


class AtmosM3ValueShaped(AtmosM3):
    """AtmosM3 with an IQL-shaped cost.

    The ``ctx`` pytree is a dict with up to three flax param trees:

        {"value":  V params,
         "critic": double-critic params,
         "actor":  tanh-Gaussian policy params}

    Only the entries needed by ``iql_mode`` are read at runtime, but JIT
    bakes in the full shape — caller must always pass a fully-shaped dict
    (use ``hydrax.tasks.iql_nets.load_iql_bundle`` or random ``init`` for
    placeholders).

    Args:
        iql_mode:    Which IQL quantity drives the cost. See module docstring.
        norm:        Output of ``load_dataset_norm(...)``. Provides obs_dim,
                     action_dim, obs_mean/std, action_min/max.
        hidden_dims: Tuple of MLP hidden sizes. Must match the trained
                     checkpoint shape (IQL mujoco_config: (256, 256)).
        gamma:       Discount used at training time. Used only by the
                     "telescoping" mode to weight the terminal value.
        ctrl_reg:    Scalar weight on sum(ctrl[:3]**2). Mild keep-base-quiet
                     regularizer applied on top of the IQL term.
        atmos_kwargs: Forwarded to ``AtmosM3.__init__``.
    """

    def __init__(
        self,
        iql_mode: IQLMode = "v",
        norm: dict = None,
        hidden_dims: tuple = (256, 256),
        gamma: float = 0.99,
        ctrl_reg: float = 0.1,
        w_value: float = 1.0,
        **atmos_kwargs,
    ) -> None:
        super().__init__(**atmos_kwargs)
        assert iql_mode in _VALID_MODES, f"iql_mode must be one of {_VALID_MODES}"
        if norm is None:
            raise ValueError(
                "AtmosM3ValueShaped requires `norm` (from iql_nets.load_dataset_norm)."
            )
        self.iql_mode = iql_mode
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = float(gamma)
        self.ctrl_reg = float(ctrl_reg)
        self.w_value = float(w_value)

        # Feature/action plumbing (stateless, just shape + normalization).
        self.iql = AtmosM3IQLFeatures(self.mj_model, norm)

        # Network defs (params live in ctx).
        self.value_net = IQLValueNet(hidden_dims=self.hidden_dims)
        self.critic_net = IQLDoubleCritic(hidden_dims=self.hidden_dims)
        self.actor_net = IQLNormalTanhPolicy(
            action_dim=self.iql.action_dim, hidden_dims=self.hidden_dims,
        )

        # Pre-compute γ^T (for "telescoping" terminal weighting). Horizon is
        # not directly exposed on the task (it lives on the controller), but
        # for the terminal multiplier we only care about a scalar that doesn't
        # affect ranking (γ^T is the same for every rollout). Default to 1.0
        # and let the controller set it via `set_horizon_steps` if desired.
        self._horizon_steps = None  # filled by controller; defaults to 1.0

    # ── Param-tree init helpers (random, for warm-up / placeholders) ────
    def init_iql_params(self, seed: int = 0) -> dict:
        """Random init for {value, critic, actor} param trees.

        Use when no checkpoint is provided OR for the JIT warm-up where the
        node hasn't yet loaded real weights but needs a correctly-shaped ctx.
        """
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
        """Set the rollout horizon (used for γ^T in telescoping mode)."""
        self._horizon_steps = int(H)

    @property
    def gamma_pow_H(self) -> float:
        """γ^horizon_steps. Falls back to 1.0 if horizon never set (no harm:
        it's a constant scalar that doesn't change MPPI ranking)."""
        if self._horizon_steps is None:
            return 1.0
        return float(self.gamma ** self._horizon_steps)

    # ── Cost helpers ────────────────────────────────────────────────────
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
            # No per-step IQL cost; everything is at terminal.
            return jnp.float32(0.0)
        # Modes that need the action go through the action mapper.
        a_norm = self.iql.build_action_norm(control)
        if self.iql_mode == "q":
            return -self._q_min(ctx["critic"], obs, a_norm)
        if self.iql_mode == "advantage":
            v = self._v(ctx["value"], obs)
            q = self._q_min(ctx["critic"], obs, a_norm)
            return v - q  # = -(Q - V) = -A
        if self.iql_mode == "logprob":
            return -self._logp(ctx["actor"], obs, a_norm)
        # Unreachable (constructor asserts).
        raise ValueError(self.iql_mode)

    def _iql_terminal(self, state: mjx.Data, ctx: dict) -> jax.Array:
        obs = self.iql.build_obs(state)
        if self.iql_mode == "telescoping":
            return -self.gamma_pow_H * self._v(ctx["value"], obs)
        # For all other modes the standard pattern is to also terminate
        # with -V(s_T) — the dense path benefits from a bootstrapped tail.
        return -self._v(ctx["value"], obs)

    # ── MPPI_WithCtx interface ──────────────────────────────────────────
    def running_cost_ctx(
        self,
        state: mjx.Data,
        control: jax.Array,
        ctx: dict,
    ) -> jax.Array:
        iql_term = self._iql_running(state, control, ctx)
        ctrl_pen = self.ctrl_reg * jnp.sum(jnp.square(control[:3]))
        return self.w_value * iql_term + ctrl_pen

    def running_cost_batch_ctx(
        self,
        states: mjx.Data,
        controls: jax.Array,
        ctx: dict,
    ) -> jax.Array:
        """Batched running-cost over a (T,) stack of states + (T, nu) controls.

        MPPI_WithCtx.eval_rollouts uses this when present to defer the
        running-cost MLP calls until after the dynamics scan, fusing T per-step
        MLP launches into a single batched forward. Same FLOPs as the scan
        path, but one kernel launch per layer instead of T.
        """
        return jax.vmap(self.running_cost_ctx, in_axes=(0, 0, None))(
            states, controls, ctx
        )

    def terminal_cost_ctx(self, state: mjx.Data, ctx: dict) -> jax.Array:
        return self.w_value * self._iql_terminal(state, ctx)
