"""AtmosM3 variant with a learned value (or Q) function as the running cost.

Used together with `MPPI_WithCtx`. The `ctx` pytree is just the flax
parameter pytree of the value network. The network architecture is fixed at
task construction (so JIT bakes the layer shapes); only the parameter values
change between ticks.

Cost shape (per-step, dense):

    V mode:   cost_t = - V(features(state))
    Q mode:   cost_t = - Q(features(state, control))

`features(state)` defaults to `jnp.concatenate([qpos, qvel])`. Override
`features` in a subclass if your IQL training used different inputs.

Network: a plain 3-layer MLP (ReLU, scalar output). Replace `value_net` with
your trained IQL critic by either (a) loading weights into the same MLP
shape or (b) subclassing and overriding `value_net`.

Checkpoint loading is left to the node — call `task.load_value_params(path)`
which expects a flat dict / pickle of param tree compatible with the MLP.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp
import flax.linen as nn
from mujoco import mjx

from hydrax.tasks.atmos_m3 import AtmosM3


class ValueMLP(nn.Module):
    """3-layer MLP, scalar output. Tanh-bounded to avoid exploding cost when
    the net is randomly initialized."""
    hidden: int = 256
    out_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return self.out_scale * jnp.tanh(x)[..., 0]


class AtmosM3ValueShaped(AtmosM3):
    """AtmosM3 with a value-network running cost.

    Args (in addition to AtmosM3's):
        value_kind:    "v" → cost = -V(s); "q" → cost = -Q(s, u).
        hidden:        Width of the value MLP's hidden layers.
        out_scale:     Multiplier on the tanh-bounded output. Tune to set the
                       value-cost magnitude relative to the rest of the cost.
        ctrl_reg:      Scalar weight on sum(control[:3]**2). Set to 0 to
                       remove. Keeps base command bounded when the value net
                       is uninformative.
    """

    def __init__(
        self,
        value_kind: Literal["v", "q"] = "v",
        hidden: int = 256,
        out_scale: float = 1.0,
        ctrl_reg: float = 0.1,
        **atmos_kwargs,
    ) -> None:
        super().__init__(**atmos_kwargs)
        assert value_kind in ("v", "q"), value_kind
        self.value_kind = value_kind
        self.ctrl_reg = float(ctrl_reg)
        self.value_net = ValueMLP(hidden=hidden, out_scale=out_scale)

        # Build dummy input to allow the node to call net.init(...) and
        # discover the param tree shape without knowing the feature dim.
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
        """Random init for the value-net parameter pytree.

        Returned pytree is exactly what `ctx` should carry into MPPI_WithCtx.
        """
        rng = jax.random.key(seed)
        if self.value_kind == "v":
            dummy = jnp.zeros(self.feature_dim_v, dtype=jnp.float32)
        else:
            dummy = jnp.zeros(self.feature_dim_q, dtype=jnp.float32)
        return self.value_net.init(rng, dummy)

    def running_cost_ctx(
        self,
        state: mjx.Data,
        control: jax.Array,
        ctx: Any,
    ) -> jax.Array:
        value_params = ctx
        if self.value_kind == "v":
            v = self.value_net.apply(value_params, self.features_v(state))
        else:
            v = self.value_net.apply(value_params, self.features_q(state, control))
        ctrl_pen = self.ctrl_reg * jnp.sum(jnp.square(control[:3]))
        return -v + ctrl_pen

    def terminal_cost_ctx(self, state: mjx.Data, ctx: Any) -> jax.Array:
        # Terminal cost only makes sense for V (no terminal control); for Q,
        # pass zero control through Q as a stand-in.
        value_params = ctx
        if self.value_kind == "v":
            v = self.value_net.apply(value_params, self.features_v(state))
        else:
            u_zero = jnp.zeros(int(self.mj_model.nu), dtype=state.qpos.dtype)
            v = self.value_net.apply(value_params, self.features_q(state, u_zero))
        return -v
