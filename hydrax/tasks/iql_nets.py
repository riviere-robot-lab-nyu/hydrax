"""IQL network defs + atmos_m3 feature/action plumbing.

Mirrors ikostrikov/implicit_q_learning so checkpoints from the upstream
trainer load straight in via flax.serialization. Specifically:

  - ``IQLValueNet``  matches ``ValueCritic``  (MLP((*h, 1)) on obs).
  - ``IQLCritic``    matches ``Critic``       (MLP((*h, 1)) on concat(obs, act)).
  - ``IQLDoubleCritic`` matches ``DoubleCritic`` (two ``Critic`` heads, returns tuple).
  - ``IQLNormalTanhPolicy`` matches ``NormalTanhPolicy`` (state-dependent
    log_std heads, tanh-squashed Gaussian). Includes a self-contained
    ``log_prob`` implementation that does NOT require tensorflow_probability,
    so the inference-side dep stack stays small.

Checkpoint loading:

  IQL's ``Model.save`` writes ``flax.serialization.to_bytes(self.params)``
  where ``self.params`` is the FrozenDict *inside* ``{"params": {...}}``.
  Top-level keys are module-named (``MLP_0``, ``Critic_0``, ``Dense_0`` ...),
  no ``params`` wrapper. ``load_params`` rebuilds the template via
  ``net.init(...)["params"]`` and unmarshals onto it.

Feature builder (atmos_m3 → 33-D IQL obs):
  see ``AtmosM3IQLFeatures``. Layout per
  iql_datasets/README.md → "Observation Definition":

      0:7   arm joint positions  (qpos[3:10])
      7:9   forward, right body velocity  (rotated from world qvel[0:2])
      9:12  ang_vel_body (x, y, z); planar robot → (0, 0, qvel[2])
      12:15 world pose (x, y, yaw) = qpos[0:3]
      15:24 carabiner_L pos(3) + rot6d(6) from mjx free body
      24:33 carabiner_R pos(3) + rot6d(6) from mjx free body

  Normalized via stored (obs_mean, obs_std).

Action mapper (MPPI u → 10-D IQL action then minmax→[-1,1]):
  MPPI ctrl layout (CONTROL_NAMES in mppi_vla_node):
      u[0] vx_body, u[1] vy_body (left+), u[2] wz, u[3:9] arm 0..5,
      u[9] gripper, u[10] _dead.
  IQL action layout (dataset README):
      [joint_0..6, forward_v, right_v, yaw_rate]
  Mapping:
      iql_a[0:6] = u[3:9]   (arm joints)
      iql_a[6]   = u[9]     (gripper)
      iql_a[7]   = u[0]     (forward)
      iql_a[8]   = -u[1]    (right = -left)
      iql_a[9]   = u[2]     (yaw rate)
  Then minmax to [-1, 1] using (action_min, action_max) from the
  training-time config.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization

import mujoco


# ── Network defs (mirroring implicit_q_learning) ──────────────────────────

def _orthogonal_init(scale: float = float(jnp.sqrt(2))):
    return nn.initializers.orthogonal(scale)


class _MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=_orthogonal_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x


class IQLValueNet(nn.Module):
    """V(s) — output: scalar (squeezed last dim)."""
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        out = _MLP((*self.hidden_dims, 1))(obs)
        return jnp.squeeze(out, -1)


class _IQLCritic(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        inp = jnp.concatenate([obs, act], -1)
        out = _MLP((*self.hidden_dims, 1), activations=self.activations)(inp)
        return jnp.squeeze(out, -1)


class IQLDoubleCritic(nn.Module):
    """Returns (Q1, Q2); use min(Q1, Q2) for clipped double Q at inference."""
    hidden_dims: Sequence[int] = (256, 256)
    activations: Callable = nn.relu

    @nn.compact
    def __call__(
        self, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        q1 = _IQLCritic(self.hidden_dims, self.activations)(obs, act)
        q2 = _IQLCritic(self.hidden_dims, self.activations)(obs, act)
        return q1, q2


# Bounds matching implicit_q_learning/policy.py.
_LOG_STD_MIN = -10.0
_LOG_STD_MAX = 2.0


class IQLNormalTanhPolicy(nn.Module):
    """Tanh-squashed Gaussian actor.

    `apply(params, obs)` returns `(mean, log_std)` — the pre-squash Gaussian
    parameters. Use ``log_prob`` (below) to evaluate log π(a | s) on an
    action already in the tanh-output domain a ∈ (-1, 1).
    """
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        out = _MLP(self.hidden_dims, activate_final=True)(obs)
        mean = nn.Dense(self.action_dim, kernel_init=_orthogonal_init())(out)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal_init(self.log_std_scale),
        )(out)
        log_std = jnp.clip(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std


def iql_actor_log_prob(
    net: IQLNormalTanhPolicy,
    params: Any,
    obs: jax.Array,
    action_norm: jax.Array,
) -> jax.Array:
    """log π(a | s) for the tanh-squashed Gaussian.

    `action_norm` is in the tanh-output domain (-1, 1) — i.e. the actor's
    direct sample space, which lines up with the minmax-normalized IQL action.

    Returns the scalar log-prob (sum over action dims).
    """
    mean, log_std = net.apply({"params": params}, obs)
    std = jnp.exp(log_std)
    # Strict interior to keep atanh finite. 1e-6 ≈ 6 std past saturation
    # — beyond that the IQL policy's log_prob is already off the chart.
    a_clip = jnp.clip(action_norm, -1.0 + 1e-6, 1.0 - 1e-6)
    x = jnp.arctanh(a_clip)
    # Diagonal Gaussian log-prob on the pre-tanh sample x.
    log_p_gauss = -0.5 * jnp.sum(
        jnp.square((x - mean) / std) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    # Tanh log-det Jacobian correction (subtract because dist is the
    # push-forward through tanh): log|d tanh(x)/dx| = log(1 - tanh(x)^2).
    log_det = jnp.sum(jnp.log1p(-jnp.square(a_clip)))
    return log_p_gauss - log_det


# ── Checkpoint loading ────────────────────────────────────────────────────

def load_iql_params(
    net: nn.Module,
    template_inputs: Tuple[jax.Array, ...],
    path: str,
    seed: int = 0,
) -> Any:
    """Load flax-serialized params from one IQL .ckpt."""
    rng = jax.random.key(seed)
    template = net.init(rng, *template_inputs)["params"]
    with open(path, "rb") as f:
        raw = f.read()
        
    state_dict = serialization.msgpack_restore(raw)

    # 1. Handle Actor case explicitly if we detect 'log_stds' array
    if "log_stds" in state_dict:
        fixed_state_dict = {}
        
        # Remap MLP_0 -> _MLP_0
        if "MLP_0" in state_dict:
            fixed_state_dict["_MLP_0"] = state_dict["MLP_0"]
        if "Dense_0" in state_dict:
            fixed_state_dict["Dense_0"] = state_dict["Dense_0"]
            
        # The upstream 'log_stds' is an array. Our template expects a Dense layer.
        # We will forge a fake Dense layer structure inside our dict using the checkpoint values.
        log_std_array = state_dict["log_stds"]
        
        # Map it to Dense_1's bias, and zero out or copy to kernel to keep shapes happy
        fixed_state_dict["Dense_1"] = {
            "bias": jnp.array(log_std_array),
            # Extract expected kernel shape from template to ensure exact match
            "kernel": jnp.zeros(template["Dense_1"]["kernel"].shape, dtype=jnp.float32)
        }
    
    # 2. Handle Value/Critic case with a safe, non-destructive key swap
    else:
        fixed_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k == "Critic_0": new_key = "_IQLCritic_0"
            elif k == "Critic_1": new_key = "_IQLCritic_1"
            elif k == "MLP_0": new_key = "_MLP_0"
            
            # Non-recursive nested fix for Critic internal MLP blocks
            if isinstance(v, dict) and "MLP_0" in v:
                v = dict(v) # break reference
                v["_MLP_0"] = v.pop("MLP_0")
                
            fixed_state_dict[new_key] = v

    fixed_raw = serialization.msgpack_serialize(fixed_state_dict)
    loaded = serialization.from_bytes(template, fixed_raw)
    # from_bytes returns numpy (host) arrays — push onto the default JAX
    # device so jit_optimize doesn't pay a host→device transfer for the
    # entire param pytree on every call. Without this, ctx that's "just"
    # ~1 MB of params can dominate per-tick latency.
    return jax.device_put(loaded)


def load_iql_bundle(
    ckpt_dir: str,
    action_dim: int,
    obs_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
) -> dict:
    """Load V, Q (double), and actor params from a ``step_NNNNNNN`` folder.

    Returns a dict ``{"value": ..., "critic": ..., "actor": ...}``. Missing
    files raise FileNotFoundError; the caller decides whether that's fatal
    for the chosen cost mode.
    """
    paths = {
        "value": os.path.join(ckpt_dir, "value.ckpt"),
        "critic": os.path.join(ckpt_dir, "critic.ckpt"),
        "actor": os.path.join(ckpt_dir, "actor.ckpt"),
    }
    obs_dummy = jnp.zeros((obs_dim,), dtype=jnp.float32)
    act_dummy = jnp.zeros((action_dim,), dtype=jnp.float32)

    out = {}
    if os.path.exists(paths["value"]):
        out["value"] = load_iql_params(
            IQLValueNet(hidden_dims=tuple(hidden_dims)),
            (obs_dummy,),
            paths["value"],
        )
    if os.path.exists(paths["critic"]):
        out["critic"] = load_iql_params(
            IQLDoubleCritic(hidden_dims=tuple(hidden_dims)),
            (obs_dummy, act_dummy),
            paths["critic"],
        )
    if os.path.exists(paths["actor"]):
        out["actor"] = load_iql_params(
            IQLNormalTanhPolicy(
                action_dim=action_dim, hidden_dims=tuple(hidden_dims)
            ),
            (obs_dummy,),
            paths["actor"],
        )
    return out


def load_dataset_norm(config_path: str) -> dict:
    """Read obs mean/std and action min/max from the dataset config JSON.

    Validates shapes against config['observation_dim'] / 'action_dim'.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    obs_dim = int(cfg["observation_dim"])
    act_dim = int(cfg["action_dim"])
    obs_mean = jnp.asarray(cfg["observation_mean"], dtype=jnp.float32)
    obs_std = jnp.asarray(cfg["observation_std"], dtype=jnp.float32)
    action_min = jnp.asarray(cfg["action_min"], dtype=jnp.float32)
    action_max = jnp.asarray(cfg["action_max"], dtype=jnp.float32)
    assert obs_mean.shape == (obs_dim,), obs_mean.shape
    assert obs_std.shape == (obs_dim,), obs_std.shape
    assert action_min.shape == (act_dim,), action_min.shape
    assert action_max.shape == (act_dim,), action_max.shape
    return {
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "obs_mean": obs_mean,
        # Floor std to avoid divide-by-zero on flat dims.
        "obs_std": jnp.maximum(
            obs_std, jnp.float32(cfg.get("observation_std_floor", 1e-6))
        ),
        "action_min": action_min,
        "action_max": action_max,
    }


# ── Atmos M3 obs / action plumbing ────────────────────────────────────────

LEFT_CARABINER_BODY = "left_carabiner"
RIGHT_CARABINER_BODY = "right_carabiner"


def _safe_body_id(mj_model: mujoco.MjModel, name: str) -> int:
    """Body id by name; -1 if absent (use_robot_only or stripped scene)."""
    try:
        return int(mj_model.body(name).id)
    except (KeyError, ValueError):
        return -1


class AtmosM3IQLFeatures:
    """Maps (mjx.Data state, MPPI control u) → IQL obs/action.

    Stateless w.r.t. flax — no params, just shape/normalization constants.
    Lives on the task instance so JIT traces the constants once.
    """

    def __init__(self, mj_model: mujoco.MjModel, norm: dict) -> None:
        self.obs_mean = norm["obs_mean"]
        self.obs_std = norm["obs_std"]
        self.action_min = norm["action_min"]
        self.action_max = norm["action_max"]
        self.obs_dim = int(norm["obs_dim"])
        self.action_dim = int(norm["action_dim"])
        self.left_car_id = _safe_body_id(mj_model, LEFT_CARABINER_BODY)
        self.right_car_id = _safe_body_id(mj_model, RIGHT_CARABINER_BODY)

    # ---- obs ----
    def _car_pos_rot6d(self, state, body_id: int) -> jax.Array:
        """(3,) xpos + (6,) rot6d from xmat first two columns. Zeros if absent."""
        if body_id < 0:
            return jnp.zeros(9, dtype=jnp.float32)
        pos = state.xpos[body_id]
        xmat = state.xmat[body_id].reshape(3, 3)
        col0 = xmat[:, 0]
        col1 = xmat[:, 1]
        return jnp.concatenate([pos, col0, col1])

    def build_obs(self, state) -> jax.Array:
        """(obs_dim,) normalized IQL obs vector."""
        qpos = state.qpos
        qvel = state.qvel

        # 0:7 arm joints — qpos[3:9] is arm 0..5, qpos[9] is one of the
        # two gripper carriages (both held equal by _assemble_state).
        joint_pos = jnp.concatenate([qpos[3:9], qpos[9:10]])  # (7,)

        # 7:9 body forward/right velocity.
        # qvel[0:2] is world-frame in the model's world axes (see
        # mppi_vla_node._assemble_state). Rotate by -yaw to get body frame.
        # Body frame conv (per dataset README): forward = +x_body, right = -y_body.
        yaw = qpos[2]
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        vx_w = qvel[0]
        vy_w = qvel[1]
        forward_body = vx_w * c + vy_w * s
        left_body = -vx_w * s + vy_w * c
        right_body = -left_body

        # 9:12 ang vel body. Planar atmos rotates only about z, no roll/pitch
        # rates available from the model's free-body chain, so x/y default to 0.
        ang_vx_body = jnp.float32(0.0)
        ang_vy_body = jnp.float32(0.0)
        ang_vz_body = qvel[2]

        body_vel = jnp.array([
            forward_body, right_body,
            ang_vx_body, ang_vy_body, ang_vz_body,
        ])

        # 12:15 world pose (x, y, yaw). Already in the same world frame the
        # IQL dataset used (FLU per README; matches /global_pose/).
        world_pose = jnp.array([qpos[0], qpos[1], qpos[2]])

        # 15:33 carabiner_L (9) + carabiner_R (9).
        left_car = self._car_pos_rot6d(state, self.left_car_id)
        right_car = self._car_pos_rot6d(state, self.right_car_id)

        obs_raw = jnp.concatenate(
            [joint_pos, body_vel, world_pose, left_car, right_car]
        )
        return (obs_raw - self.obs_mean) / self.obs_std

    # ---- action ----
    def build_action_norm(self, control: jax.Array) -> jax.Array:
        """MPPI 11-D u → IQL 10-D action in [-1, 1] (tanh / minmax domain).

        Both spaces are dense; out-of-range minmax inputs are clipped to
        keep the actor's log_prob (and Q's training-domain Q) sane.
        """
        # See module docstring for mapping rationale.
        a_raw = jnp.array([
            control[3], control[4], control[5], control[6], control[7], control[8],
            control[9],
            control[0],
            -control[1],
            control[2],
        ])
        a_norm = 2.0 * (a_raw - self.action_min) / (self.action_max - self.action_min) - 1.0
        return jnp.clip(a_norm, -1.0, 1.0)
