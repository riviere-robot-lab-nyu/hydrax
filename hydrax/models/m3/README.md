# ATMOS M3 – Planar Base + WidowX AI Arm

Stripped-down planar model for hydrax/MJX training.
Tiled ground, standard 3-point lighting, no skybox.

## Files

| File | Purpose |
|------|---------|
| `scene.xml` | **Main model** – tiled ground, lighting, full robot. Load this in the task. |
| `atmos_robot.xml` | Same robot, no scene decorations – lightweight for isolated tests. |
| `meshes/` | STL meshes and textures (shared with `atmos_rrl`). |
| `hydrax/tasks/atmos_m3.py` | `AtmosM3(Task)` – cost function, ctrl_transform, arm modes. |
| `hydrax/tasks/atmos_control.py` | JAX-jittable helpers for the drone control pipeline. |

## State / ctrl layout

`nq = nv = nu = 11`.  `ctrl[i]` maps to `qpos[i]` for the arm (indices 3–10).

| idx | joint name | type | ctrl meaning | units |
|-----|-----------|------|-------------|-------|
| 0 | `slide_x` | prismatic | vx_cmd (body frame) | m/s |
| 1 | `slide_y` | prismatic | vy_cmd (body frame) | m/s |
| 2 | `hinge_z` | revolute | wz_cmd (yaw rate) | rad/s |
| 3 | `left/joint_0` | revolute | shoulder pan target/delta | rad |
| 4 | `left/joint_1` | revolute | shoulder lift target/delta | rad |
| 5 | `left/joint_2` | revolute | elbow target/delta | rad |
| 6 | `left/joint_3` | revolute | wrist angle target/delta | rad |
| 7 | `left/joint_4` | revolute | wrist rotate target/delta | rad |
| 8 | `left/joint_5` | revolute | wrist servo target/delta | rad |
| 9 | `left/right_carriage_joint` | prismatic | right gripper target/delta | m |
| 10 | `left/left_carriage_joint` | prismatic | left gripper target/delta | m |

`state.qpos` and `state.qvel` follow the same ordering.
Velocities 0–2 are in **world frame** (MuJoCo slide/hinge convention).

## What MPPI actually searches over

MPPI samples **velocity commands** (base) and **joint targets or deltas** (arm),
not forces. `ctrl_transform()` converts them to actuator commands before each
`mjx.step`.

### Base (ctrl[0:3])
Sampled in velocity space; `ctrl_transform` runs the drone inner-loop PD law
to produce world-frame forces.  Bounds set directly on `task.u_min/u_max`.

| | min | max |
|--|-----|-----|
| vx_cmd | −2.0 m/s | 2.0 m/s |
| vy_cmd | −2.0 m/s | 2.0 m/s |
| wz_cmd | −1.0 rad/s | 1.0 rad/s |

### Arm (ctrl[3:11]) — two modes

**`arm_mode="absolute"` (default)**
MPPI samples target joint angles directly.
`u_min/u_max` = joint hard limits (e.g. ±3.05 rad for joint_0).
`ctrl_transform` clips the targets to limits and passes them to position servos.

**`arm_mode="delta"`**
MPPI samples small angle increments around zero.
`ctrl_transform` adds the delta to the current `qpos` and clips to joint limits.
Useful when the arm goal is implicit (e.g. follow an EE trajectory) or when you
want the optimizer to think in terms of motion rather than configuration.

Default delta bounds per step: ±0.1 rad (joints), ±0.005 m (grippers).

## Arm actuators – Trossen position servos

```
kp=450  dampratio=0.95  forcerange=±55 N  (from wxai defaults class)
```
`inheritrange=1` sets ctrlrange = joint range automatically, so MuJoCo clips
arm targets to joint limits as a safety net (in addition to `ctrl_transform`).

Arm joint limits:

| joint | min (rad) | max (rad) |
|-------|---------|---------|
| joint_0 | −3.054 | 3.054 |
| joint_1 | 0 | π |
| joint_2 | 0 | 2.356 |
| joint_3 | −π/2 | π/2 |
| joint_4 | −π/2 | π/2 |
| joint_5 | −π | π |
| gripper (×2) | 0 m | 0.044 m |

## Core hydrax framework changes

These two files were modified to support the `ctrl_transform` hook. The changes
are backward-compatible — all existing tasks are unaffected because the default
implementation is the identity.

### `hydrax/task_base.py`

Added a new overridable method to the `Task` base class:

```python
def ctrl_transform(self, state: mjx.Data, ctrl: jax.Array) -> jax.Array:
    """Transform sampled controls into actuator commands.
    Default: identity (pass-through). Override in subclasses."""
    return ctrl
```

### `hydrax/alg_base.py`

`SamplingBasedController.eval_rollouts._scan_fn` was modified to call
`ctrl_transform` before writing to `data.ctrl` and stepping the physics:

```python
# Before (original):
x = x.replace(ctrl=u)
x = mjx.step(model, x)
cost = self.dt * self.task.running_cost(x, u)

# After:
actual_ctrl = self.task.ctrl_transform(x, u)
x = x.replace(ctrl=actual_ctrl)
x = mjx.step(model, x)
cost = self.dt * self.task.running_cost(x, u)  # cost still receives u, not actual_ctrl
```

### `hydrax/algs/mppi.py` — NOT updated

`mppi.py` contains two MPPI variant classes (`MPPI_ctrl_chunk` and
`MPPI_bangbang`) that each override `eval_rollouts` with their own `_scan_fn`.
Those overrides write `ctrl` directly to `data.ctrl` and **do not** call
`ctrl_transform`. If you use `AtmosM3` with one of those variants, the
velocity-command → force conversion will be skipped. Use the base `MPPI` class
(which inherits `eval_rollouts` from `alg_base`) with this task.

---

## ctrl_transform – the inner-loop control law

`task_base.Task` exposes a `ctrl_transform(state, ctrl) -> jax.Array` hook.
`alg_base._scan_fn` calls it between sampling and stepping:

```python
actual_ctrl = self.task.ctrl_transform(x, u)   # u = [vel_cmd | arm_targets]
x = x.replace(ctrl=actual_ctrl)                # actual_ctrl = [forces | angles]
x = mjx.step(model, x)
cost = self.dt * self.task.running_cost(x, u)  # cost still sees sampled u
```

`AtmosM3.ctrl_transform` does:
```
ctrl[0:3]  velocity cmd (body frame)
  → rotate qvel to body frame via yaw
  → PD: wrench_body = kp * (v_cmd − v_body)
  → rotate wrench to world frame
  → clip to ±25 N / ±8 N·m
  → written to data.ctrl[0:3]

ctrl[3:11]  arm commands
  absolute: clip to joint limits  → data.ctrl[3:11]
  delta:    qpos[3:11] + ctrl[3:11], clip  → data.ctrl[3:11]
```

### Why `ctrllimited="false"` on base actuators

MuJoCo clips `data.ctrl` to `ctrlrange` *before* applying actuator gain.
The base actuators have no ctrlrange so the converted forces pass through.
Velocity bounds the optimizer sees come from `self.u_min/u_max` set in
`AtmosM3.__init__`.  Arm position servos keep `inheritrange=1` (their
ctrlrange = joint limits), which correctly clips stale/extreme targets.

## Full drone control pipeline (real-robot deployment)

```
velocity command  [vx_cmd, vy_cmd, ωz_cmd]  ← body frame
        │  vel_to_body_wrench()  (PD, gains kp)
        ▼
body-frame wrench  [Fx_b, Fy_b, Tz]
        │
        ├─► body_wrench_to_world_ctrl()  → forces → data.ctrl[0:3] → MJX
        │
        └─► body_wrench_to_thrusters()  → [T_f, T_b, T_r, T_l] → real robot
```

Helpers in `hydrax/tasks/atmos_control.py`, all `jax.jit`-able.

### Thruster allocation (cross, r = 0.15 m)

```
         [front]  fires +y_b
            ↑
[left] ←  CoM  → [right]  fires +x_b
            ↓
         [back]   fires −y_b
```

```
T_f =  Fy_b/2 + Tz/(4r)
T_b = -Fy_b/2 + Tz/(4r)
T_r =  Fx_b/2 + Tz/(4r)
T_l = -Fx_b/2 + Tz/(4r)
```

## Quick start

```python
import jax.numpy as jnp
from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.algs.mppi import MPPI

# Absolute arm mode (MPPI plans joint angles directly)
task = AtmosM3(
    goal=jnp.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0]),   # [x, y, θ, vx, vy, ωz]
    state_cost=jnp.array([100., 100., 5., 10., 10., 2.]),
    ctrl_cost=jnp.array([0.1, 0.1, 0.5]),
    arm_ctrl_cost=jnp.zeros(8),                          # set non-zero to penalise arm motion
    kp=jnp.array([50., 50., 20.]),
    arm_mode="absolute",
)

# Delta arm mode (MPPI plans small increments)
task_delta = AtmosM3(arm_mode="delta")

ctrl = MPPI(task, num_samples=512, noise_level=0.3, temperature=0.1)
params = ctrl.init_params()
# params.mean shape: (num_knots, 11)
#   [:, 0:3]  velocity commands  ∈ [−2,2] × [−2,2] × [−1,1]
#   [:, 3:11] joint targets      ∈ joint limits  (absolute mode)
```

### Swapping in your own drone control law

```python
from hydrax.tasks.atmos_m3 import AtmosM3
from mujoco import mjx
import jax.numpy as jnp

class MyAtmos(AtmosM3):
    def ctrl_transform(self, state: mjx.Data, ctrl: jax.Array) -> jax.Array:
        # ctrl[0:3] = [vx_cmd, vy_cmd, wz_cmd] body frame
        # ctrl[3:11] = arm joint targets/deltas
        # state.qpos = [x, y, θ, j0…j5, gr, gl]
        # state.qvel = [vx_w, vy_w, wz, j0_dot…]
        # Must return shape (11,) of actuator commands — pure JAX ops only.
        ...
```
