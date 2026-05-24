# `hydrax_on_thor` — ROS 2 nodes for AtmosM3 MPPI on the Thor (Jetson)

ROS 2 (Jazzy) wrapper around the hydrax MPPI planner targeting the AtmosM3
platform: planar drone base + 6-DoF WidowX arm + 1-DoF binary gripper.
Everything in this directory runs inside the docker image defined in
[docker_stuff/](docker_stuff/).

---

## 1. Files at a glance

| File | Role | Pairs with |
|------|------|------------|
| [mppi_ros_node.py](mppi_ros_node.py) | Original MPPI planner. State → optimized plan. | [mppi_driver.py](mppi_driver.py) or [mppi_ros_sim_node.py](mppi_ros_sim_node.py) |
| [mppi_vla_node.py](mppi_vla_node.py) | MPPI planner + VLA action-chunk warm-start, with 4 selectable cost modes. Superset of `mppi_ros_node.py`. | same as above, plus a VLA chunk publisher |
| [mppi_driver.py](mppi_driver.py) | Trajectory follower → real robot. Plan slicer, watchdogs, saturation, slew-limit. | a planner + the PX4/arm bridge |
| [mppi_ros_sim_node.py](mppi_ros_sim_node.py) | Headless MuJoCo simulator that closes the loop in software and records video. | a planner |
| [mppi_ros_test_driver.py](mppi_ros_test_driver.py) | Dummy state publisher for end-to-end testing of `mppi_ros_node.py` without a robot. | `mppi_ros_node.py` (only) |
| [docker_stuff/](docker_stuff/) | `Dockerfile` (jax + mjx + flax + ROS 2 Jazzy) and `run_container.sh`. | — |

---

## 2. Data flow

```
                        ┌──────────────────────────────────┐
                        │  STATE SOURCES                   │
                        │  /px4/bridge/vehicle_odometry    │
                        │  /px4/bridge/vehicle_local_pos_v1│
                        └──────────────┬───────────────────┘
                                       │ Float64MultiArray
                                       ▼
                    ┌──────────────────────────────────────┐
   VLA chunk        │  PLANNER                             │   /mppi/plan
   /vla/chunk ────► │  mppi_ros_node.py   (vanilla MPPI)   │ ─────────────┐
   (vla_node only)  │  mppi_vla_node.py   (+ warm-start +  │              │
                    │                      3 cost modes)   │              │
                    └──────────────────────────────────────┘              │
                                                                          ▼
                              JointTrajectory                ┌──────────────────────┐
                                                             │  CONSUMER (one of):  │
                                                             │  - mppi_driver.py    │  → real drone / arm
                                                             │  - mppi_ros_sim_node │  → MuJoCo loopback
                                                             │  - test_driver.py    │  → prints summary
                                                             └──────────────────────┘
```

State sources can be either the real PX4 bridge **or** a sim node running in
loopback (`openloop:=false`).

---

## 3. Topics (live wiring)

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/px4/bridge/vehicle_odometry` | `std_msgs/Float64MultiArray` (25 floats) | PX4 → planner, driver, sim | Position at `[3:5]`, lin-vel at `[10:13]`, yaw-rate at `[15]`. Index conventions are documented in the planner source. |
| `/px4/bridge/vehicle_local_position_v1` | `std_msgs/Float64MultiArray` (16 floats) | PX4 → planner, driver, sim | Heading at `[14]` (minus `pi/2` for the AtmosM3 frame convention). |
| `/mppi/plan` | `trajectory_msgs/JointTrajectory` | planner → driver / sim / test | One point per `sim_dt` over the planning horizon. `joint_names = CONTROL_NAMES[:nu]`. Each `positions[]` is the full 11-D control vector (see §5). `time_from_start` is from-now offsets. `header.stamp` is the timestamp of the state the plan was computed from (for end-to-end latency). |
| `/vla/chunk` | `trajectory_msgs/JointTrajectory` | VLA → `mppi_vla_node.py` | Override with `vla_chunk_topic` param. Same layout as `/mppi/plan`: per-point `positions[]` carries the 11-D control vector; `time_from_start` + `header.stamp` define absolute times. |
| `/TODO/arm/joint_states` | `sensor_msgs/JointState` | arm bridge → planner | **Not yet wired in either planner.** Constant zero is used. Placeholder name. |
| `/TODO/base/odom` | `nav_msgs/Odometry` | PX4 bridge → planner | **Not used** — the PX4 Float64MultiArray topics above are. Kept as a docstring artifact. |

---

## 4. The 11-D control vector

Both `/mppi/plan` and `/vla/chunk` use the same layout in `JointTrajectoryPoint.positions[]`:

| Index | Field | Units | Notes |
|---|---|---|---|
| 0 | `vx_body_cmd` | m/s | Base velocity, body frame |
| 1 | `vy_body_cmd` | m/s | Base velocity, body frame |
| 2 | `wz_body_cmd` | rad/s | Base yaw rate |
| 3–8 | `joint_0..5` | rad | Arm joint setpoints (absolute or delta, see `AtmosM3.arm_mode`) |
| 9 | `gripper_cmd` | 0/1 | Binary: 0 = closed, 1 = open (thresholded at 0.5 inside the task) |
| 10 | `_dead` | — | Always 0. Driven from index 9; kept for ctrl shape alignment. |

Note: physical meaning is mixed (velocity for base, position for arm,
binary for gripper). They all live in `positions[]` because hydrax treats
them uniformly as "controls". `velocities[]` and `effort[]` are not used.

---

## 5. The four cost modes in `mppi_vla_node.py`

Selected by the `cost_mode` ROS param at startup.

### `default`
Stock `MPPI` + stock `AtmosM3` (the circle-tracking cost currently in
[hydrax/tasks/atmos_m3.py:239](../hydrax/tasks/atmos_m3.py#L239)). Identical
to `mppi_ros_node.py` except for the VLA warm-start logic (see §6) which
runs in *all* modes.

### `vla_track`
Uses `MPPI_WithCtx` + `AtmosM3VlaTrack`. Once per VLA chunk arrival, the
node rolls the chunk through `mjx.step` (using the same
`ctrl_transform_with_integral` pipeline MPPI uses internally) starting from
the *current* state. The resulting reference state trajectory becomes the
cost target:

```
cost_t = Σ_i W_base[i] * (state_i - s_ref_i)²
       + Σ_j W_arm[j]  * (arm_j   - arm_ref_j)²
       + ctrl_reg * |u_base|²
```

Yaw error is wrapped to `[-π, π]`. The rollout-once-per-chunk choice
mitigates compounding errors — if the actual state drifts, MPPI's job is
to pull it back to the reference (rather than the reference drifting with
the state).

Params: `vla_track_ref_horizon_s` (default 1.0 — only needs to cover one
chunk period plus the plan horizon; cost clamps past the end of the ref),
`vla_track_base_weights` (default `[10,10,5,1,1,0.1]`), `vla_track_arm_weights`
(default all-zero), `vla_track_ctrl_reg` (default 0.1).

**Async rollout:** the per-chunk `mjx.step` rollout that builds the new
reference runs on a worker thread, so it doesn't block the 20 Hz plan loop.
The very first chunk after startup is rolled out synchronously (so the
first plan sees a valid reference instead of the zero-initialized one);
every subsequent chunk produces a future whose result is swapped into
`self.ctx` on the first plan tick after it completes. One-tick staleness
(~50 ms) is negligible vs the chunk period (~667 ms). See
[mppi_vla_node._maybe_rebuild_vla_ref](mppi_vla_node.py).

### `value_shaped`
Uses `MPPI_WithCtx` + `AtmosM3ValueShaped`. Cost is `-V(s)` per step (V
mode) or `-Q(s,u)` per step (Q mode). Network is a 3-layer MLP defined
in [hydrax/tasks/atmos_m3_value.py](../hydrax/tasks/atmos_m3_value.py).

Per-step (not telescoping) was chosen for denser optimization signal — for
~5000 (N×H) state evaluations per plan tick, an MLP forward is negligible
on the Jetson.

Params: `value_kind` (`"v"` or `"q"`), `value_hidden` (default 256),
`value_out_scale` (default 1.0, tanh-bounded), `value_ckpt_path` (pickled
flax params; empty string = random init = ~zero signal), `value_ctrl_reg`
(default 0.1).

**Feature inputs:** `concat(qpos, qvel)` for V, `concat(qpos, qvel,
control)` for Q. Override `features_v`/`features_q` in a subclass if your
IQL critic was trained on different inputs.

### `vla_track_value`
Blend of the previous two. Uses `MPPI_WithCtx` +
[`AtmosM3VlaTrackValue`](../hydrax/tasks/atmos_m3_vla_track_value.py). Cost
is a weighted sum:

```
cost_t = w_track * tracking_term  +  w_value * (-V(s))  +  ctrl_reg * |u_base|²
```

with `tracking_term` identical to `vla_track`'s and `V` identical to
`value_shaped`'s (Q mode also supported). The ctx threaded into MPPI is the
pair `(vla_ctx, value_params)` — the per-chunk rollout only touches the
first half; `value_params` is loaded once at startup and pinned.

This mode is for the regime where the value net is still being trained and
you want the VLA rollout to keep MPPI honest in case V is noisy or
miscalibrated. Two independent weights mean you can set `w_value=0` to
fall back to pure tracking without restarting (the ckpt is still required
to launch — see below).

**Refuses to start without `value_ckpt_path`.** Deliberate: if you want
pure tracking, use `cost_mode=vla_track` directly so the cost shape
matches what you ran.

Params: `vla_track_value_w_track` (default 1.0), `vla_track_value_w_value`
(default 1.0). Reuses `vla_track_*` (base/arm weights, ctrl_reg,
ref_horizon_s) and `value_*` (kind, hidden, out_scale, ckpt_path).

---

## 6. VLA warm-start (runs in all `cost_mode`s of `mppi_vla_node.py`)

Independent of the cost: this controls the MPPI **action prior**.

1. **Per plan tick (~30 Hz):** after `jit_optimize`, overwrite the last
   `vla_tail_knots` knots of `policy_params.mean` with the VLA chunk
   resampled at the corresponding knot times. Blend strength = `vla_tail_alpha`
   (1.0 = hard pin, 0.0 = off).
2. **On new VLA chunk (~1–2 Hz):** set a "full warmstart" flag. Next tick,
   *before* `jit_optimize`, resample the chunk at all knot times and
   overwrite `policy_params.mean` entirely. Also (in `vla_track` mode)
   rebuild the reference state trajectory by rolling the chunk through
   `mjx.step` from the current state.
3. **Cold start:** node refuses to publish until base state AND a first
   VLA chunk have arrived.
4. **Staleness:** if `chunk_age > vla_max_chunk_age_s` (default 2.0 s),
   tail pinning is skipped and a throttled warning is logged. Tracking
   reference keeps being used (last rollout).

---

## 7. ROS params reference

### `mppi_ros_node.py` / `mppi_vla_node.py` shared
| Param | Default | Meaning |
|---|---|---|
| `plan_rate_hz` | 20.0 | Replanning rate. |
| `num_samples` | 256 | MPPI rollouts per optimize call. |
| `plan_horizon` | 0.25 | Seconds. Spline endpoint. |
| `num_knots` | 8 | Spline knots over the horizon. |
| `temperature` | 0.2 | MPPI softmax temperature. |

### `mppi_vla_node.py` only — warm-start
| Param | Default | Meaning |
|---|---|---|
| `vla_chunk_topic` | `/vla/chunk` | Subscription topic. |
| `vla_tail_knots` | 1 | Number of trailing knots to pin from VLA. |
| `vla_tail_alpha` | 1.0 | Blend: 1.0 = hard-pin, 0.0 = disabled. |
| `vla_full_warmstart_on_new_chunk` | True | Resample all knots on new chunk arrival. |
| `vla_max_chunk_age_s` | 2.0 | After this, skip pinning + warn. |

### `mppi_vla_node.py` only — cost mode
| Param | Default | Meaning |
|---|---|---|
| `cost_mode` | `"default"` | `default` \| `vla_track` \| `value_shaped` \| `vla_track_value` |
| `vla_track_ref_horizon_s` | 1.0 | Length of the reference rollout (seconds). Must exceed (max chunk gap + plan_horizon); cost clamps past the end. |
| `vla_track_base_weights` | `[10,10,5,1,1,0.1]` | Per-dim weight on `(x,y,yaw,vx,vy,wz)`. |
| `vla_track_arm_weights` | `[0]*6` | Per-joint weight on arm tracking. Bump when arm is wired. |
| `vla_track_ctrl_reg` | 0.1 | Weight on `\|u_base\|²`. |
| `value_kind` | `"v"` | `"v"` (cost = `-V(s)`) or `"q"` (cost = `-Q(s,u)`). |
| `value_hidden` | 256 | Hidden width of the value MLP. |
| `value_out_scale` | 1.0 | Multiplier on `tanh` output (sets cost magnitude). |
| `value_ckpt_path` | `""` | Path to pickled flax param pytree. Empty = random init in `value_shaped`; **required** in `vla_track_value`. |
| `value_ctrl_reg` | 0.1 | Weight on `\|u_base\|²` (value_shaped only). |
| `vla_track_value_w_track` | 1.0 | `vla_track_value` only — multiplier on the tracking term. |
| `vla_track_value_w_value` | 1.0 | `vla_track_value` only — multiplier on the `-V(s)` (or `-Q(s,u)`) term. Set to 0 to disable the value side without restarting. |

### `mppi_driver.py`
| Param | Default | Meaning |
|---|---|---|
| `publish_rate_hz` | 50.0 | Command publish rate. |
| `plan_timeout_s` | 0.2 | No new plan within this → safe stop. |
| `state_timeout_s` | 0.2 | No new state within this → safe stop. |
| `max_linear_accel` | 2.0 | Slew limit, m/s². |
| `max_angular_accel` | 4.0 | Slew limit, rad/s². |
| `max_linear_vel` | 0.3 | Saturation, m/s. |
| `max_angular_vel` | 0.4 | Saturation, rad/s. |
| `max_arm_pos` / `min_arm_pos` | ±3.14 | Per-joint clamp, rad. |

### `mppi_ros_sim_node.py`
| Param | Default | Meaning |
|---|---|---|
| `state_pub_rate_hz` | 50.0 | Loopback state rate. |
| `video_fps` | 30.0 | Recording fps (post-hoc render, not live). |
| `video_width` / `video_height` | 720 / 480 | Frame size. |
| `video_dir` | `/workspace/hydrax/mppi_videos/` | Output dir. |
| `record_video` | True | Disable to save Jetson time. |
| `realtime` | True | Sleep to wallclock; if False, run as fast as possible. |
| `camera_id` | -1 | -1 = follow-robot virtual cam. |
| `cam_distance` / `cam_azimuth` / `cam_elevation` / `cam_lookat_z` | 4.0 / 90.0 / -20.0 / 0.5 | Virtual cam pose. |
| `duration_sec` | 60.0 | ≤0 = run forever. |
| `openloop` | False | True = consume plans but don't publish state (closed-loop on real robot). |

---

## 8. Testing without a real VLA or value network

Three levels of test you can run *right now* — no robot, no trained VLA,
no IQL critic.

### 8.1 Wiring smoke test (open-loop state, fake VLA chunks)
Verifies: ROS plumbing, JIT compile, plan rate, all four cost modes
construct cleanly.

[mock_bridge.py](mock_bridge.py) publishes fake state + fake VLA chunks
and subscribes to `/mppi/plan` to print one-line summaries. State does
**not** react to commands — this only tells you the pipeline is wired,
not that the controls are good.

```
# Terminal 1
python3 mppi_vla_node.py --ros-args \
  -p cost_mode:=vla_track \
  -p vla_tail_knots:=2 -p vla_tail_alpha:=1.0

# Terminal 2
python3 mock_bridge.py --ros-args \
  -p state_mode:=static \
  -p vla_mode:=constant_forward
```

Expected output: `plan #N @ ~19.5Hz n_pts=25 first[base]=... last[base]=[+0.100 +0.000 +0.000]`.
The `last[base]` triple should sit on the pinned VLA value
(`vx=0.1, vy=0, wz=0`). `first[base]` is whatever MPPI optimizes from
the current state.

You can flip the cost mode without changing anything else:
- `cost_mode=default` → no tracking; `last[base]` still pinned by tail.
- `cost_mode=value_shaped` (no ckpt) → random-init MLP gives `-V ≈ 0`,
  so `ctrl_reg` dominates; both `first[base]` and `last[base]` should
  sit near zero except for the pinned tail. Verifies plumbing only.
- `cost_mode=vla_track_value` → see §8.2 (needs a dummy ckpt).

`vla_mode` options in `mock_bridge.py`: `zero`, `constant_forward`,
`step`, `sinusoid`, `yaw_spin`. [vla_chunk_publisher.py](vla_chunk_publisher.py)
has a superset including `circle_track` (use it with `sim_node`, not
`mock_bridge`).

### 8.2 `vla_track_value` smoke test with a dummy value ckpt
`vla_track_value` refuses to start without `value_ckpt_path`. Generate a
random-init pickle that matches the node's MLP shape:

```
python3 -c "
import pickle
from hydrax.tasks.atmos_m3_vla_track_value import AtmosM3VlaTrackValue
task = AtmosM3VlaTrackValue(value_kind='v', hidden=256, out_scale=1.0)
with open('/tmp/dummy_value.pkl', 'wb') as f:
    pickle.dump(task.init_value_params(seed=0), f)
print('wrote /tmp/dummy_value.pkl')
"
```

Then:
```
python3 mppi_vla_node.py --ros-args \
  -p cost_mode:=vla_track_value \
  -p value_ckpt_path:=/tmp/dummy_value.pkl \
  -p vla_track_value_w_track:=1.0 \
  -p vla_track_value_w_value:=0.0     # value side off → behavior = pure tracking
```

With `w_value=0`, output should match plain `vla_track`. Flip it to
`0.1` or `1.0` to confirm the value term doesn't NaN or wreck planning
(it'll just contribute ~zero cost because the MLP is random; behavior
stays tracking-dominated). When your real ckpt is ready, swap the path
and weights — no other changes needed.

### 8.3 Closed-loop sim (MuJoCo loopback)
The strongest test you can run without hardware: see
[README_VLA_SIM_TEST.md](README_VLA_SIM_TEST.md). State *does* react to
commands, so this is the right place to validate control quality, not
just plumbing.

---

## 9. Common deployment combos

### A. Planner-in-loop on the real drone
```
Terminal 1:  python3 mppi_ros_node.py        # or mppi_vla_node.py
Terminal 2:  python3 mppi_driver.py
Terminal 3:  (PX4 bridge publishing /px4/bridge/*)
```

### B. Hardware-in-the-loop sim (planner + MuJoCo closed-loop, no robot)
```
Terminal 1:  python3 mppi_ros_node.py
Terminal 2:  MUJOCO_GL=egl python3 mppi_ros_sim_node.py
```
Sim publishes loopback state on the same `/px4/bridge/*` topics the planner
already subscribes to. Video lands in `video_dir`.

### C. Sanity-check the planner without any sim
```
Terminal 1:  python3 mppi_ros_node.py
Terminal 2:  python3 mppi_ros_test_driver.py
```
Note: `mppi_ros_test_driver.py` currently publishes on the legacy
`BASE_STATE_TOPIC`/`ARM_STATE_TOPIC` (the `/TODO/*` placeholder names),
not the live `/px4/bridge/*` topics the node now reads. This means it
will NOT drive `mppi_ros_node.py` as currently wired — to actually feed
the planner you need a `/px4/bridge/*` mock. See §9 for the planned
extension.

### D. VLA in the loop (real or sim)
```
Terminal 1:  python3 mppi_vla_node.py --ros-args -p cost_mode:=vla_track
Terminal 2:  (VLA model publishing /vla/chunk at ~1-2 Hz)
Terminal 3:  python3 mppi_driver.py     # or mppi_ros_sim_node.py
Terminal 4:  (PX4 bridge or nothing if using sim loopback)
```

### D'. VLA + learned value, on the real drone
```
Terminal 1:  python3 mppi_vla_node.py --ros-args \
               -p cost_mode:=vla_track_value \
               -p value_ckpt_path:=/path/to/iql_critic.pkl \
               -p vla_track_value_w_track:=1.0 \
               -p vla_track_value_w_value:=0.1     # start small while V matures
Terminal 2:  (VLA model publishing /vla/chunk)
Terminal 3:  python3 mppi_driver.py
Terminal 4:  (PX4 bridge)
```
Raise `vla_track_value_w_value` as you gain confidence in the value net.

### E. Real-robot openloop sim alongside (for visualization parity)
```
Terminal 1:  python3 mppi_ros_node.py
Terminal 2:  python3 mppi_driver.py
Terminal 3:  MUJOCO_GL=egl python3 mppi_ros_sim_node.py --ros-args -p openloop:=true
Terminal 4:  (PX4 bridge)
```
The sim only consumes plans + records video; the real bridge owns state.

---

## 9. Known TODOs

- **Arm not wired.** Both planners zero out arm `qpos`/`qvel` in
  `_assemble_state`. `ARM_STATE_TOPIC` is the `/TODO/arm/joint_states`
  placeholder. To wire: implement `_arm_cb` (commented out in source),
  set `ARM_JOINT_NAMES` to real names, and bump
  `vla_track_arm_weights` if using `vla_track` mode.
- **Test driver lags the real wiring.** `mppi_ros_test_driver.py`
  publishes `Odometry`/`JointState` on the placeholder topics, but the
  planner reads `/px4/bridge/vehicle_odometry` +
  `/px4/bridge/vehicle_local_position_v1` (Float64MultiArray). A
  PX4-mock test driver + a dummy VLA chunk publisher are the obvious
  next things to add (see §10).
- **No `JointState` arm cb in `mppi_driver.py`.** Same as planner.
- **IQL value net is a stub.** `AtmosM3ValueShaped` uses a randomly-initialized
  MLP until you train + drop in a `value_ckpt_path`.
- **`use_sim_time`.** Nothing in this directory sets `use_sim_time`; clock
  conversions in `mppi_vla_node._vla_cb` assume the VLA publisher uses the
  same clock as `rclpy.get_clock().now()`.

---

## 10. Docker

```bash
# Build (run once)
docker build -t hydrax-thor:base hydrax_on_thor/docker_stuff

# Launch (mounts repo at /workspace/hydrax, exposes host network for ROS)
bash hydrax_on_thor/docker_stuff/run_container.sh
```

The image is based on `ghcr.io/nvidia/jax:jax-2025-12-17` and installs
mujoco-mjx, flax, evosax, interpax, and ROS 2 Jazzy. JAX compilation cache
lives at `/workspace/.jax_cache` inside the container — first cold JIT is
~30 s (longer for `vla_track`/`value_shaped` modes), subsequent runs reuse
the cache and warm in ~1 s.
