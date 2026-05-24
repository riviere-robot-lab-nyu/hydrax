# VLA + MuJoCo sim closed-loop test

End-to-end eyeball test for [mppi_vla_node.py](mppi_vla_node.py): VLA chunks
in, optimized plans out, closed loop through a headless MuJoCo simulator.
No real robot or trained VLA model required.

## What this exercises (and what the simpler `mock_bridge` test doesn't)

`mock_bridge.py` publishes fake static/circle/drift state — the state does
**not react to MPPI's commands**, so you can verify wiring but not control
quality. This setup is different: [mppi_ros_sim_node.py](mppi_ros_sim_node.py)
steps mjx forward in response to each plan and republishes the resulting
state on `/px4/bridge/*`, closing the loop. The planner sees its own
controls' effect on simulated dynamics.

This is the right test for:

- Whether **`cost_mode=vla_track`** actually drives the simulated state
  toward the rolled-out VLA reference (mock_bridge can only show that the
  cost shape *is being computed* — sim shows whether MPPI *follows* it).
- Whether the **warm-start tail pinning** produces controls that, when
  actually executed, behave like the VLA prior.
- Whether **plan latency / publish rate / chunk staleness** behave the way
  the tunings predict, over a multi-second rollout.
- Visual sanity check via the recorded mp4 in `video_dir`.

## Process topology

```
       /vla/chunk                            /mppi/plan
  ┌─────────────────┐ ────────► ┌──────────┐ ─────────► ┌──────────────┐
  │ vla_chunk_pub   │           │  planner │            │ sim_node     │
  └─────────────────┘           │ (vla)    │            │ (mjx, non-   │
                                │          │ ◄──────────│  openloop)   │
                                └──────────┘            └──────────────┘
                                  ▲                          │
                                  │ /px4/bridge/vehicle_odom │
                                  │ /px4/bridge/local_pos_v1 │
                                  └──────────────────────────┘

  vla_chunk_publisher.py    mppi_vla_node.py        mppi_ros_sim_node.py
  (publishes /vla/chunk)    (publishes /mppi/plan)  (publishes state +
                                                     records video)
```

Three processes, one topic graph. Critically: **only `sim_node` publishes
state** — `vla_chunk_publisher.py` deliberately does *not* publish on
`/px4/bridge/*` (this is why it exists separately from `mock_bridge.py`,
which does).

## Prerequisites

All of these run inside the docker image from [docker_stuff/](docker_stuff/).

- `MUJOCO_GL=egl` (or `osmesa`) in the sim terminal — sim_node renders the
  mp4 at shutdown.
- Video directory exists: default is `/workspace/hydrax/mppi_videos/` (set
  by `video_dir` ROS param). Should exist already in the docker; create it
  if not.

## Recipe

Three terminals (or three tmux panes). **Start the sim FIRST** — it logs the
sim model's initial state to the bridge topics, which is what the planner
needs to come out of "Waiting for state..." gating.

### Terminal 1 — sim
```
MUJOCO_GL=egl python3 mppi_ros_sim_node.py --ros-args \
  -p record_video:=true \
  -p video_fps:=30.0 \
  -p duration_sec:=30.0
```
Tweaks worth knowing:
- `-p record_video:=false` skips the mp4 render at shutdown (faster
  iteration; you lose the visualization).
- `-p duration_sec:=0` runs until Ctrl-C.
- `-p realtime:=false` runs the sim as fast as physics allows. Useful for
  short-duration debugging, but the plan/state rates desync from wallclock,
  which can hide latency-related issues.
- `-p openloop:=true` is the **wrong setting** for this test — it disables
  state publishing, and the planner would never get state. Leave it false.

### Terminal 2 — planner (vla_track mode)
```
python3 mppi_vla_node.py --ros-args \
  -p cost_mode:=vla_track \
  -p vla_tail_knots:=1 -p vla_tail_alpha:=1.0 \
  -p vla_track_ref_horizon_s:=2.0
```
Substitute `cost_mode:=default` or `cost_mode:=value_shaped` for the other
modes. First JIT compile is ~30–60 s; subsequent runs reuse
`/workspace/.jax_cache` and warm in ~1–2 s.

### Terminal 3 — VLA chunk publisher
```
python3 vla_chunk_publisher.py --ros-args \
  -p vla_mode:=constant_forward \
  -p vla_pub_rate_hz:=1.5
```
This terminal also subscribes to `/mppi/plan` and prints a one-line
summary (the `first[base]` / `last[base]` triples) per plan, so you can
watch the planner respond as you change `vla_mode` or restart with
different params.

## What to look for, per mode

### `cost_mode=default` + `vla_mode=constant_forward (vx=0.1)`
The default cost is circle-tracking (see
[hydrax/tasks/atmos_m3.py:249](../hydrax/tasks/atmos_m3.py#L249)). The VLA
only influences the **action prior** via tail pinning. So:

- `last[base]` printout: should converge to `≈ [+0.100, 0.000, 0.000]`
  (the pinned tail knot).
- `first[base]`: whatever MPPI optimizes for circle-following from the
  current sim state.
- Recorded mp4: the robot should make some attempt to follow the circle
  (`_circle_cost` uses radius 5 m, period 45 s) — note that 30 s of
  simulation will only cover ~2/3 of a circle.

### `cost_mode=vla_track` + `vla_mode=constant_forward`
The chunk gets rolled through mjx once per arrival (~1.5 Hz), producing a
reference state trajectory of forward motion. MPPI tries to drive the sim
state along that reference.

- `first[base]` printout: should sit near `[+0.10, 0.00, 0.00]` while the
  planner pushes toward the forward-motion reference.
- Recorded mp4: robot should accelerate forward and hold ~0.1 m/s. Body
  velocity converges to vx ≈ 0.1, with small base-control oscillations as
  MPPI corrects against the integrator's lag.
- If `last[base]` drifts away from `[+0.10, ...]`, that's not a bug — the
  tail is being pinned, not the head, and `vla_track` mode is doing
  state-tracking rather than control-echoing.

### `cost_mode=vla_track` + `vla_mode=circle_track`
Tells the VLA prior to drive a body-frame circle (default `r=1.0 m`,
`ω=0.3 rad/s`). MPPI should produce coordinated `vx` + `wz` commands.

- Recorded mp4: robot should trace approximately a circle. Compare its
  radius to `vla_circle_r`. Loose match is expected (MPPI's circle cost
  isn't on; only the VLA-rollout reference is).
- Watch for **drift between updates**: at `vla_pub_rate_hz=1.5`, the
  reference is re-anchored every ~0.67 s. Between re-anchors, the sim
  state can drift; on re-anchor, you should see a small but visible
  correction in `first[base]`.

### `cost_mode=vla_track` + `vla_mode=step`
The chunk has `vx=0` for the first half and `vx=0.3` for the second half.

- Recorded mp4: robot should sit still, then *accelerate* into forward
  motion at the rough timing of "halfway through each chunk's horizon".
  With a 2 s chunk and 1.5 Hz pub rate, you'll see this rhythm overlap
  with itself — interesting but a little muddled.
- Try `vla_pub_rate_hz:=0.5` and `vla_chunk_horizon_s:=4.0` for a cleaner
  picture: full step-response per chunk before the next one arrives.

### `cost_mode=value_shaped` (V mode, untrained MLP)
The value net is random-init, so `−V(s) ≈ 0` for any state.

- `first[base]` and `last[base]`: should sit near `[0, 0, 0]` (control reg
  dominates; "do nothing" minimizes `ctrl_reg * |u|²`).
- Recorded mp4: robot should be nearly stationary.
- This run only verifies the plumbing doesn't crash. For meaningful
  behavior, supply `-p value_ckpt_path:=/path/to/iql_critic.pkl` and tune
  `value_out_scale` to set the value-cost magnitude.

## Reading the recorded video

Output lands in `video_dir` (default `/workspace/hydrax/mppi_videos/`).
The sim's camera follows the robot (`camera_id:=-1`, the default).
Filename format: see [mppi_ros_sim_node.py](mppi_ros_sim_node.py) (a
timestamped `simulation_YYYYMMDD_HHMMSS.mp4`).

If render fails: `MUJOCO_GL` not set, or no EGL libs. Try
`MUJOCO_GL=osmesa` instead, or set `record_video:=false` and just rely on
the plan-listener printouts.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Planner stuck on `Waiting for state... odom=False pos=False vla=...` | Sim didn't start, or sim was started with `openloop:=true`. Check sim terminal for errors. |
| Planner stuck on `... vla=False` | `vla_chunk_publisher.py` not running, or `vla_topic` mismatch. Default is `/vla/chunk` on both sides; verify with `ros2 topic list`. |
| Two state publishers, planner sees stale or flipping state | You're running `mock_bridge.py` AND `sim_node`. Don't — use `vla_chunk_publisher.py` (this file) when the sim is the state source. |
| `plan #N @ 0.5Hz n_pts=30 ...` (plan rate way below the 30 Hz target) | Sim is starving the planner of CPU. Bump `-p num_samples:=512` (lower) or run `sim_node` with `record_video:=false`. |
| `VLA chunk is stale (X.XXs > 2.00s); skipping tail pinning.` | `vla_pub_rate_hz` is too low or VLA publisher died. Defaults of 1.5 Hz pub × 2.0 s max-age leave only ~0.65 s of slack — bump pub rate or raise `vla_max_chunk_age_s`. |
| mp4 not produced at shutdown | `MUJOCO_GL` not set, render lib missing, or `video_dir` doesn't exist / not writable. `record_video:=false` to bypass. |
| First plan-listener line is `plan #1 @ nanHz ...` | Expected — first message has no previous timestamp to diff against. |

## Quick variations matrix

| What you want to test | Cost mode | vla_mode | Other |
|---|---|---|---|
| Tail pinning only | `default` | `constant_forward` | — |
| Reference state tracking | `vla_track` | `constant_forward` | — |
| New-chunk full warmstart firing | `vla_track` | `step` | `vla_pub_rate_hz:=0.5 vla_chunk_horizon_s:=4.0` |
| Coordinated turn + drive | `vla_track` | `circle_track` | `vla_circle_r:=0.5 vla_circle_w:=0.4` |
| Value-net plumbing smoke test | `value_shaped` | anything | `value_kind:=v` |
| Verify warm-start survives stale chunk | any | `constant_forward` | Kill the VLA publisher mid-run; watch for "stale" warnings + tail unpinning. |

## What this still doesn't test

- **Real PX4 clock skew.** Both sim_node and vla_chunk_publisher stamp
  with `rclpy.get_clock().now()` — no clock mismatch can happen.
- **Real arm dynamics / wiring.** Arm is zeroed throughout this directory.
  Once the arm is wired, bump `vla_track_arm_weights` to non-zero to start
  exercising it.
- **Plan latency under network jitter.** Everything is on localhost.
