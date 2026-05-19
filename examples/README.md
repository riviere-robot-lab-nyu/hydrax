# MPPI ROS Pipeline (AtmosM3)

ROS 2 nodes that run the AtmosM3 MPPI controller in three configurations:

1. **Sim loop** — planner + headless MuJoCo sim that closes the loop and records video.
2. **Hardware deployment** — planner + trajectory follower that talks to the real drone.
3. **Latency test** — planner + dummy state driver, for end-to-end timing.

All nodes live in this directory:

| File | Role |
|---|---|
| `mppi_ros_node.py` | MPPI planner. Subscribes to state, publishes control trajectories (`JointTrajectory`). |
| `mppi_ros_sim_node.py` | Headless MuJoCo simulator + video recorder. Subscribes to plans, publishes state back. |
| `trajectory_follower.py` | High-rate hardware-side follower with watchdogs + safety layer. |
| `mppi_ros_test_driver.py` | Dummy state publisher for latency benchmarking. |

---

## Architecture

### Sim setup

```
  ┌─────────────────┐   plan    ┌─────────────────────┐
  │ mppi_ros_node   │ ────────▶ │ mppi_ros_sim_node   │
  │  (planner)      │           │ (mujoco + video)    │
  │                 │ ◀──────── │                     │
  └─────────────────┘   state   └─────────────────────┘
       ~30 Hz                          100 Hz physics
```

### Hardware setup

```
  ┌──────────────┐  state    ┌────────────┐  plan    ┌────────────────────┐  vel/arm cmd   ┌─────────┐
  │ drone driver │ ────────▶ │ mppi plan- │ ───────▶ │ trajectory_        │ ─────────────▶ │  drone  │
  │  (vendor)    │           │ ner node   │          │ follower           │                │ (vendor)│
  └──────────────┘           └────────────┘          └────────────────────┘                └─────────┘
         ▲                                                                                       │
         └──────────────────────────────── state ────────────────────────────────────────────────┘
       ~50 Hz                       ~30 Hz                   100-200 Hz
```

The drone driver publishes state to **both** the planner and the follower. The follower does not feed state back to the planner — that edge goes directly drone → planner.

---

## Running the sim setup

Two terminals. Order doesn't matter; topics use BEST_EFFORT QoS so nodes discover each other.

```bash
# terminal 1 — planner
python3 mppi_ros_node.py

# terminal 2 — sim + recorder
MUJOCO_GL=egl python3 mppi_ros_sim_node.py
```

The sim runs for `duration_sec` (default 60 s), then shuts itself down and renders the recorded run to mp4. Output goes to `/workspace/hydrax/mppi_videos/simulation_<timestamp>.mp4` (override with `-p video_dir:=...`).

### Common overrides

```bash
# tweak camera (free tracking cam follows the base by default)
python3 mppi_ros_sim_node.py --ros-args \
    -p video_dir:=/workspace/mppi_videos \
    -p cam_distance:=6.0 \
    -p cam_elevation:=-30.0

# run until Ctrl-C instead of fixed duration
python3 mppi_ros_sim_node.py --ros-args -p duration_sec:=-1.0

# different planner params (faster JIT, less compute per tick)
python3 mppi_ros_node.py --ros-args \
    -p num_samples:=512 \
    -p plan_horizon:=0.3 \
    -p num_knots:=5
```

### Headless rendering

The sim records video offscreen (faster + leaves more CPU/GPU for physics). Set `MUJOCO_GL=egl` (or `osmesa`) before launching so MuJoCo's renderer can init without a display. The `ffmpeg` binary must be on `PATH`:

```bash
# inside the container, if ffmpeg is missing:
apt-get update && apt-get install -y ffmpeg
# or:
pip install imageio-ffmpeg && \
  export PATH="$(python -c 'import imageio_ffmpeg, os; print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))'):$PATH"
```

---

## Running on hardware

Three terminals (or three systemd units, etc.). Topic names in all four `.py` files are placeholders (`/TODO/...`); replace them with your real wiring before flying.

```bash
# terminal 1 — drone driver (vendor-provided, not in this repo)
# publishes:  Odometry on BASE_STATE_TOPIC
#             JointState on ARM_STATE_TOPIC
# subscribes: Twist on BASE_CMD_TOPIC
#             JointTrajectory on ARM_CMD_TOPIC
#             Float32 on GRIPPER_CMD_TOPIC

# terminal 2 — planner
python3 mppi_ros_node.py

# terminal 3 — follower
python3 trajectory_follower.py
```

---

## Node reference

### `mppi_ros_node.py` (planner)

Runs MPPI on the AtmosM3 MuJoCo model. JIT-compiles `optimize` and `interp_func` at startup; the on-disk XLA cache at `/workspace/.jax_cache/` makes subsequent starts ~1 s instead of 8-30 s.

**Subscribes**
- `BASE_STATE_TOPIC` (`nav_msgs/Odometry`) — base pose and twist.
- `ARM_STATE_TOPIC` (`sensor_msgs/JointState`) — arm joint positions and velocities.

**Publishes**
- `PLAN_TOPIC` (`trajectory_msgs/JointTrajectory`) — `(i+1)·sim_dt` parameterized control sequence. `header.stamp` = the Odometry stamp the plan was computed from (used by the follower/sim for latency-correct indexing).

**Parameters**

| param | default | notes |
|---|---|---|
| `plan_rate_hz` | 30 | timer rate; actual rate may be lower if compute exceeds period |
| `num_samples` | 1024 | rollout count; biggest knob on compute time |
| `plan_horizon` | 0.3 | seconds; `ctrl_steps = plan_horizon / 0.01` |
| `num_knots` | 5 | spline parameterization; cheap to change |
| `temperature` | 0.2 | low → near-argmax over samples, high → averaged |

### `mppi_ros_sim_node.py` (sim)

Owns a MuJoCo `MjData` in a dedicated thread that ticks at `mj_model.opt.timestep` paced to wallclock. Applies plans via `ctrl_transform_with_integral` + `update_integral` (matches `deterministic_headless.py`). Snapshots `qpos`/`qvel` at `video_fps` during the run, then renders to mp4 after the sim stops (rendering live would steal 10-30 ms per frame from physics).

**Subscribes**: `PLAN_TOPIC`.
**Publishes**: `BASE_STATE_TOPIC`, `ARM_STATE_TOPIC`.

**Parameters**

| param | default | notes |
|---|---|---|
| `duration_sec` | 60 | auto-shutdown after this; ≤0 = run until Ctrl-C |
| `state_pub_rate_hz` | 50 | rate at which Odom + JointState are published |
| `video_fps` | 30 | snapshot cadence and output framerate |
| `video_width`/`height` | 720/480 | upscaled if larger than model's offscreen framebuffer |
| `video_dir` | `/workspace/hydrax/mppi_videos/` | output directory; **bind-mount this if you're in a container** |
| `record_video` | true | set false to disable video entirely |
| `realtime` | true | pace sim to wallclock; false = run as fast as possible |
| `camera_id` | -1 | XML camera id; -1 = free tracking camera follows base |
| `cam_distance` / `azimuth` / `elevation` / `lookat_z` | 4.0 / 90 / -20 / 0.5 | tracking-camera offsets |

### `trajectory_follower.py` (hardware follower)

Subscribes to plans, indexes by `now - plan.header.stamp`, and publishes the resulting commands at high rate to the drone's vendor controllers. **This is the safety boundary** — every output is hard-clamped and slew-limited, and a watchdog forces safe-stop if either the plan or the state goes stale.

**Subscribes**: `PLAN_TOPIC`, `BASE_STATE_TOPIC`, `ARM_STATE_TOPIC` (state only used for watchdog).
**Publishes**:
- `BASE_CMD_TOPIC` (`geometry_msgs/Twist`)
- `ARM_CMD_TOPIC` (`trajectory_msgs/JointTrajectory`, single point)
- `GRIPPER_CMD_TOPIC` (`std_msgs/Float32`)

**Safe-stop semantics**: zero base velocity (slew-limited from current), **hold** arm at last commanded position, **hold** gripper at last commanded state. Deliberately does *not* command the arm to zero.

**Parameters**

| param | default | notes |
|---|---|---|
| `publish_rate_hz` | 100 | tick rate (runs in its own thread, not on the ROS executor) |
| `plan_timeout_s` | 0.2 | safe-stop if no new plan within this window |
| `state_timeout_s` | 0.2 | same for Odom + JointState |
| `max_linear_accel` | 2.0 m/s² | slew limit on base linear vel |
| `max_angular_accel` | 4.0 rad/s² | slew limit on base angular vel |
| `max_linear_vel` | 1.5 m/s | hard cap on base linear vel |
| `max_angular_vel` | 1.5 rad/s | hard cap on base angular vel |
| `max_arm_pos` / `min_arm_pos` | ±π | per-joint clamp (placeholder; set real values per joint) |

**Before flying — checklist**

1. Replace every `/TODO/...` topic and `TODO_joint_N` joint name with real wiring (in **all** node files, since `mppi_ros_sim_node.py` and `mppi_ros_test_driver.py` import from `mppi_ros_node`).
2. If your drone takes a vendor-specific message instead of `geometry_msgs/Twist`, swap the base publisher.
3. If your arm controller takes `sensor_msgs/JointState` or a vendor message instead of `JointTrajectory`, swap the arm publisher.
4. Set realistic `max_linear_vel` / `max_angular_vel` / accel limits for your platform.
5. Set proper per-joint arm limits — the ±π placeholder is too permissive for most joints.

### `mppi_ros_test_driver.py` (latency benchmark)

Publishes fake Odometry + JointState at 50 Hz (base traces a 0.5 m circle, arm static) and listens for plans, reporting `now - plan.header.stamp` end-to-end latency. Useful for tuning planner config without spinning up the full sim.

```bash
python3 mppi_ros_node.py
python3 mppi_ros_test_driver.py
```

---

## Design notes

### Timing — why the planner stamps with the state's timestamp

The planner copies `last_base_stamp` (the Odometry stamp) into `plan.header.stamp` rather than its own `now()`. The follower/sim then computes `elapsed = now - plan.header.stamp` and indexes the plan by that. Two consequences:

- **Latency-correct indexing**: the plan slice the follower applies is the one the planner intended for "now," accounting for the round-trip sim→planner→sim latency (~30 ms typical).
- **No clock sync needed across machines**: the stamp originates from one machine and returns to the same one. The math never crosses clocks. NTP/PTP doesn't matter for this loop.

### Why the sim runs physics in a thread, not a ROS timer

ROS timers fire on the executor, which serializes with subscriber callbacks under a single-threaded executor (and contends for the GIL under a multi-threaded one). A dedicated Python thread that calls `time.monotonic()` and `time.sleep()` is more deterministic for `mj_step` pacing on the Jetson. (We tried `MultiThreadedExecutor` early on — plan rate dropped 30× due to GIL contention. Reverted.)

### Why video rendering is offline, not live

On the Jetson, `mujoco.Renderer.render()` is 10-30 ms per frame. Doing it live at 30 fps would steal a third of the sim thread's time and starve the plan subscriber. We snapshot `qpos`/`qvel` during the run (~184 B per snapshot — 60 s at 30 fps = ~330 KB) and re-render after sim stops via `mj_forward` + `update_scene` + `render`.

### Why BEST_EFFORT KEEP_LAST(1) QoS on state subscribers

Default `RELIABLE` with `depth=10` lets up to 200 ms of stale Odometry queue up when the planner gets behind, which inflates apparent latency by exactly that amount. `BEST_EFFORT` + `KEEP_LAST(1)` bounds staleness to one message (~20 ms at 50 Hz publisher) and silently drops anything older.

### Persistent JAX compile cache

`mppi_ros_node.py` enables `jax_compilation_cache_dir = "/workspace/.jax_cache"`. First run on a fresh machine pays the full ~8-30 s XLA compile; subsequent runs (same JAX/jaxlib version, same source-hash) read from disk and `JIT done` collapses to ~1 s. Bind-mount this directory if you want cache persistence across container rebuilds.

---

## Troubleshooting

**`plans=0` for many seconds at startup.** Almost always the planner's initial JIT. Check the planner terminal — `JIT done in X.XXs` shows when it's actually ready. If your cache dir is empty after this, you'll pay it again next run; if it's populated, the next run will be sub-second.

**Sim rtr drops below 1.0x.** Per-step compute exceeded `mj_dt` (default 10 ms). Either reduce `mj_model.opt.timestep` work (simpler model), shrink planner config so it doesn't share GPU time with sim, or accept slower-than-realtime sim.

**Video says "Skipping video — renderer init failed".** Either `MUJOCO_GL` isn't set, the offscreen framebuffer is too small (we auto-expand it now, but check), or `ffmpeg` isn't on `PATH`.

**Follower's `safe_stops` counter is non-zero.** Plan or state went stale within `*_timeout_s`. Check the planner terminal and the drone driver's state publish rate.

**Video saved to `/tmp/...` and you can't find it.** `/tmp` is container-local. Use `-p video_dir:=/workspace/...` (or whatever bind mount you have) to write to the host.
