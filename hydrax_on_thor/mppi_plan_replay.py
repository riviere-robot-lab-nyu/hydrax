"""Throwaway: grab one (or N) /mppi/plan messages + a snapshot of the real
robot state, roll the plan forward in MuJoCo open-loop with the SAME
ctrl_transform the planner uses, render to mp4, AND report any non-finite
values that appear in inputs, ctrls, or state during the rollout.

This is the MPPI-plan analogue of vla_chunk_replay.py. Key differences:
  - subscribes to /mppi/plan (JointTrajectory), not /vla/action_chunk
  - plan points are already in MPPI ctrl layout (no VLA->MPPI reorder)
  - plan step cadence == sim_dt (set by the planner via time_from_start)
  - every published plan and every rollout step is NaN-scanned, with the
    first offending row / step / channel printed loudly

Initialization (matches mppi_vla_node._assemble_state):
  /global_pose                   -> qpos[0:3]
  /px4/bridge/vehicle_odometry   -> qvel[0:3]
  /follower/joint_state          -> qpos/qvel[3:11]

Usage:
    MUJOCO_GL=egl python3 hydrax_on_thor/mppi_plan_replay.py

Run it while the planner + px4 bridge + follower joint-state publisher are
all up. The script subscribes, waits until pose/odom/arm AND num_plans plans
have arrived, then exits as soon as the rollout is rendered.
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from hydrax.tasks.atmos_m3 import AtmosM3
from hydrax.utils.video import VideoRecorder

ARM_STATE_LEN = 7              # 6 arm + 1 gripper
ODOM_LIN_VEL_START = 10
BASE_VEL_YAW_RATE_INDEX = 15

L = 0.16665


def _scan_nan(arr: np.ndarray, label: str) -> bool:
    """Print first NaN/Inf occurrence in arr. Returns True if any found."""
    mask = ~np.isfinite(arr)
    if not mask.any():
        return False
    if arr.ndim == 1:
        bad = np.where(mask)[0]
        print(f"  !! {label}: non-finite at indices {bad.tolist()} -> {arr}")
    else:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        print(f"  !! {label}: non-finite at rows={rows.tolist()} cols={cols.tolist()}")
        first = rows[0]
        print(f"     first bad row {first}: {arr[first]}")
    return True


def grab_inputs(
    plan_topic: str,
    pose_topic: str,
    odom_topic: str,
    arm_topic: str,
    timeout_s: float,
    num_plans: int,
    min_plan_gap_s: float,
):
    """Spin until one pose/odom/arm message AND `num_plans` distinct plans
    are in hand. Distinct = at least `min_plan_gap_s` since the last accepted
    plan (the planner runs at 20 Hz, so without spacing we'd grab nearly
    identical consecutive plans).
    """
    rclpy.init()
    node = Node("mppi_plan_grabber")
    got = {"pose": None, "odom": None, "arm": None}
    plans: list[np.ndarray] = []
    plan_stamps: list[float] = []
    last_plan_t = [0.0]

    def plan_cb(msg: JointTrajectory):
        if len(plans) >= num_plans:
            return
        now = time.monotonic()
        if plans and (now - last_plan_t[0]) < min_plan_gap_s:
            return
        n = len(msg.points)
        if n == 0:
            print("Empty plan (no points); ignoring")
            return
        nu = len(msg.points[0].positions)
        if nu == 0:
            print("Plan point has empty positions; ignoring")
            return
        arr = np.empty((n, nu), dtype=np.float64)
        for i, pt in enumerate(msg.points):
            if len(pt.positions) != nu:
                print(f"Plan point {i} width mismatch ({len(pt.positions)} vs {nu})")
                return
            arr[i, :] = pt.positions
        # Loud NaN scan but ACCEPT anyway — capturing NaN plans is the point.
        if _scan_nan(arr, f"plan #{len(plans)+1} as published"):
            print(f"     header.stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                  f"frame_id={msg.header.frame_id}")
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        plans.append(arr)
        plan_stamps.append(stamp_s)
        last_plan_t[0] = now
        print(f"Captured plan {len(plans)}/{num_plans}  shape={arr.shape}  "
              f"frame_id={msg.header.frame_id}")

    def pose_cb(msg: PoseStamped):
        if got["pose"] is None:
            got["pose"] = msg

    def odom_cb(msg: Float64MultiArray):
        if got["odom"] is None:
            got["odom"] = msg

    def arm_cb(msg: Float64MultiArray):
        # Take only valid-length samples — the planner rejects shorter, we
        # should too or our seed state will be junk.
        if got["arm"] is None and len(msg.data) >= 2 * ARM_STATE_LEN:
            got["arm"] = msg

    node.create_subscription(JointTrajectory, plan_topic, plan_cb, qos_profile_sensor_data)
    node.create_subscription(PoseStamped, pose_topic, pose_cb, qos_profile_sensor_data)
    node.create_subscription(Float64MultiArray, odom_topic, odom_cb, qos_profile_sensor_data)
    node.create_subscription(Float64MultiArray, arm_topic, arm_cb, qos_profile_sensor_data)

    deadline = time.monotonic() + timeout_s
    last_log = 0.0

    def done():
        return all(v is not None for v in got.values()) and len(plans) >= num_plans

    while not done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - last_log > 1.0:
            missing = [k for k, v in got.items() if v is None]
            if len(plans) < num_plans:
                missing.append(f"plan {len(plans) + 1}/{num_plans}")
            print(f"Waiting for: {missing}")
            last_log = time.monotonic()

    node.destroy_node()
    rclpy.shutdown()
    missing = [k for k, v in got.items() if v is None]
    if len(plans) < num_plans:
        missing.append(f"plans ({len(plans)}/{num_plans})")
    if missing:
        raise SystemExit(f"Timed out waiting for: {missing}")
    # Concatenate plans along the time axis. Note: each plan was built from a
    # different initial state, so the open-loop rollout drifts from reality
    # past plan #1. That's fine for "does this control sequence go NaN".
    plan = np.concatenate(plans, axis=0)
    return plan, plan_stamps, got["pose"], got["odom"], got["arm"]


def assemble_qpos_qvel(task, pose_msg, odom_msg, arm_msg):
    """Verbatim port of mppi_vla_node._assemble_state."""
    qpos = np.asarray(task.mj_model.qpos0, dtype=np.float64).copy()
    qvel = np.zeros(int(task.mj_model.nv), dtype=np.float64)

    pos = pose_msg.pose.position
    q = pose_msg.pose.orientation
    yaw = float(np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    ))
    qpos[0] = float(pos.x) - float(L * np.cos(yaw))
    qpos[1] = float(pos.y) + float(L * np.sin(yaw))
    qpos[2] = yaw

    qvel[0] = float(odom_msg.data[ODOM_LIN_VEL_START + 1])
    qvel[1] = float(odom_msg.data[ODOM_LIN_VEL_START])
    qvel[2] = float(odom_msg.data[BASE_VEL_YAW_RATE_INDEX])

    arm_data = np.asarray(arm_msg.data, dtype=np.float64)
    arm_pos = arm_data[:ARM_STATE_LEN]
    arm_vel = arm_data[ARM_STATE_LEN:2 * ARM_STATE_LEN]
    qpos[3:9] = arm_pos[:6]
    qvel[3:9] = arm_vel[:6]
    qpos[9] = qpos[10] = float(arm_pos[6])
    qvel[9] = qvel[10] = float(arm_vel[6])
    return qpos, qvel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_topic", default="/mppi/plan")
    ap.add_argument("--pose_topic", default="/global_pose")
    ap.add_argument("--odom_topic", default="/px4/bridge/vehicle_odometry")
    ap.add_argument("--arm_topic", default="/follower/joint_state")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--out_dir", default="/workspace/hydrax/mppi_videos/")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cam_distance", type=float, default=4.0)
    ap.add_argument("--cam_azimuth", type=float, default=90.0,
                    help="Static camera yaw (deg). Ignored if --chase is set.")
    ap.add_argument("--cam_elevation", type=float, default=-20.0)
    ap.add_argument("--cam_lookat_z", type=float, default=0.5)
    ap.add_argument("--chase", action="store_true",
                    help="Chase camera: sit behind the robot, rotate with yaw.")
    ap.add_argument("--chase_offset_deg", type=float, default=180.0)
    ap.add_argument("--num_plans", type=int, default=1,
                    help="Concatenate N consecutive plans into one long ctrl sequence.")
    ap.add_argument("--min_plan_gap_s", type=float, default=0.2,
                    help="Ignore plans arriving within this window of the previous "
                         "accepted one. Planner runs at 20 Hz so the default keeps "
                         "every ~4th plan.")
    ap.add_argument("--stop_on_nan", action="store_true",
                    help="Halt the rollout the first time qpos/qvel goes non-finite. "
                         "Without this, we keep stepping so you can see how bad it gets.")
    args = ap.parse_args()

    plan, plan_stamps, pose_msg, odom_msg, arm_msg = grab_inputs(
        args.plan_topic, args.pose_topic, args.odom_topic, args.arm_topic,
        args.timeout, args.num_plans, args.min_plan_gap_s,
    )
    print(f"Got plan shape={plan.shape}  stamps={plan_stamps}")

    task = AtmosM3()
    model = task.mj_model
    data = mujoco.MjData(model)
    nu = int(model.nu)
    sim_dt = float(model.opt.timestep)

    if plan.shape[1] != nu:
        print(f"WARNING: plan width {plan.shape[1]} != model.nu {nu}; "
              "this is the planner publishing the wrong channel count")

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    qpos0, qvel0 = assemble_qpos_qvel(task, pose_msg, odom_msg, arm_msg)
    print("=== seed state ===")
    _scan_nan(qpos0, "seed qpos")
    _scan_nan(qvel0, "seed qvel")
    print(
        f"qpos[0:3]=({qpos0[0]:.3f}, {qpos0[1]:.3f}, {qpos0[2]:.3f})  "
        f"qvel[0:3]=({qvel0[0]:.3f}, {qvel0[1]:.3f}, {qvel0[2]:.3f})  "
        f"arm_pos={qpos0[3:9]}"
    )
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    u_min = np.asarray(task.u_min, dtype=np.float64)
    u_max = np.asarray(task.u_max, dtype=np.float64)
    # The PUBLISHED plan should already be within u_min/u_max, but clip
    # defensively and warn if anything was actually out of range.
    pre_clip = plan.copy()
    ctrls = np.clip(plan, u_min, u_max)
    out_of_range = np.where(np.any(pre_clip != ctrls, axis=1))[0]
    if out_of_range.size:
        print(f"WARN: {out_of_range.size} plan rows had values outside u_min/u_max "
              f"(first row {out_of_range[0]})")
    print("=== input plan scan ===")
    _scan_nan(ctrls, "plan ctrls (post-clip)")

    # JIT ctrl_transform_with_integral / update_integral exactly like mppi_ros_sim_node.
    mjx_data = mjx.put_data(model, data)
    integral = jnp.zeros(1)
    jit_ctrl_transform = jax.jit(task.ctrl_transform_with_integral)
    jit_update_integral = jax.jit(task.update_integral)
    print("JITting ctrl_transform / update_integral...")
    t0 = time.time()
    _ac = jit_ctrl_transform(mjx_data, jnp.zeros(nu), integral)
    _ = jit_update_integral(mjx_data, jnp.zeros(nu), integral, _ac)
    jax.block_until_ready(_ac)
    print(f"JIT done in {time.time() - t0:.2f}s")

    # The planner's time_from_start increments by sim_dt per point, so one
    # mj_step per plan row. Keep the parameter explicit in case someone
    # changes the planner cadence.
    plan_dt = sim_dt
    steps_per_row = max(int(round(plan_dt / sim_dt)), 1)
    frame_period = 1.0 / args.fps
    print(
        f"sim_dt={sim_dt*1e3:.2f}ms  plan_dt={plan_dt*1e3:.1f}ms  "
        f"steps_per_row={steps_per_row}  rows={len(ctrls)}  "
        f"total_sim={len(ctrls)*plan_dt:.2f}s"
    )

    print("=== open-loop rollout ===")
    snaps = []
    next_snap_t = 0.0
    sim_t = 0.0
    nan_first_step: int | None = None
    nan_first_origin: str | None = None
    for row_idx, row in enumerate(ctrls):
        # Per-row input scan — quiet unless something is wrong.
        if not np.all(np.isfinite(row)):
            print(f"row {row_idx} (sim_t={sim_t*1e3:.0f}ms): INPUT u has non-finite")
            print(f"  u={row}")
            if nan_first_step is None:
                nan_first_step = row_idx
                nan_first_origin = "input u"
            if args.stop_on_nan:
                break
        u = jnp.asarray(row)
        for _ in range(steps_per_row):
            mjx_data = mjx_data.replace(
                qpos=jnp.asarray(data.qpos),
                qvel=jnp.asarray(data.qvel),
            )
            actual_ctrl = jit_ctrl_transform(mjx_data, u, integral)
            integral = jit_update_integral(mjx_data, u, integral, actual_ctrl)
            actual_np = np.asarray(actual_ctrl)
            integral_np = np.asarray(integral)
            if (not np.all(np.isfinite(actual_np))) and nan_first_step is None:
                print(f"row {row_idx} (sim_t={sim_t*1e3:.0f}ms): "
                      f"ctrl_transform produced NaN")
                print(f"  u={row}")
                print(f"  actual_ctrl={actual_np}")
                nan_first_step = row_idx
                nan_first_origin = "ctrl_transform"
            if (not np.all(np.isfinite(integral_np))) and nan_first_step is None:
                print(f"row {row_idx} (sim_t={sim_t*1e3:.0f}ms): "
                      f"update_integral produced NaN  integral={integral_np}")
                nan_first_step = row_idx
                nan_first_origin = "integral"
            data.ctrl[:] = actual_np
            mujoco.mj_step(model, data)
            sim_t += sim_dt
            qpos_bad = not np.all(np.isfinite(data.qpos))
            qvel_bad = not np.all(np.isfinite(data.qvel))
            if (qpos_bad or qvel_bad) and nan_first_step is None:
                print(f"row {row_idx} (sim_t={sim_t*1e3:.0f}ms): mj_step diverged")
                print(f"  qpos_ok={not qpos_bad}  qvel_ok={not qvel_bad}")
                print(f"  qpos={data.qpos}")
                print(f"  qvel={data.qvel}")
                print(f"  preceding u={row}  actual_ctrl={actual_np}")
                nan_first_step = row_idx
                nan_first_origin = "mj_step"
            if sim_t >= next_snap_t:
                snaps.append(data.qpos.copy())
                next_snap_t += frame_period
        if nan_first_step is not None and args.stop_on_nan:
            break

    if nan_first_step is None:
        print(f"Clean rollout: {len(ctrls)} rows, {sim_t:.2f}s sim, no NaN.")
    else:
        print(f"NaN summary: first appeared at row {nan_first_step} "
              f"(sim_t≈{nan_first_step*plan_dt*1e3:.0f}ms) origin={nan_first_origin!r}")
    print(f"Captured {len(snaps)} frames over {sim_t:.2f}s sim")

    if not snaps:
        print("No frames captured (rollout aborted before first frame). Skipping video.")
        return

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    rec = VideoRecorder(args.out_dir, width=args.width, height=args.height, fps=args.fps)
    if not rec.start():
        raise SystemExit("VideoRecorder failed to start")
    scratch = mujoco.MjData(model)
    skipped = 0
    for qpos in snaps:
        # Skip any NaN snapshots so the renderer doesn't crash on bad qpos.
        if not np.all(np.isfinite(qpos)):
            skipped += 1
            continue
        scratch.qpos[:] = qpos
        mujoco.mj_forward(model, scratch)
        if args.chase:
            cam.azimuth = math.degrees(float(qpos[2])) + args.chase_offset_deg
        cam.lookat[0] = float(qpos[0])
        cam.lookat[1] = float(qpos[1])
        cam.lookat[2] = args.cam_lookat_z
        renderer.update_scene(scratch, camera=cam)
        rec.add_frame(renderer.render().tobytes())
    rec.stop()
    if skipped:
        print(f"Skipped {skipped} NaN snapshots while rendering")
    print(f"Wrote {rec.video_path}")


if __name__ == "__main__":
    main()
