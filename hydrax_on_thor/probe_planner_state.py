"""Confirm whether the zeroed-qpos init in mppi_ros_node._assemble_state
puts the AtmosM3 model into an unstable / invalid state.

Theory under test: the planner does `qpos = np.zeros(nq)` and only writes
the base slots [0:3]. If the arm / gripper qpos slots have a non-zero
`qpos0` (model default), starting them at zero is an out-of-keyframe
configuration that may diverge under mjx.step.

What this script does (no ROS, no rclpy):
  1. Build AtmosM3 task.
  2. Print qpos0 for each joint (model default).
  3. Print joint ranges / types so you can see if the arm has limits the
     zero pose violates.
  4. Roll mjx forward with zero controls for N steps starting from:
       (a) the model's qpos0   (sanity baseline)
       (b) all-zero qpos       (what the planner actually sends)
     Report per-step finite-ness, max |qpos|, max |qvel|. The first step
     at which (b) goes non-finite or wildly larger than (a) is the smoking
     gun.

Run:
  python3 probe_planner_state.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.tasks.atmos_m3 import AtmosM3


N_STEPS = 200       # ~2 s at 100 Hz
PRINT_EVERY = 20


def _joint_label(mj_model: mujoco.MjModel, jnt_id: int) -> str:
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or "?"
    jtype = int(mj_model.jnt_type[jnt_id])
    typename = {
        mujoco.mjtJoint.mjJNT_FREE: "FREE(7)",
        mujoco.mjtJoint.mjJNT_BALL: "BALL(4)",
        mujoco.mjtJoint.mjJNT_SLIDE: "SLIDE(1)",
        mujoco.mjtJoint.mjJNT_HINGE: "HINGE(1)",
    }.get(jtype, f"?({jtype})")
    limited = bool(mj_model.jnt_limited[jnt_id])
    if limited:
        lo, hi = mj_model.jnt_range[jnt_id]
        rng_str = f"range=[{lo:+.3f},{hi:+.3f}]"
    else:
        rng_str = "unlimited"
    return f"[{jnt_id}] {name:<24s} {typename:<8s} {rng_str}"


def _rollout(mj_model: mujoco.MjModel, qpos_init: np.ndarray) -> dict:
    """Step mjx for N_STEPS with zero controls. Return per-step stats."""
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)
    data = data.replace(
        qpos=jnp.asarray(qpos_init, dtype=jnp.float32),
        qvel=jnp.zeros(mj_model.nv, dtype=jnp.float32),
        ctrl=jnp.zeros(mj_model.nu, dtype=jnp.float32),
        time=jnp.float32(0.0),
    )

    @jax.jit
    def step(d):
        return mjx.step(model, d)

    out = {
        "step_first_nonfinite": -1,
        "max_qpos_abs_at_end": float("nan"),
        "max_qvel_abs_at_end": float("nan"),
        "qpos_trace": [],   # (step, max|qpos|, max|qvel|, finite?) sampled
    }
    for i in range(N_STEPS):
        data = step(data)
        qp = np.asarray(data.qpos)
        qv = np.asarray(data.qvel)
        finite = bool(np.isfinite(qp).all() and np.isfinite(qv).all())
        if not finite and out["step_first_nonfinite"] == -1:
            out["step_first_nonfinite"] = i
            out["qpos_trace"].append((i, float(np.nanmax(np.abs(qp))),
                                      float(np.nanmax(np.abs(qv))), False))
            break
        if (i % PRINT_EVERY) == 0:
            out["qpos_trace"].append((i, float(np.max(np.abs(qp))),
                                      float(np.max(np.abs(qv))), True))
    # Final stats (or last-finite if diverged).
    qp = np.asarray(data.qpos)
    qv = np.asarray(data.qvel)
    if np.isfinite(qp).all():
        out["max_qpos_abs_at_end"] = float(np.max(np.abs(qp)))
    if np.isfinite(qv).all():
        out["max_qvel_abs_at_end"] = float(np.max(np.abs(qv)))
    return out


def main() -> None:
    task = AtmosM3()
    mj = task.mj_model
    print(f"AtmosM3 loaded. nq={mj.nq} nv={mj.nv} nu={mj.nu}")
    print(f"sim_dt={mj.opt.timestep*1000:.2f}ms\n")

    print("Joints (mj id, name, type, range):")
    for j in range(mj.njnt):
        print(f"  {_joint_label(mj, j)}")
    print()

    # qpos0 may differ from zeros (e.g., quaternion for free joint = [1,0,0,0]).
    print(f"qpos0 (model default) = {np.asarray(mj.qpos0)}")
    print(f"qpos_spring         = {np.asarray(mj.qpos_spring)}")

    # Is there a 'home' or named keyframe?
    if mj.nkey > 0:
        print(f"\nNamed keyframes ({mj.nkey}):")
        for k in range(mj.nkey):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_KEY, k) or "?"
            print(f"  [{k}] {name}: qpos={np.asarray(mj.key_qpos[k])}")
    else:
        print("\n(no named keyframes)")

    print("\n" + "=" * 70)
    print("Rollout A: qpos = mj.qpos0 (model default)")
    print("=" * 70)
    a = _rollout(mj, np.asarray(mj.qpos0))
    if a["step_first_nonfinite"] >= 0:
        print(f"  diverged at step {a['step_first_nonfinite']}")
    else:
        print(f"  stayed finite for all {N_STEPS} steps")
    print(f"  max|qpos| at end: {a['max_qpos_abs_at_end']:.4f}")
    print(f"  max|qvel| at end: {a['max_qvel_abs_at_end']:.4f}")
    for step, mp, mv, finite in a["qpos_trace"]:
        print(f"    step {step:3d}: max|qpos|={mp:8.3f} max|qvel|={mv:8.3f} finite={finite}")

    print("\n" + "=" * 70)
    print("Rollout B: qpos = np.zeros(nq)  (what the planner sends)")
    print("=" * 70)
    b = _rollout(mj, np.zeros(mj.nq))
    if b["step_first_nonfinite"] >= 0:
        print(f"  *** diverged at step {b['step_first_nonfinite']} ***")
    else:
        print(f"  stayed finite for all {N_STEPS} steps")
    print(f"  max|qpos| at end: {b['max_qpos_abs_at_end']:.4f}")
    print(f"  max|qvel| at end: {b['max_qvel_abs_at_end']:.4f}")
    for step, mp, mv, finite in b["qpos_trace"]:
        print(f"    step {step:3d}: max|qpos|={mp:8.3f} max|qvel|={mv:8.3f} finite={finite}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if b["step_first_nonfinite"] >= 0 and a["step_first_nonfinite"] < 0:
        print(
            "*** Hypothesis CONFIRMED: zero-init qpos diverges, model-default "
            "does not. The fix is in _assemble_state: initialize qpos from "
            "mj_model.qpos0 (or a named keyframe), then overwrite only the "
            "base indices [0:3]."
        )
    elif b["step_first_nonfinite"] < 0 and a["step_first_nonfinite"] < 0:
        print(
            "Both rollouts stayed finite. The zero-init qpos itself is not "
            "the divergence cause. NaN must originate elsewhere — likely "
            "the cost function or the optimize loop (try plan_health_check)."
        )
    elif b["step_first_nonfinite"] >= 0 and a["step_first_nonfinite"] >= 0:
        print(
            "Both rollouts diverge — the model itself is unstable under zero "
            "controls. Check actuator gain ranges and integrator settings."
        )
    else:
        print(
            "Default-qpos rollout diverged but zero-qpos did not — unusual; "
            "inspect the model file."
        )


if __name__ == "__main__":
    main()
