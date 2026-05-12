"""
ATMOS M3 – drone-style control pipeline helpers.

All functions are JAX-jittable (no Python-side branching on traced values).

Pipeline for real-robot deployment:

    velocity_cmd (body frame)
        │  vel_to_body_wrench()
        ▼
    (Fx_b, Fy_b, Tz) – body-frame wrench
        │  body_wrench_to_world_ctrl()
        ▼
    (Fx_w, Fy_w, Tz) – world-frame joint ctrl  ──► MuJoCo/MJX simulation
        │  body_wrench_to_thrusters()
        ▼
    (T_f, T_b, T_r, T_l) – 4 physical thruster forces

Control allocator geometry (cross arrangement, arm r = 0.15 m, body frame):
    T_f : front  thruster, fires +y,  pos =( r, 0)
    T_b : back   thruster, fires -y,  pos =(-r, 0)
    T_r : right  thruster, fires +x,  pos =( 0,-r)
    T_l : left   thruster, fires -x,  pos =( 0, r)

Allocation matrix  B (3×4),  [Fx, Fy, Tz]ᵀ = B @ [T_f, T_b, T_r, T_l]ᵀ:
    B = [[  0,   0,  1, -1 ],    ← Fx
         [  1,  -1,  0,  0 ],    ← Fy
         [  r,   r,  r,  r ]]    ← Tz

Pseudoinverse (BB^T = diag(2, 2, 4r²)):
    T_f = Fy_b/2  + Tz/(4r)
    T_b = -Fy_b/2 + Tz/(4r)
    T_r = Fx_b/2  + Tz/(4r)
    T_l = -Fx_b/2 + Tz/(4r)

Unidirectional thrusters (T ≥ 0):  apply jnp.clip(T, 0, T_max).
Bidirectional (ducted fans, reversible props): jnp.clip(T, -T_max, T_max).
"""

import jax.numpy as jnp
import jax


# ── Physical constants ─────────────────────────────────────────────────────────

THRUSTER_ARM = 0.11          # metres from CoM to each thruster site
ROBOT_MASS   = 16.8          # kg
#ROBOT_Iz     = 0.56          # kg·m²  (yaw inertia, largest principal value)
ROBOT_Iz     = 0.297          # kg·m²  (yaw inertia, largest principal value)

# Allocation matrix and its pseudoinverse (analytical, computed once at import)
_r = THRUSTER_ARM
ALLOC_B = jnp.array([
    [0.0,  0.0,  1.0, -1.0],   # Fx contributions
    [1.0, -1.0,  0.0,  0.0],   # Fy contributions
    [_r,   _r,   _r,   _r ],   # Tz contributions
])
# B @ B^T = diag(2, 2, 4r²)  → B_pinv = B^T @ inv(BB^T)
_BBt_inv = jnp.diag(jnp.array([0.5, 0.5, 1.0 / (4 * _r**2)]))
ALLOC_B_PINV = ALLOC_B.T @ _BBt_inv   # shape (4, 3)


# ── Tier 1 – velocity → body-frame wrench ─────────────────────────────────────

def vel_to_body_wrench(
    v_cmd: jax.Array,
    v_current_body: jax.Array,
    kp: jax.Array = jnp.array([50.0, 50.0, 20.0]),
) -> jax.Array:
    """Proportional velocity controller → body-frame wrench.

    Args:
        v_cmd:           (3,) desired velocity [vx, vy, wz] in body frame.
        v_current_body:  (3,) current velocity [vx, vy, wz] in body frame.
        kp:              (3,) proportional gains [kp_x, kp_y, kp_z].

    Returns:
        wrench_body: (3,) [Fx_b, Fy_b, Tz] body-frame force/torque.
    """
    err = v_cmd - v_current_body
    return kp * err


# ── Frame conversion helpers ───────────────────────────────────────────────────

def world_vel_to_body_vel(vel_world: jax.Array, yaw: jax.Array) -> jax.Array:
    """Rotate world-frame velocity [vx_w, vy_w, wz] to body frame.

    Args:
        vel_world: (3,) [vx_w, vy_w, wz]
        yaw:       scalar heading angle θ (hinge_z qpos).

    Returns:
        (3,) [vx_b, vy_b, wz]
    """
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    vx_b =  c * vel_world[0] + s * vel_world[1]
    vy_b = -s * vel_world[0] + c * vel_world[1]
    return jnp.array([vx_b, vy_b, vel_world[2]])


def body_wrench_to_world_ctrl(
    wrench_body: jax.Array, yaw: jax.Array
) -> jax.Array:
    """Rotate body-frame wrench to world-frame joint ctrl.

    The slide_x / slide_y actuators apply forces along world axes, so Fx_b/Fy_b
    must be rotated by the current yaw before being sent as ctrl.

    Args:
        wrench_body: (3,) [Fx_b, Fy_b, Tz].
        yaw:         scalar heading angle θ.

    Returns:
        ctrl: (3,) [Fx_w, Fy_w, Tz] – ready to write to mjx.Data.ctrl.
    """
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    fx_w = c * wrench_body[0] - s * wrench_body[1]
    fy_w = s * wrench_body[0] + c * wrench_body[1]
    return jnp.array([fx_w, fy_w, wrench_body[2]])


# ── Tier 3 – body-frame wrench → 4 physical thruster forces ───────────────────

def body_wrench_to_thrusters(
    wrench_body: jax.Array,
    T_max: float = 20.0,
    bidirectional: bool = True,
) -> jax.Array:
    """Control allocator: body-frame wrench → 4 thruster forces.

    Uses the minimum-norm (pseudoinverse) solution.  Forces are clipped to
    the physical saturation limits.

    Args:
        wrench_body:   (3,) [Fx_b, Fy_b, Tz].
        T_max:         maximum thruster force magnitude (N).
        bidirectional: if True, allow T ∈ [-T_max, T_max] (reversible props);
                       if False, clip to [0, T_max] (unidirectional).

    Returns:
        thrusters: (4,) [T_f, T_b, T_r, T_l] in Newtons.
    """
    T = ALLOC_B_PINV @ wrench_body   # (4,) minimum-norm solution
    if bidirectional:
        return jnp.clip(T, -T_max, T_max)
    else:
        return jnp.clip(T, 0.0, T_max)


def thrusters_to_body_wrench(thrusters: jax.Array) -> jax.Array:
    """Forward map: 4 thruster forces → body-frame wrench (for verification).

    Args:
        thrusters: (4,) [T_f, T_b, T_r, T_l].

    Returns:
        (3,) [Fx_b, Fy_b, Tz].
    """
    return ALLOC_B @ thrusters


# ── Full pipeline (jit-able) ───────────────────────────────────────────────────

def velocity_cmd_to_ctrl(
    v_cmd_body: jax.Array,
    qpos: jax.Array,
    qvel_world: jax.Array,
    kp: jax.Array = jnp.array([50.0, 50.0, 20.0]),
) -> jax.Array:
    """End-to-end: velocity command → world-frame joint ctrl.

    Combines vel_to_body_wrench + body_wrench_to_world_ctrl.
    Feed the result directly into mjx.Data.ctrl (or mj_data.ctrl).

    Args:
        v_cmd_body:   (3,) desired [vx, vy, wz] in body frame.
        qpos:         (3,) joint positions [x, y, theta].
        qvel_world:   (3,) joint velocities [vx_w, vy_w, wz] (world frame).
        kp:           (3,) proportional gains.

    Returns:
        ctrl: (3,) [Fx_w, Fy_w, Tz] clipped to joint actuator ranges.
    """
    yaw = qpos[2]
    vel_body = world_vel_to_body_vel(qvel_world, yaw)
    wrench_body = vel_to_body_wrench(v_cmd_body, vel_body, kp)
    ctrl = body_wrench_to_world_ctrl(wrench_body, yaw)
    # Clip to actuator ctrlrange defined in XML
    ctrl = jnp.clip(ctrl, jnp.array([-25.0, -25.0, -8.0]),
                          jnp.array([ 25.0,  25.0,  8.0]))
    return ctrl
