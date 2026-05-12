"""
ATMOS M3 – hydrax Task for the planar ATMOS + WidowX AI arm.

ctrl / qpos layout  (nu = nq = nv = 11):
  idx  meaning                         units
  0    vx_cmd  – desired body-x vel    m/s
  1    vy_cmd  – desired body-y vel    m/s
  2    wz_cmd  – desired yaw rate      rad/s
  3    joint_0 – shoulder pan          rad
  4    joint_1 – shoulder lift         rad
  5    joint_2 – elbow                 rad
  6    joint_3 – wrist angle           rad
  7    joint_4 – wrist rotate          rad
  8    joint_5 – wrist servo           rad
  9    gripper – open/close command    [0=closed, 1=open]  (binary, drives both carriages)
  10   (unused – both carriages driven from ctrl[9])

Arm control modes (static, chosen at task construction):
  "absolute" – ctrl[3:9] is written directly as the position servo target.
               The servo moves the joint to that angle.
  "delta"    – ctrl[3:9] is added to the current joint position before
               being written as the servo target.
               The servo moves the joint by that amount from where it is now.

Both modes are sampled by MPPI the same way (mean + noise).  The mode only
changes what ctrl_transform does with the sample before passing it to MuJoCo.

Gripper: ctrl[9] ∈ [0, 1].  Thresholded at 0.5:
  > 0.5 → open  (both carriages target 0.044 m)
  ≤ 0.5 → closed (both carriages target 0.0 m)
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import Literal

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.tasks.atmos_control import (
    world_vel_to_body_vel,
    vel_to_body_wrench,
    body_wrench_to_world_ctrl,
)

# Physical force/torque saturation for the base thrusters
_F_MAX = 25.0   # N
_T_MAX = 8.0    # N·m

THRUSTER_ARM = 0.11          # metres from CoM to each thruster site
THRUSTER_MAX = 1.5
# Arm joint hard limits  (order matches qpos / ctrl indices 3:9)
_ARM_JOINT_MIN = jnp.array([-3.05433, 0.0,     0.0,     -1.5708, -1.5708, -3.14159])
_ARM_JOINT_MAX = jnp.array([ 3.05433, 3.14159, 2.35619,  1.5708,  1.5708,  3.14159])

# Per-step delta limits (delta mode only)
_ARM_DELTA_MAX = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

_GRIPPER_OPEN   = 0.044   # m
_GRIPPER_CLOSED = 0.0     # m
_MAX_TAU = 4*THRUSTER_MAX*THRUSTER_ARM

class AtmosM3(Task):
    """Planar ATMOS + WidowX AI arm task.

    MPPI action space (10 meaningful dims, 1 dead dim):
      ctrl[0:3]  – base velocity commands [vx, vy, wz] body frame
      ctrl[3:9]  – arm joint commands (absolute angles or deltas, see arm_mode)
      ctrl[9]    – gripper: 0 = closed, 1 = open (thresholded at 0.5)
      ctrl[10]   – dead (always 0; both gripper carriages driven from ctrl[9])

    Args:
        goal:          (6,) target base state [x, y, theta, vx_w, vy_w, wz].
        state_cost:    (6,) Q weights on base state error.
        ctrl_cost:     (3,) R weights on base velocity commands.
        arm_ctrl_cost: (6,) R weights on arm joint commands.
        kp:            (3,) P-gains for the velocity → force control law.
        arm_mode:      "absolute" – servo targets are commanded directly.
                       "delta"    – servo targets = current_qpos + command.
        use_robot_only: load atmos_robot.xml instead of scene.xml.
    """

    def __init__(
        self,
        goal: jax.Array = jnp.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        state_cost: jax.Array = jnp.array([1000.0, 1000.0, 500.0, 1000.0, 1000.0, 20.0]),
        ctrl_cost: jax.Array = jnp.zeros(3),
        arm_ctrl_cost: jax.Array = jnp.zeros(6),
        kp_vel: float = 6.55,
        kp_att: float = 2.8,
        kp_rate: float = 10.0,
        ki_rate: float = 0.865,
        i_fac: float = 6.981317,
        rate_i_lim: float = 0.2,
        arm_mode: Literal["absolute", "delta"] = "absolute",
        use_robot_only: bool = False,
    ) -> None:
        xml_name = "atmos_robot.xml" if use_robot_only else "scene.xml"
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/m3/" + xml_name)
        super().__init__(mj_model, trace_sites=["body_com"])

        self.goal = goal
        self.Q = state_cost
        self.R = ctrl_cost
        self.R_arm = arm_ctrl_cost
        self.kp_vel = kp_vel
        self.kp_att = kp_att
        self.kp_rate = kp_rate
        self.ki_rate = ki_rate
        self.i_fac = i_fac
        self.rate_i_lim = rate_i_lim
        self.arm_mode = arm_mode

        # u_min/u_max define what the optimizer samples — not forces/torques.
        base_min = jnp.array([-2.0, -2.0, -1.0])
        base_max = jnp.array([ 2.0,  2.0,  1.0])

        if arm_mode == "absolute":
            arm_min, arm_max = _ARM_JOINT_MIN, _ARM_JOINT_MAX
        else:  # delta
            arm_min, arm_max = -_ARM_DELTA_MAX, _ARM_DELTA_MAX

        # ctrl[9] = gripper binary [0,1]; ctrl[10] = dead (force to 0)
        self.u_min = jnp.concatenate([base_min, arm_min, jnp.array([0.0, 0.0])])
        self.u_max = jnp.concatenate([base_max, arm_max, jnp.array([1.0, 0.0])])
        self.B = jnp.array([[0., 0., -THRUSTER_MAX, THRUSTER_MAX],
                            [THRUSTER_MAX, -THRUSTER_MAX, 0., 0.],
                            [THRUSTER_MAX*THRUSTER_ARM, THRUSTER_MAX*THRUSTER_ARM, THRUSTER_MAX*THRUSTER_ARM, THRUSTER_MAX*THRUSTER_ARM]])
        self.B_inv = jnp.array([[0., 1., 1.],[0., -1., 1.],[-1., 0., 1.],[1., 0., 1.]])
        #self.B_inv = jnp.linalg.pinv(self.B)
        self.wrench_min= jnp.array([-2*THRUSTER_MAX, -2*THRUSTER_MAX, -4*THRUSTER_MAX*THRUSTER_ARM])
        self.wrench_max= jnp.array([2*THRUSTER_MAX, 2*THRUSTER_MAX, 4*THRUSTER_ARM*THRUSTER_MAX])


    # ── Inner-loop control law ────────────────────────────────────────────────

    def ctrl_transform_with_integral(
        self, state: mjx.Data, ctrl: jax.Array, integral: jax.Array
    ) -> jax.Array:
        """Convert sampled commands → actuator commands written to data.ctrl.

        Base (ctrl[0:3]):
          PI velocity controller → world-frame forces for slide/hinge joints.
          integral[0] carries the yaw-rate integral error.

        Arm (ctrl[3:9]):
          absolute: clipped target angles sent directly to position servos.
          delta:    current qpos + ctrl, clipped, sent to position servos.

        Gripper (ctrl[9], binary):
          > 0.5 → open (0.044 m), ≤ 0.5 → closed (0.0 m).
          Both carriages (data.ctrl[9] and [10]) get the same target.
        """
        # ── Base ──
        yaw = state.qpos[2]
        vel_body = world_vel_to_body_vel(state.qvel[:3], yaw)
        vel_error = ctrl[0:2] - vel_body[0:2]
        #yaw_error = ctrl[2] - yaw
        #omega_des = self.kp_att * 2 * jnp.sin(yaw_error/2)
        #omega_err = omega_des - vel_body[2]
        omega_err = ctrl[2] - vel_body[2]
        wrench_body = jnp.zeros_like(vel_body)
        wrench_body = wrench_body.at[0:2].set(self.kp_vel * vel_error)
        wrench_body = wrench_body.at[2].set(self.kp_rate*omega_err + integral[0])
        wrench_body = wrench_body.clip(self.wrench_min, self.wrench_max)
        thrust = (self.B_inv @ wrench_body).clip(-1., 1.)
        wrench_body_real = self.B @ thrust
        wrench_world = body_wrench_to_world_ctrl(wrench_body_real, yaw)



        
        # ── Arm joints ──
        if self.arm_mode == "absolute":
            arm_ctrl = jnp.clip(ctrl[3:9], _ARM_JOINT_MIN, _ARM_JOINT_MAX)
        else:  # delta
            arm_ctrl = jnp.clip(
                state.qpos[3:9] + ctrl[3:9], _ARM_JOINT_MIN, _ARM_JOINT_MAX
            )

        # ── Gripper (binary, both carriages coupled) ──
        gripper_target = jnp.where(ctrl[9] > 0.5, _GRIPPER_OPEN, _GRIPPER_CLOSED)

        return jnp.concatenate([
            #base_ctrl,
            wrench_world,
            arm_ctrl,
            jnp.array([gripper_target, gripper_target]),  # right then left carriage
        ])

    def update_integral(
        self, state: mjx.Data, ctrl: jax.Array, integral: jax.Array, actual_ctrl: jax.Array = None
    ) -> jax.Array:
        """Update the yaw-rate integral error.

        Args:
            state:       Current simulation state.
            ctrl:        Raw optimizer action (ctrl[2] = wz_cmd, body frame).
            integral:    Shape (1,), current yaw-rate integral.
            actual_ctrl: Actuator command from ctrl_transform_with_integral (use actual_ctrl[:3] to check saturation).

        Returns:
            Updated integral, shape (1,).
        """
        # TODO: implement your integral update formula here.
        omega_err = ctrl[2] - state.qvel[2]
        saturated = ((actual_ctrl[2] >= _MAX_TAU) & (omega_err > 0)) | \
                    ((actual_ctrl[2] <= -_MAX_TAU) & (omega_err < 0))
        ifac = omega_err / self.i_fac
        ifac = jnp.maximum(0., 1 - ifac**2)
        new_integral = integral + ifac * self.ki_rate * omega_err * self.dt
        new_integral = jnp.clip(new_integral, -self.rate_i_lim, self.rate_i_lim)
        return jnp.where(saturated, integral, new_integral)

    # ── Cost components ───────────────────────────────────────────────────────

    def _base_state_err(self, state: mjx.Data) -> jax.Array:
        """6-D base state error [Δx, Δy, Δθ_wrapped, Δvx, Δvy, Δwz]."""
        dtheta = state.qpos[2] - self.goal[2]
        dtheta = (dtheta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.concatenate([
            state.qpos[:2] - self.goal[:2],
           jnp.array([dtheta]),
            state.qvel[:3] - self.goal[3:],
        ])

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        err = self._base_state_err(state)
        base_state = jnp.sum(self.Q   * jnp.square(err))
        base_ctrl  = jnp.sum(self.R   * jnp.square(control[:3]))
        arm_ctrl   = jnp.sum(self.R_arm * jnp.square(control[3:9]))
        arm_ctrl = arm_ctrl + jnp.sum(jnp.square(100. * state.qpos[3:9]))
        return base_state + base_ctrl + arm_ctrl

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return 10.0 * jnp.sum(self.Q * jnp.square(self._base_state_err(state))) * self.dt
