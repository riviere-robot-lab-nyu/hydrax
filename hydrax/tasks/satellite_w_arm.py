import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

class Sat_w_arm(Task):
    def __init__(self,
                 goal: jax.Array = jnp.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                state_cost: jax.Array = jnp.array([500000., 500000., 500000., 500000., 500000., 500000., 500000., 500000., 500000., 500000., 500000., 500000.]),
                ctrl_cost: jax.Array = 0.001*jnp.ones(18,),
                u_on: jax.Array = 50. * jnp.ones(18,),
                u_off: jax.Array | None = None,
                threshold: float = 0.5,
                w_object: bool = False) -> None:
        if w_object:
            mj_model = mujoco.MjModel.from_xml_path(
                ROOT + "/models/space_craft/wxai_follower_satellite2_w_object.xml"
            )
        else:
            mj_model = mujoco.MjModel.from_xml_path(
                ROOT + "/models/space_craft/wxai_follower_satellite2.xml"
            )
        
        # CORRECT (Remove '.opt')
        # Disable collisions for ALL geometries by zeroing out their collision masks
# The '[:]' syntax is crucial: it modifies the array in place.
        mj_model.geom_conaffinity[:] = 0
        mj_model.geom_contype[:] = 0
        super().__init__(mj_model, trace_sites=["thruster1_tip"])
        self.goal = goal
        self.p_goal = goal[:3]
        self.quat_goal = goal[3:7]
        self.v_goal = goal[7:10]
        self.w_goal = goal[10:13]
        self.Q = state_cost
        self.R = ctrl_cost
        self.u_on = u_on
        if u_off is None:
            self.u_off = jnp.zeros_like(u_on)
        else:
            self.u_off = u_off
        
        self.threshold = threshold
        self.f=0.0002
        self.t = 1
        # self.u_min = jnp.where(
        #     mj_model.actuator_ctrllimited,
        #     mj_model.actuator_ctrlrange[:, 0],
        #     -jnp.inf,
        # )
        # self.u_min = self.u_min[:18]
        # self.u_max = jnp.where(
        #     mj_model.actuator_ctrllimited,
        #     mj_model.actuator_ctrlrange[:, 1],
        #     jnp.inf,
        # )
        # self.u_max = self.u_max[:18]

    def _get_quat_error(self, quat: jax.Array) -> jax.Array:
        w, x, y, z = quat
        wr, xr, yr, zr = self.quat_goal

        x_err =  wr * x - xr * w - yr * z + zr * y
        y_err =  wr * y + xr * z - yr * w - zr * x
        z_err =  wr * z - xr * y + yr * x - zr * w

        e_orient = jnp.array([x_err, y_err, z_err])

        return e_orient
    
    def _get_state_cost(self, state: mjx.Data) -> jax.Array:
        quat_err = self._get_quat_error(state.qpos[3:7])
        pos_err = self.p_goal - state.qpos[:3]
        v_err = self.v_goal - state.qvel[:3]
        w_err = self.w_goal - state.qvel[3:6]
        return jnp.sum((quat_err ** 2) * self.Q[3:6]) + jnp.sum((pos_err**2)*self.Q[:3]) + jnp.sum((v_err**2)*self.Q[6:9]) + jnp.sum((w_err**2)*self.Q[9:])
    
    def _get_ctrl_cost(self, control: jax.Array) -> jax.Array:
        return jnp.sum(self.R*jnp.square(control[:18]))
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        return self._get_state_cost(state) + self._get_ctrl_cost(control) + self._get_arm_cost(state)
    
    def _get_arm_cost(self, state: mjx.Data) -> jax.Array:
        time = state.time
        q1 = 1.5*jnp.sin(self.f*time)
        q2 = 1.5*jnp.cos(self.f*time)
        q3 = 1.5*jnp.sin(2*self.f*time + 0.5)
        desired_arm_state = jnp.array([q1, q2, q3, 0.0, 0.0, 0.0, 0., 0.])
        desired_arm_vel = jnp.array([self.f*1.5*jnp.cos(self.f*time),-self.f*1.5*jnp.sin(self.f*time), 2*self.f*1.5*jnp.cos(2*self.f*time + 0.5), 0., 0., 0., 0., 0.])
        cost_q = jnp.sum(jnp.square(desired_arm_state - state.qpos[7:]))*1.
        cost_v = jnp.sum(jnp.square(desired_arm_vel - state.qvel[6:]))*1.
        return cost_q + cost_v

    def terminal_cost(self, state:mjx.Data) -> jax.Array:
        return 100*self._get_state_cost(state) + 0*1000*jnp.square(state.qvel) + 100*self._get_arm_cost(state)
    
    def get_arm_traj(self, times: jax.Array) -> jax.Array:
        times = jnp.atleast_1d(times)
        q1 = 1.5*jnp.sin(self.f*times)
        q2 = 1.5*jnp.cos(self.f*times)
        q3 = 1.5*jnp.sin(2*self.f*times + 0.5)
        zeros = jnp.zeros((times.shape[0], 5))
        arm_traj = jnp.column_stack([q1, q2, q3, zeros])
        return arm_traj