import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

class Sat_no_arm(Task):

    def __init__(self,
                 goal: jax.Array = jnp.array([2., 2., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
                 state_cost: jax.Array = jnp.array([5000., 5000., 5000., 5000., 5000., 5000., 1000., 1000., 1000., 100000., 100000., 1000.]),
                 ctrl_cost: jax.Array = 1.0*jnp.ones(18,),
                 u_on: jax.Array = 50. * jnp.ones(18,),
                 u_off: jax.Array | None = None,
                 threshold: float = 0.5) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/space_craft/satellite2.xml"
        )
        super().__init__(mj_model, trace_sites=["thruster1_tip"])
        self.goal = goal
        self.p_goal = goal[:3]
        self.quat_goal = goal[3:7]
        self.v_goal = goal[7:10]
        self.w_goal = goal[10:]
        self.Q = state_cost
        self.R = ctrl_cost
        self.u_on = u_on
        if u_off is None:
            self.u_off = jnp.zeros_like(u_on)
        else:
            self.u_off = u_off
        self.threshold = threshold
    
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
        w_err = self.w_goal - state.qvel[3:]
        return jnp.sum((quat_err ** 2) * self.Q[3:6]) + jnp.sum((pos_err**2)*self.Q[:3]) + jnp.sum((v_err**2)*self.Q[6:9]) + jnp.sum((w_err**2)*self.Q[9:])
    
    def _get_ctrl_cost(self, control: jax.Array) -> jax.Array:
        return jnp.sum(self.R*jnp.square(control))
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        return self._get_state_cost(state) + self._get_ctrl_cost(control)

    def terminal_cost(self, state:mjx.Data) -> jax.Array:
        return 200*self._get_state_cost(state) + 1000*jnp.square(state.qvel)

