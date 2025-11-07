import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
# from hydrax import ROOT
from hydrax.task_base import Task
from pathlib import Path

class CartpoleSam(Task):
    def __init__(self) -> None:
        current_file = Path(__file__).resolve()
        ROOT = str(current_file.parent.parent)
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/Sam_test/cart_poleV2.xml")
        super().__init__(mj_model, trace_sites=["tip"])

        self.target_x = 0.0
        self.target_angle_factor = jnp.pi
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        theta = state.qpos[1] + jnp.pi
        theta_err = jnp.array([jnp.cos(theta)-1, jnp.sin(theta)])
        theta_err = jnp.sum(jnp.square(theta_err))
        x_cost = jnp.sum(jnp.square(state.qpos[0]))
        vel_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        control_cost = 0.01*jnp.sum(jnp.square(control))
        return theta_err + x_cost + vel_cost + control_cost
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        theta = state.qpos[1] + jnp.pi
        theta_err = 10*jnp.array([jnp.cos(theta)-1, jnp.sin(theta)])
        theta_err = jnp.sum(jnp.square(theta_err))
        x_cost = jnp.sum(jnp.square(state.qpos[0]))
        vel_cost = 0.01*jnp.sum(jnp.square(state.qvel))
        return theta_err + x_cost + vel_cost
    
