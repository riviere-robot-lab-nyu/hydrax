import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

class Atmos(Task):

    def __init__(self) -> None:#, u_on: jax.Array = 1.7*jnp.ones(8), u_off: jax.Array | None = None, threshold: float = 0.5) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/atmos_rrl/atmos_rrl.xml"
        )
        super().__init__(mj_model, trace_sites=["thrust_site_FR"])
        # self.goal = jnp.array([1.0, 1.0, 0.0])
        # self.u_on=u_on
        # if u_off is None:
        #     self.u_off = jnp.zeros_like(u_on)
        # else:
        #     self.u_off=u_off
        # self.threshold = threshold
        

    def _distance_to_goal(self, state: mjx.Data) -> jax.Array:
        # print(state.qpos[:2])
        # jax.debug.print("qpos:  {}", state.qpos[:2])
        return (1.0 - state.xpos[self.mj_model.body("atmos").id,0])**2 + (1.0 - state.xpos[self.mj_model.body("atmos").id, 1])**2
        return jnp.sum(jnp.square(self.goal[:2]-state.qpos[:2]))
        # return (self.goal[0] - state.qpos[0])**2 + (self.goal[1]-state.qpos[1])**2

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        dist_goal = self._distance_to_goal(state)
        # print(dist_goal)
        dist_goal = jnp.clip(dist_goal, 0, 100)
        dist_goal = jnp.reshape(dist_goal, ())
        # jax.debug.print("dist_goal: {}", dist_goal)
        # jax.debug.print("GOAL: {}", self.goal[:2])
        # jax.debug.print("STATE: {}", state.qpos[:2])
        # jax.debug.print("CONTROL: {}", control)
        control_cost = 0.1 * jnp.sum(jnp.square(control))
        # jax.debug.print("TOTAL COST: {}", dist_goal + control_cost)
        return dist_goal + control_cost
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return 10 * self._distance_to_goal(state)