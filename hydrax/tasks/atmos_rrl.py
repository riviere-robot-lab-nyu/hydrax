import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

class Atmos(Task):

    def __init__(self, 
                 goal: jax.Array = jnp.array([4.0, 3.0, 0., 0.0, 0.0, 0.0]), 
                 bad: jax.Array = jnp.array([1.5, 1.5]),
                 state_cost: jax.Array = jnp.array([100., 100., 5.0, 100., 100., 5.0]), 
                 ctrl_cost: jax.Array = 5*jnp.ones(8,),
                 u_on: jax.Array = 1.7*jnp.ones(8,),
                 u_off: jax.Array | None = None,
                 threshold: float = 0.5) -> None:#, u_on: jax.Array = 1.7*jnp.ones(8), u_off: jax.Array | None = None, threshold: float = 0.5) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/atmos_rrl/atmos_rrl.xml"
        )
        super().__init__(mj_model, trace_sites=["thrust_site_FR"])
        self.goal = goal
        self.Q = state_cost
        self.R = ctrl_cost
        self.u_on=u_on
        self.bad = bad
        if u_off is None:
            self.u_off = jnp.zeros_like(u_on)
        else:
            self.u_off=u_off
        self.threshold = threshold
        

    def _distance_to_goal(self, state: mjx.Data) -> jax.Array:
        # print(state.qpos[:2])
        # jax.debug.print("qpos:  {}", state.qpos[:2])
        # return (1.0 - state.xpos[self.mj_model.body("atmos").id,0])**2 + (1.0 - state.xpos[self.mj_model.body("atmos").id, 1])**2
        return jnp.sum(jnp.square(self.goal[:2]-state.qpos[:2]))
        #return (1.0 - state.qpos[0])**2 + (1.0 - state.qpos[1])**2
    
    def _state_cost(self, state: mjx.Data) -> jax.Array:
        err = self.goal - jnp.concatenate([state.qpos, state.qvel])
        # pen = jnp.linalg.norm(state.qpos[:2] - self.bad)
        # penalty_cost = 10000 * jnp.exp(-pen**2)
        # err = jnp.where(pen < 0.5, err + penalty_cost, err)
        return jnp.sum(self.Q * jnp.square(err))

    def _ctrl_cost(self, control: jax.Array) -> jax.Array:
        # return jnp.sum(self.R*jnp.square(control))
        return jnp.sum(self.R*jnp.abs(control))
    
    def _circle_cost(self, state:mjx.Data, radius: float = 1.0, period: float =20.0) -> jax.Array:
        
        B = 2*jnp.pi/period
        x_des = radius * jnp.cos(B * state.time)- radius
        y_des = radius * jnp.sin(B * state.time)
        x_des_dot = -radius * B * jnp.sin(B*state.time)
        y_des_dot = B * radius* jnp.cos(B*state.time)
        return 10*((state.qpos[0]-x_des)**2 + (state.qpos[1] - y_des)**2 + (state.qvel[0]-x_des_dot)**2 + (state.qvel[1]-y_des_dot)**2) + 10*state.qvel[2]**2
        

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        # dist_goal = self._distance_to_goal(state)
        # # print(dist_goal)
        # dist_goal = jnp.clip(dist_goal, 0, 100)
        # dist_goal = jnp.reshape(dist_goal, ())
        # jax.debug.print("dist_goal: {}", dist_goal)
        # jax.debug.print("GOAL: {}", self.goal[:2])
        # jax.debug.print("STATE: {}", state.qpos[:2])
        # jax.debug.print("CONTROL: {}", control)
        # control_cost = 10*0.1 * jnp.sum(jnp.square(control))
        # jax.debug.print("TOTAL COST: {}", dist_goal + control_cost)
        # return 3*dist_goal + control_cost + 0.1*state.qvel[0]**2 + 0.1*state.qvel[1]**2 + state.qvel[5]**2
        return self._state_cost(state) + self._ctrl_cost(control)
        # return self._circle_cost(state) + 0.1*self._ctrl_cost(control)
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        # return 100 * self._distance_to_goal(state) + 100*state.qvel[0]**2 + 100*state.qvel[1]**2 + 100 * state.qvel[5]**2
        return 80*self._state_cost(state)
        # return 80*self._circle_cost(state)