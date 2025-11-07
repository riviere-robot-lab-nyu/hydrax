import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

_STAND_HEIGHT = 1.2

WALK_SPEED = 1
_DEFAULT_VALUE_AT_MARGIN = 0.1

class Walker(Task):
    """A planar biped tasked with walking forward."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/walker/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["torso_site"])

        # Get sensor ids
        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        # Set the target velocity (m/s) and height
        # TODO: make these parameters
        self.target_velocity = 1.0
        self.target_height = 1.2

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
        return state.sensordata[sensor_adr + 2]  # px, py, pz

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_deviation_from_upright(self, state: mjx.Data) -> jax.Array:
        """Get the deviation of the torso from the upright position."""
        sensor_adr = self.model.sensor_adr[self.torso_zaxis_sensor]
        return state.sensordata[sensor_adr + 2] - 1.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        return self._get_move_reward(state)
    
    def terminal_cost(self, state:mjx.Data) -> jax.Array:
        return self._get_move_reward(state)

    # def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
    #     """The running cost ℓ(xₜ, uₜ)."""
    #     state_cost = self.terminal_cost(state)
    #     control_cost = jnp.sum(jnp.square(control))
    #     return state_cost + 0.1 * control_cost

    # def terminal_cost(self, state: mjx.Data) -> jax.Array:
    #     """The terminal cost ϕ(x_T)."""
    #     height_cost = jnp.square(
    #         self._get_torso_height(state) - self.target_height
    #     )
    #     orientation_cost = jnp.square(
    #         self._get_torso_deviation_from_upright(state)
    #     )
    #     velocity_cost = jnp.square(
    #         self._get_torso_velocity(state) - self.target_velocity
    #     )
    #     return 10.0 * height_cost + 3.0 * orientation_cost + 1.0 * velocity_cost
    

    #Added by me to match reward model of mujoco_playground WalkerWalk environment

    # sigmoids and tolerance are taken from mujoco_playground
    def _sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid in ("linear", "quadratic"):
            if not 0 <= value_at_1 < 1:
                raise ValueError(
                    f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
      )
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError(
                    f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
                )

        if sigmoid == "gaussian":
            scale = jnp.sqrt(-2 * jnp.log(value_at_1))
            return jnp.exp(-0.5 * (x * scale) ** 2)
        
        elif sigmoid == "linear":
            scale = 1 - value_at_1
            scaled_x = x * scale
            return jnp.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        
        else:
            raise ValueError(
                f"Unknown sigmoid type {sigmoid!r}."
            )

    def tolerance(
            self,
            x:jnp.ndarray,
            bounds: tuple[float, float] = (0.0, 0.0),
            margin: float = 0.0,
            sigmoid: str = "gaussian",
            value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN
    ) -> jnp.ndarray:
        lower, upper = bounds
        if lower > upper:
            raise ValueError("Lower bound must be <= upper bound.")
        if margin < 0:
            raise ValueError("`margin` must be non-negative.")
        in_bounds = jnp.logical_and(lower <= x, x<= upper)
        if margin == 0:
            value = jnp.where(in_bounds, 1.0, 0.0)
        else:
            d = jnp.where(x < lower, lower - x, x - upper) / margin
            value = jnp.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))

        return value
    
    def _get_stand_reward(
                        self,
                        state: mjx.Data,
                        ) -> jax.Array:
        torso_height = state.xpos[self.mj_model.body("torso").id, -1]
        # torso_height = self._get_torso_height(state)
        standing = self.tolerance(
            torso_height,
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
            )
        
        torso_upright = state.xmat[self.mj_model.body("torso").id, 2, 2]
        # torso_upright = self._get_torso_deviation_from_upright(state)
        upright = (1 + torso_upright) / 2
        # upright = -torso_upright
        stand_reward = (3 * standing + upright) / 4

        return 1*stand_reward
    
    def _get_move_reward(
            self,
            state: mjx.Data
    ) -> jax.Array:
        stand_reward = self._get_stand_reward(state)

        horizontal_velocity = self._get_torso_velocity(state)
        
        move_reward = self.tolerance(horizontal_velocity, 
                bounds=(WALK_SPEED, float("inf")),
                margin = WALK_SPEED / 2,
                value_at_margin=0.5,
                sigmoid="linear"
        )
        return -1*stand_reward * ( 5* move_reward + 1) / 6


