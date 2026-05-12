from abc import ABC, abstractmethod
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def ctrl_transform(self, state: mjx.Data, ctrl: jax.Array) -> jax.Array:
        """Transform sampled controls into actuator commands before the physics step.

        Override this in a Task subclass to implement a deterministic inner control
        law so that the optimizer samples in a different space (e.g. velocity
        commands) while the physics simulation still receives forces/torques.

        The default implementation is the identity: ctrl passes through unchanged.

        Args:
            state: The current simulation state xₜ (qpos, qvel, etc.).
            ctrl:  The raw control action sampled by the optimizer.

        Returns:
            The actuator command array written to mjx.Data.ctrl before mjx.step.
        """
        return ctrl

    def ctrl_transform_with_integral(
        self, state: mjx.Data, ctrl: jax.Array, integral: jax.Array
    ) -> jax.Array:
        """Like ctrl_transform but also receives the running integral state.

        Override this (instead of ctrl_transform) when the inner control law
        needs integral feedback (e.g. a PI velocity controller).  The integral
        is carried through the rollout scan so each simulated step sees the
        correctly accumulated value.

        The default implementation ignores the integral and delegates to
        ctrl_transform, so existing subclasses require no changes.

        Args:
            state:    The current simulation state xₜ.
            ctrl:     The raw control action sampled by the optimizer.
            integral: Running integral state, shape (integral_dim,).

        Returns:
            The actuator command array written to mjx.Data.ctrl before mjx.step.
        """
        return self.ctrl_transform(state, ctrl)

    def update_integral(
        self,
        _state: mjx.Data,
        _ctrl: jax.Array,
        integral: jax.Array,
        _actual_ctrl: jax.Array = None,
    ) -> jax.Array:
        """Update the integral state after ctrl_transform but before mjx.step.

        Args:
            _state:       Current simulation state xₜ.
            _ctrl:        Raw optimizer action uₜ (optimizer space).
            integral:     Current integral state, shape (integral_dim,).
            _actual_ctrl: Actuator command produced by ctrl_transform_with_integral.
                          Use this to detect saturation for anti-windup.

        Returns:
            Updated integral state, same shape as integral.
        """
        return integral

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}
