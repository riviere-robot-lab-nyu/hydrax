from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mujoco import mjx
from typing import Any, Tuple
from functools import partial

@dataclass
class MPPIParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

class MPPI_ctrl_chunk(SamplingBasedController):
    def __init__(
            self,
            task: Task,
            chunk_index: int,
            num_samples: int,
            noise_level: float,
            temperature: float,
            num_randomizations: int = 1,
            risk_strategy: RiskStrategy = None,
            seed: int = 0,
            plan_horizon: float = 1.0,
            num_knots: int = 4,
            iterations: int = 1,
    ) -> None:
        
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.chunk_ix = chunk_index

    def init_params(
            self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIParams:
        rng = jax.random.key(seed)
        mean = (
            initial_knots
            if initial_knots is not None
            else jnp.zeros((self.num_knots, self.chunk_ix))
        )
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        _params =  SamplingParams(tk=tk, mean=mean, rng=rng)
        return MPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)
    
    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.chunk_ix,
            ),
        )
        controls = params.mean + self.noise_level * noise
        # print(controls)
        # jax.debug.print("CONTROLS: \n{}", controls)
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # jax.debug.print("COSTS: {}", costs)
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
    
    

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        arm_trajectory: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """
        # jax.debug.print("CONTROLS: {}", controls)
        def _scan_fn(
            x: mjx.Data, u_inputs: Tuple[jax.Array, jax.Array]
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            u_base, u_arm = u_inputs
            u_full = jnp.concatenate([u_base, u_arm])
            x = x.replace(ctrl=u_full)
            # jax.debug.print("HELLO STATE: {}", x.qpos)
            x = mjx.step(model, x)  # step model + compute site positions
            # jax.debug.print("STATES: {}",x.qpos)
            cost = self.dt * self.task.running_cost(x, u_full)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        
        scan_inputs = (controls, arm_trajectory)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, scan_inputs
        )

        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )
    
    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
        integral_init: jax.Array = None,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)

        arm_traj = self.task.get_arm_traj(tq)
        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = jax.vmap(
            self.eval_rollouts, 
            in_axes=(self.randomized_axes, 0, None, None, None)
        )(self.model, states, controls, knots, arm_traj)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        )


class MPPI_bangbang(SamplingBasedController):
    def __init__(
            self,
            task: Task,
            num_samples: int,
            noise_level: float,
            temperature: float,
            num_randomizations: int = 1,
            risk_strategy: RiskStrategy = None,
            seed: int = 0,
            plan_horizon: float = 1.0,
            num_knots: int = 4,
            iterations: int = 1,
    ) -> None:
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature    
    
    def init_params(
            self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIParams:
        _params = super().init_params(initial_knots, seed)
        return MPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        integral_init: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        
        def _scan_fn(
                x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)
            cost = self.dt * self.task.running_cost(x,u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        

        control_bb = jnp.where(controls >= self.task.threshold, self.task.u_on, self.task.u_off)
        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, control_bb
        )

        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis= 0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites
        )
    
    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)



        


class MPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float | jax.Array,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        # print(_params.mean)
        # jax.debug.print("params.mean init_params:{}", _params.mean)
        return MPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        # print(controls)
        # jax.debug.print("CONTROLS: \n{}", controls)
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # jax.debug.print("COSTS: {}", costs)
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
