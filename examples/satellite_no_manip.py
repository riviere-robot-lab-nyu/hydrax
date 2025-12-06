import mujoco
from hydrax.algs import MPPI, MPPI_bangbang
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.satellite_no_arm import Sat_no_arm
import jax
import jax.numpy as jnp

task = Sat_no_arm()

ctrl = MPPI_bangbang(
    task,
    num_samples=4096*4,
    noise_level=0.05,
    plan_horizon=0.5,
    num_knots=50,
    temperature=0.05,
    iterations=1
)

ctrl = MPPI(
    task,
    num_samples=4096*4,
    noise_level=5,
    plan_horizon=0.5,
    num_knots=50,
    temperature=0.05,
    iterations=2
)

mj_model = task.mj_model
mj_model.opt.timestep=0.005
mj_model.opt.iterations = 50
mj_data = mujoco.MjData(mj_model)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    max_traces=1,
    record_video=False,
    # bang_bang=True,
    # initial_knots=0.45*jnp.ones((50,18),dtype=float)
)