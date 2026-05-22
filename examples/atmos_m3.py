import mujoco
import jax.numpy as jnp
from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.atmos_m3 import AtmosM3

task = AtmosM3(
    arm_ctrl_cost=jnp.ones(6) * 2000.0,  # keep arm at zero for now
    ctrl_cost = 50 * jnp.ones(3)
   # kp_vel=20.0,                         # higher gain so robot moves within horizon
)

noise = jnp.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
ctrl = MPPI(
    task,
    #num_samples=1024*2,
    #num_samples=1024*8,
    #num_samples=512,
    num_samples=1024,
    #num_samples=1024*8,
    #noise_level=0.1,
    noise_level=noise,
    plan_horizon=0.3,
    num_knots=5,
    temperature=0.2,
    spline_type="zero",
)

mj_model = task.mj_model
mj_model.opt.timestep = 0.01
mj_data = mujoco.MjData(mj_model)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=20,
    show_traces=False,
    max_traces=0,
    record_video=False,
)
