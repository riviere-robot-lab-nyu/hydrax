import mujoco
from hydrax.algs import MPPI, MPPI_bangbang
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.atmos_rrl import Atmos
import jax
import jax.numpy as jnp

task = Atmos()

# ctrl = MPPI_bangbang(
#     task,
#     num_samples=128,
#     noise_level=0.3,
#     plan_horizon=0.6,
#     num_knots=5,
#     temperature=0.1,
# )

ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=0.1,
    plan_horizon=0.6,
    num_knots=4,
    temperature=0.5,
    spline_type="zero",
)
mj_model = task.mj_model
mj_model.opt.timestep=0.005
mj_model.opt.iterations=50
mj_data = mujoco.MjData(mj_model)
print(mj_data.qpos)
# print(task.u_off)
# print(task.u_on)
# print(task.threshold)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=False,
    max_traces=1,
    record_video=False,
    bang_bang=False,)
    # initial_knots=0.5*jnp.ones((5,8),dtype=float)
# )

