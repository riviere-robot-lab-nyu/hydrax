
import jax
import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)




import mujoco
from hydrax.algs import MPPI, MPPI_bangbang, MPPI_ctrl_chunk
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.satellite_w_arm import Sat_w_arm



task = Sat_w_arm()


# MUST be enabled for robotics with high mass ratios

# ctrl = MPPI_ctrl_chunk(
#     task,
#     num_samples=1024,
#     plan_horizon=0.5,
#     noise_level=20.,
#     num_knots = 10,
#     temperature=0.1,
#     iterations=1,
#     chunk_index=18
# )

ctrl = MPPI(
    task,
    num_samples=4096,
    plan_horizon=0.5,
    noise_level=jnp.array([30.0, 30.0, 30.0, 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    num_knots = 10,
    temperature=0.05,
    iterations=1,
)

mj_model = task.mj_model
mj_model.opt.timestep=0.005
mj_model.opt.iterations=50
mj_data = mujoco.MjData(mj_model)
# CORRECT (Remove '.opt')
# Disable collisions for ALL geometries by zeroing out their collision masks
# The '[:]' syntax is crucial: it modifies the array in place.
mj_model.geom_conaffinity[:] = 0
mj_model.geom_contype[:] = 0
print(mj_model.body_mass)
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    max_traces=1,
    record_video=False,
    # chunking=True
)
