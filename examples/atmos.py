import mujoco
from hydrax.algs import MPPI, MPPI_bangbang
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.atmos_rrl import Atmos
import jax
import jax.numpy as jnp

task = Atmos()

ctrl = MPPI_bangbang(
    task,
    num_samples=256,
    noise_level=0.2,
    plan_horizon=0.8,
    num_knots=10,
    temperature=0.1,
)

# ctrl = MPPI(
#     task,
#     num_samples=256,
#     noise_level=0.2,
#     plan_horizon=0.8,
#     num_knots=8,
#     temperature=0.1,
#     spline_type="zero",
# )

extra = """
<body name="target_body" pos="2 0.5 0." mocap="true">
    <geom type="sphere" size="0.3" rgba="0 1 0 1" conaffinity="0" contype="0"/>
</body>
"""
mj_model = task.mj_model
mj_model.opt.timestep=0.005
mj_model.opt.iterations=50
mj_data = mujoco.MjData(mj_model)
print(mj_data.qpos)
print(mj_data.qvel)
# print(task.u_off)
# print(task.u_on)
# print(task.threshold)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    # fixed_camera_id=0,
    max_traces=1,
    record_video=True,
    bang_bang=True,
    initial_knots=0.5*jnp.ones((10,8),dtype=float),
)

