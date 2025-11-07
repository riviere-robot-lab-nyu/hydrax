import mujoco

from hydrax.algs import MPPI, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from cart_pole_sam import CartpoleSam

task = CartpoleSam()

ctrl = MPPI(task,
            num_samples=128,
            noise_level=5.0,
            temperature=0.1,
            plan_horizon=1.0,
            spline_type="linear",
            num_knots=5)

mj_model = task.mj_model
# mj_model.opt.timestep=0.005
# mj_model.opt.iterations=50
mj_data = mujoco.MjData(mj_model)

run_interactive(ctrl,
                mj_model,
                mj_data,
                frequency=50,
                # fixed_camera_id=0,
                show_traces=False,
                max_traces=1)
