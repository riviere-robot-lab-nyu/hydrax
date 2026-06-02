"""ROS 2 trajectory-follower node for the AtmosM3 MPPI loop.

Sits between `mppi_ros_node.py` and the real drone:
  - Subscribes to the MPPI plan (JointTrajectory).
  - At a configurable high rate, picks the right slice of the latest plan
    based on `now - plan.header.stamp` and publishes the resulting command.
  - Subscribes to robot state purely for staleness watchdogs (it does not
    feed state back to the planner — the planner reads state directly).

Safety layer (the whole point of this node):
  - Plan-staleness watchdog: no new plan within `plan_timeout_s` → safe stop.
  - State-staleness watchdog: same for Odometry + JointState inputs.
  - Hard saturation on every output channel.
  - Slew-rate limit on base velocity to absorb plan-to-plan setpoint jumps.

Topic / joint names are placeholders (replace with the real wiring).

Run:
  python3 trajectory_follower.py
"""

import select
import sys
import termios
import threading
import time as _time
import tty

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import trossen_arm
from scipy.interpolate import PchipInterpolator

# ── Topic placeholders ──────────────────────────────────────────────────
# Inbound
PLAN_TOPIC       = "/mppi/plan"
#BASE_STATE_TOPIC = "/TODO/base/odom"
#ARM_STATE_TOPIC  = "/TODO/arm/joint_states"
# Outbound (to drone / arm vendor controllers)
#BASE_CMD_TOPIC    = "/TODO/base/cmd_vel"
#ARM_CMD_TOPIC     = "/TODO/arm/cmd"
#GRIPPER_CMD_TOPIC = "/TODO/gripper/cmd"

ARM_JOINT_NAMES = [
    "TODO_joint_0", "TODO_joint_1", "TODO_joint_2",
    "TODO_joint_3", "TODO_joint_4", "TODO_joint_5",
]

# AtmosM3 ctrl layout:
#   0..2  base velocity (vx_body, vy_body, wz_body)
#   3..8  arm joint commands
#   9     gripper command (0=closed, 1=open)
#   10    dead channel (ignored)
NU = 11
#NOTE THAT MPPI DOES BODY VELOCITY
# FROM TROSSEN_CLIENT_ROS.py
ODOM_LIN_VEL_START = 10
ODOM_ANG_VEL_START = 13
LOCAL_POS_HEADING_INDEX = 14
BASE_VEL_LIN_X_INDEX = 3
BASE_VEL_LIN_Y_INDEX = 4
BASE_VEL_YAW_RATE_INDEX = 15



class MPPIDriver(Node):
    def __init__(self):
        super().__init__("trajectory_follower")

        # Output rate
        self.declare_parameter("publish_rate_hz", 50.0)
        # Watchdog timeouts
        self.declare_parameter("plan_timeout_s", 0.2)
        self.declare_parameter("state_timeout_s", 0.2)
        # Slew limits on base velocity command
        self.declare_parameter("max_linear_accel", 2.0)    # m/s^2
        self.declare_parameter("max_angular_accel", 4.0)   # rad/s^2
        # Hard saturation (defense in depth on top of planner u_min/u_max)
        self.declare_parameter("max_linear_vel", 0.3)      # m/s
        self.declare_parameter("max_angular_vel", 0.4)     # rad/s
        # Arm joint limits (placeholder — set per-joint values for real robot)
        self.declare_parameter("max_arm_pos", 3.14)
        self.declare_parameter("min_arm_pos", -3.14)
        # "autonomous" actually drives the arm; "test" logs only.
        self.declare_parameter("test_mode", "autonomous")
        # Duration of the smooth ramp from current pose to first MPPI command.
        self.declare_parameter("start_move_duration_s", 3.0)

        self.publish_rate  = float(self.get_parameter("publish_rate_hz").value)
        self.plan_timeout  = float(self.get_parameter("plan_timeout_s").value)
        self.state_timeout = float(self.get_parameter("state_timeout_s").value)
        self.max_lin_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_ang_accel = float(self.get_parameter("max_angular_accel").value)
        self.max_lin_vel   = float(self.get_parameter("max_linear_vel").value)
        self.max_ang_vel   = float(self.get_parameter("max_angular_vel").value)
        self.max_arm       = float(self.get_parameter("max_arm_pos").value)
        self.min_arm       = float(self.get_parameter("min_arm_pos").value)
        self.test_mode     = str(self.get_parameter("test_mode").value)
        self.start_move_duration = float(self.get_parameter("start_move_duration_s").value)
        self.dt = 1.0 / self.publish_rate

        # ── Shared state, lock-protected ──────────────────────────────────
        self._lock = threading.Lock()
        self.plan_times    = None        # (N,) seconds-from-state-stamp
        self.plan_ctrls    = None        # (N, NU)
        self.plan_state_t  = None        # ros sec (the planner's input stamp)
        self.last_plan_recv = None       # ros sec
        self.last_base_recv = None
        self.last_arm_recv  = None
        # Slew + hold state
        self.prev_base_cmd    = np.zeros(3, dtype=np.float64)
        self.prev_arm_cmd     = np.zeros(6, dtype=np.float64)
        # Gripper convention here is VLA (0=open, 1=close), matching
        # _vla_gripper_to_position and execute_arm_action_abs.
        self.prev_gripper_cmd = 0.0
        # First-arm-step ramp: gates arm publishing until move_to_start_position
        # finishes in a background thread.
        self.is_first_arm_step = True
        self.arm_start_thread = None
        # Serializes all access to self.driver — the background ramp thread,
        # the tick thread's state read+command, and shutdown can otherwise race.
        self._driver_lock = threading.Lock()
        # Counters
        self._n_plans      = 0
        self._n_safe_stops = 0
        self._n_ticks      = 0

        self.last_odom = None
        self.last_local_pos = None

        # ── ROS plumbing ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointTrajectory, PLAN_TOPIC, self._plan_cb, sensor_qos,
        )

        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_odometry', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, '/px4/bridge/vehicle_local_position_v1', self.local_pos_cb, qos_profile_sensor_data)


        self.base_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/vla_base/traj_setpoint',
            qos_profile_sensor_data
        )

        self.state_pub = self.create_publisher(
            Float64MultiArray,
            '/follower/joint_state',
            sensor_qos
        )
        # ── High-rate tick thread ─────────────────────────────────────────
        self._stop = threading.Event()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, name="follower_tick", daemon=True,
        )
        self._tick_thread.start()

        # ── Keyboard e-stop ('q') ─────────────────────────────────────────
        # Background thread reads stdin in cbreak mode so a single 'q' (no
        # Enter) triggers rclpy.shutdown(), letting main()'s finally block run
        # the normal cleanup path (move-to-sleep + driver.cleanup).
        self._kb_thread = threading.Thread(
            target=self._keyboard_listener, name="follower_kb", daemon=True,
        )
        self._kb_thread.start()

        # Stats log every second
        self.create_timer(1.0, self._log_stats)
        # State publish, independent of plan validity — mppi_vla_node needs
        # /follower/joint_state to start producing plans at all, so this must
        # NOT be gated on having a fresh plan or the ramp finishing.
        self.create_timer(self.dt, self._publish_state)

        self.get_logger().info(
            f"Trajectory follower ready. publish={self.publish_rate:.0f}Hz, "
            f"plan_timeout={self.plan_timeout*1000:.0f}ms, "
            f"state_timeout={self.state_timeout*1000:.0f}ms"
        )

        # Connect to trossen arm
        print("Initializing the drivers...")
        self.driver = trossen_arm.TrossenArmDriver()

        print("Configuring the drivers...")
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            "192.168.55.3",
            False
        )
        print("Finished configuring the drivers")

        print("setting arm control mode")
        self.driver.set_arm_modes(trossen_arm.Mode.position)
        self.driver.set_gripper_mode(trossen_arm.Mode.position)
        # Raise the gripper torque cap so blocked-close commands squeeze harder.
        # Joint 6 is the gripper; in position mode it pulls up to effort_max when stalled.
        limits = self.driver.get_joint_limits()
        limits[6].position_min = -0.006        # current rest is -0.00512; this + 0.004 tolerance gives envelope down to -0.010
        limits[6].position_max = 0.006 + 0.044
        limits[6].effort_max = self.GRIPPER_EFFORT_MAX
        self.driver.set_joint_limits(limits)
        print("Finished setting arm control mode")

        self.sleep_positions = np.array(self.driver.get_all_positions())
    GRIPPER_OPEN     = 0.045   # m — fully open (matches widened position_max envelope)
    GRIPPER_CLOSE    = -0.005  # m — fully closed (matches widened position_min envelope)
    GRIPPER_EFFORT_MAX = 200.0  # N — tune; factory cap is 100, teleop uses 200

    GRIPPER_BINARY_THRESHOLD = 0.4

    def _vla_gripper_to_position(self, vla_value: float) -> float:
        """Map VLA gripper output to driver position (m). VLA: 0=open, 1=close. Binary."""
        return self.GRIPPER_CLOSE if vla_value >= self.GRIPPER_BINARY_THRESHOLD else self.GRIPPER_OPEN

    def execute_arm_action(self, current_js: np.ndarray, rel_action: np.ndarray):
        """Execute action on the arm.
        action layout: [0:6] relative joint deltas, [6] gripper position (VLA: 0=open, 1=close)
        Arm: position mode. Gripper: position mode with raised effort_max for hard gripping.
        """
        if self.test_mode == "test":
            self.get_logger().info(f"TEST MODE: Would execute action: {current_js + rel_action[:6]}")
            return
        if self.test_mode == "autonomous":
            try:
                arm_goal = current_js + rel_action[:6]         # relative -> absolute
                gripper_goal = self._vla_gripper_to_position(rel_action[6])
                # TODO: Check if the goal time self.df is good or no
                with self._driver_lock:
                    self.driver.set_arm_positions(arm_goal, self.dt, False)
                    self.driver.set_gripper_position(gripper_goal, self.dt, False)
            # Move to sleep position if huge movement occured
            except Exception as e:
                print("An error occurred: ", e)
                print("Recovering from the error...")
                self.driver.cleanup()
                self.driver.configure(
                    trossen_arm.Model.wxai_v0,
                    trossen_arm.StandardEndEffector.wxai_v0_leader,
                    '192.168.55.3',
                    True
                )
                self.driver.set_all_modes(trossen_arm.Mode.position)
                self.move_to_sleep_position()
                # self.driver.set_all_positions(self.sleep_positions)
                
        else:
            self.get_logger().error(f"Unknown mode: {self.test_mode}. No action executed.")

    def execute_arm_action_abs(self, action: np.ndarray):
        """Execute action on the arm in absolute way"""
        full_action = action.copy() 

        if self.test_mode == "test":
            self.get_logger().info(f"TEST MODE: Would execute action: {full_action}")
            return
        if self.test_mode == "autonomous":
            # TODO: seperate arm and gripper mode, set small effort to gripper 
            try:
                full_action[-1] = self._vla_gripper_to_position(full_action[-1])
                with self._driver_lock:
                    self.driver.set_all_positions(full_action, self.dt * 20, False)

                # self.driver.set_arm_positions(full_action[:-1], self.dt * 15, False)
                # self.driver.set_gripper_position(self._vla_gripper_to_position(full_action[-1]), 
                #                                 self.dt * 5, False)
                
            # Move to sleep position if huge movement occured
            except Exception as e:
                print("An error occurred: ", e)
                print("Recovering from the error...")
                self.driver.cleanup()
                self.driver.configure(
                    trossen_arm.Model.wxai_v0,
                    trossen_arm.StandardEndEffector.wxai_v0_leader,
                    '192.168.55.3',
                    True
                )
                self.driver.set_all_modes(trossen_arm.Mode.position)
                self.move_to_sleep_position()
                # self.driver.set_all_positions(self.sleep_positions)
                
        else:
            self.get_logger().error(f"Unknown mode: {self.test_mode}. No action executed.")

    def move_to_sleep_position(self):
        with self._driver_lock:
            self.driver.set_gripper_position(0.04, 2.0, False)
            self.driver.set_arm_positions(
                np.zeros(self.driver.get_num_joints() - 1),
                4.0,
                True
            )
        return
    
    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits)."""

        # joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith(".pos")]
        # current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        with self._driver_lock:
            current_pose = np.array(self.driver.get_all_positions())
        # Example stage_pose for bimanual WidowX arms.
        # Each value corresponds to a joint position (in radians) for the 14 joints:
        # [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_left_carriage_joint,
        #  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_left_carriage_joint]
        # The values below represent a "stage" pose, e.g. arms up and open, ready for task start.
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # up_pose = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = _time.time()
        end_time = start_time + timepoints[-1]

        while _time.time() < end_time:
            loop_start_time = _time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_arm_action_abs(positions)


    # ── Helpers ──────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ── Subscriber callbacks ─────────────────────────────────────────────
    def _plan_cb(self, msg: JointTrajectory):
        n = len(msg.points)
        if n == 0:
            return
        times = np.empty(n, dtype=np.float64)
        ctrls = np.empty((n, NU), dtype=np.float64)
        for i, pt in enumerate(msg.points):
            times[i] = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            ctrls[i, :] = pt.positions[:NU]
        state_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        recv_t = self._now()
        with self._lock:
            self.plan_times    = times
            self.plan_ctrls    = ctrls
            self.plan_state_t  = state_t
            self.last_plan_recv = recv_t
            self._n_plans += 1

  #  def _base_cb(self, msg: Odometry):
  #      with self._lock:
  #          self.last_base_recv = self._now()

    def odom_cb(self, msg: Float64MultiArray):
        t = self._now()
        self.last_odom = (msg, t)
        # Feeds the base-staleness watchdog in _tick — odom is the more
        # frequent of the two state streams, so use it as the heartbeat.
        with self._lock:
            self.last_base_recv = t

    def local_pos_cb(self, msg: Float64MultiArray):
        self.last_local_pos = (msg, self._now())

    def publish_base_cmd(self, base_action: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(base_action[0]), float(base_action[1]), float(base_action[2])]
        self.base_cmd_pub.publish(msg)

    def _publish_state(self):
        """Publish current arm joint state as positions(7) + velocities(7).

        Fires unconditionally on its own timer so mppi_vla_node can begin
        planning before any plan exists (otherwise mppi_driver and the planner
        deadlock waiting on each other).
        """
        try:
            with self._driver_lock:
                positions_list = list(self.driver.get_all_positions())
                velocities_list = list(self.driver.get_all_velocities())
        except Exception as e:
            self.get_logger().warn(
                f"State read failed: {e}",
                throttle_duration_sec=2.0,
            )
            return
        state_msg = Float64MultiArray()
        state_msg.data = positions_list + velocities_list
        self.state_pub.publish(state_msg)

    def _arm_cb(self, msg: JointState):
        with self._lock:
            self.last_arm_recv = self._now()

    # ── Keyboard e-stop (separate thread) ────────────────────────────────
    def _keyboard_listener(self):
        """Background thread: press 'q' (no Enter) to trigger e-stop.

        Calls rclpy.shutdown() which unblocks main()'s rclpy.spin(); the
        existing finally block then runs shutdown() (zero base, move arm to
        sleep, driver cleanup). TTY mode is restored on exit so the terminal
        isn't left in cbreak after the process dies.
        """
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError, AttributeError):
            self.get_logger().info(
                "stdin is not a TTY — keyboard e-stop disabled. "
                "Use Ctrl-C to stop instead."
            )
            return

        try:
            tty.setcbreak(fd)
            self.get_logger().info(
                "Keyboard e-stop armed: press 'q' to stop and sleep the arm."
            )
            while not self._stop.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    try:
                        key = sys.stdin.read(1)
                    except (OSError, IOError):
                        break
                    if key and key.lower() == "q":
                        self.get_logger().warn(
                            "E-STOP: 'q' pressed — shutting down."
                        )
                        # Unblock rclpy.spin() in main(); the finally block
                        # runs shutdown() once we return.
                        try:
                            rclpy.shutdown()
                        except Exception:
                            pass
                        break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    # ── High-rate loop (separate thread) ─────────────────────────────────
    def _tick_loop(self):
        period = 1.0 / self.publish_rate
        next_t = _time.monotonic()
        while not self._stop.is_set():
            self._tick()
            next_t += period
            sleep = next_t - _time.monotonic()
            if sleep > 0:
                _time.sleep(sleep)
            else:
                # Behind schedule — reset reference instead of accumulating lag.
                next_t = _time.monotonic()

    def _tick(self):
        now = self._now()
        with self._lock:
            plan_age = (
                now - self.last_plan_recv
                if self.last_plan_recv is not None else float("inf")
            )
            base_age = (
                now - self.last_base_recv
                if self.last_base_recv is not None else float("inf")
            )
            plan_times   = self.plan_times
            plan_ctrls   = self.plan_ctrls
            plan_state_t = self.plan_state_t

        self._n_ticks += 1

        # Cheap watchdogs first — these handle the "no plan yet" / "no state
        # yet" cases before we try to subtract None below. Arm watchdog is
        # disabled until the arm subscription is wired up; re-add when it is.
        if (
            plan_ctrls is None
            or plan_age > self.plan_timeout
            or base_age > self.state_timeout
        ):
            self._publish_safe_stop()
            self._n_safe_stops += 1
            
            return

        # Past here, plan_state_t and plan_times are guaranteed non-None
        # (they're set atomically with plan_ctrls under _lock).
        elapsed = now - plan_state_t
        if elapsed > plan_times[-1]:
            self._publish_safe_stop()
            self._n_safe_stops += 1
            return

        # Pick the plan slice that corresponds to *right now* in the state's
        # timeline (header.stamp == the Odometry stamp the planner used).
        idx = int(np.searchsorted(plan_times, elapsed))
        idx = min(idx, len(plan_ctrls) - 1)
        u = plan_ctrls[idx]
        self._publish_cmd(u)


    # ── Output: apply limits + publish ───────────────────────────────────
    def _publish_cmd(self, u: np.ndarray):
        self._publish_base(u[0], u[1], u[2])
        self._send_arm(u[3:9], u[9])

    def _publish_base(self, vx: float, vy: float, wz: float):
        """Hard-clamp + slew-limit + publish base velocity to /vla_base/traj_setpoint."""
        dt = self.dt
        # Note: vy is negated to match the base controller's frame convention
        # (planner-frame y is opposite the bridge's body-y).
        target_base = np.array([
            np.clip(vx,       -self.max_lin_vel, self.max_lin_vel),
            np.clip(-1.0 * vy, -self.max_lin_vel, self.max_lin_vel),
            np.clip(wz,       -self.max_ang_vel, self.max_ang_vel),
        ], dtype=np.float64)
        max_lin_step = self.max_lin_accel * dt
        max_ang_step = self.max_ang_accel * dt
        base = np.empty(3, dtype=np.float64)
        base[0] = self.prev_base_cmd[0] + np.clip(
            target_base[0] - self.prev_base_cmd[0], -max_lin_step, max_lin_step,
        )
        base[1] = self.prev_base_cmd[1] + np.clip(
            target_base[1] - self.prev_base_cmd[1], -max_lin_step, max_lin_step,
        )
        base[2] = self.prev_base_cmd[2] + np.clip(
            target_base[2] - self.prev_base_cmd[2], -max_ang_step, max_ang_step,
        )
        self.prev_base_cmd = base
        self.publish_base_cmd(base)

    def _send_arm(self, arm_cmd: np.ndarray, gripper_mppi: float):
        """Clamp arm to joint limits, invert gripper convention, dispatch to driver.

        First call kicks off a background smooth-ramp from the current pose to
        the planner's first command (move_to_start_position is blocking, so it
        can't run on the tick thread). The tick keeps publishing base in the
        meantime; arm commands resume once the ramp finishes.
        """
        arm_target = np.clip(
            np.array(arm_cmd, dtype=np.float64), self.min_arm, self.max_arm,
        )
        # MPPI plan: 0=closed, 1=open. _vla_gripper_to_position: 0=open, 1=close.
        gripper_vla = 1.0 - float(np.clip(gripper_mppi, 0.0, 1.0))

        if self.is_first_arm_step:
            if self.arm_start_thread is None:
                self.get_logger().info(
                    "First arm command — ramping to start pose over "
                    f"{self.start_move_duration:.1f}s"
                )
                goal_vla = np.concatenate([arm_target, [gripper_vla]])
                self.arm_start_thread = threading.Thread(
                    target=self._do_start_move,
                    args=(goal_vla,),
                    name="arm_start",
                    daemon=True,
                )
                self.arm_start_thread.start()
            return

        self.prev_arm_cmd = arm_target
        self.prev_gripper_cmd = gripper_vla
        self.execute_arm_action_abs(
            np.concatenate([arm_target, [gripper_vla]])
        )

    def _do_start_move(self, goal_vla: np.ndarray):
        """Background-thread smooth move to the planner's first commanded pose."""
        arm = np.array(goal_vla[:6], dtype=np.float64)
        # move_to_start_position interpolates against driver.get_all_positions(),
        # which returns gripper in meters — so convert before passing.
        gripper_m = self._vla_gripper_to_position(float(goal_vla[6]))
        goal_m = np.concatenate([arm, [gripper_m]])
        try:
            self.move_to_start_position(goal_m, duration=self.start_move_duration)
        except Exception as e:
            self.get_logger().error(f"Arm start move failed: {e}")
        finally:
            self.prev_arm_cmd = arm
            self.prev_gripper_cmd = float(goal_vla[6])
            self.is_first_arm_step = False
            self.get_logger().info("Arm start move complete; tracking plan")

    def _publish_safe_stop(self):
        """Slew base toward zero; hold arm at last commanded pose if started."""
        self._publish_base(0.0, 0.0, 0.0)
        if not self.is_first_arm_step:
            try:
                self.execute_arm_action_abs(
                    np.concatenate([self.prev_arm_cmd, [self.prev_gripper_cmd]])
                )
            except Exception as e:
                self.get_logger().error(f"Hold-arm during safe stop failed: {e}")

    def _log_stats(self):
        with self._lock:
            n_plans = self._n_plans
            n_safe  = self._n_safe_stops
            n_ticks = self._n_ticks
            plan_age = (
                self._now() - self.last_plan_recv
                if self.last_plan_recv is not None else float("inf")
            )
        self.get_logger().info(
            f"ticks={n_ticks} | plans={n_plans} | safe_stops={n_safe} | "
            f"plan_age={plan_age*1000:6.1f}ms"
        )

    def shutdown(self):
        self._stop.set()
        if self._tick_thread.is_alive():
            self._tick_thread.join(timeout=2.0)
        # Wait for start-move thread if it's still ramping the arm.
        if self.arm_start_thread is not None and self.arm_start_thread.is_alive():
            self.arm_start_thread.join(timeout=2.0)
        # Final safe stop on the base.
        try:
            self.publish_base_cmd(np.zeros(3))
        except Exception as e:
            self.get_logger().error(f"Final base zero publish failed: {e}")
        # Move arm to sleep, then clean up the trossen driver.
        try:
            self.get_logger().info("Moving arm to sleep position...")
            self.move_to_sleep_position()
        except Exception as e:
            self.get_logger().error(f"Move to sleep failed: {e}")
        try:
            self.driver.cleanup()
        except Exception as e:
            self.get_logger().error(f"Driver cleanup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPPIDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
