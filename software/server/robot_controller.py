"""
XLeRobot target-state controller.

The robot loop owns hardware lifecycle and continuously drives the robot toward
the current target state. Input devices should update targets through
RobotController public methods.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from lerobot.model.SO101Robot import SO101Kinematics
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

from arm_controller import LEFT_JOINT_MAP, RIGHT_JOINT_MAP, SimpleTeleopArm
from base_controller import SmoothBaseController
from head_controller import SimpleHeadControl


class RobotController:
    """Thread-safe target state and robot hardware control loop."""

    def __init__(self, config_path=None, fps=30, log_rerun=False):
        self.config_path = config_path or Path(__file__).resolve().parents[1] / "config" / "xlerobot.json"
        self.robot_config = XLerobot2WheelsConfig.from_json(self.config_path)
        self.robot = XLerobot2Wheels(self.robot_config)
        self.fps = fps
        self.log_rerun = log_rerun
        self.return_position_config = _resolve_initial_motor_position_file(self.config_path)
        self.return_motor_states = None
        self.left_arm = None
        self.right_arm = None
        self.head = None
        self.base_controller = SmoothBaseController()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._control_thread = None
        self._control_thread_error = None
        self.sleeping = False
        self._quit_requested = False

    def connect(self):
        self.robot.connect()
        print("[MAIN] Successfully connected to robot")
        if not self.robot.is_calibrated:
            raise RuntimeError("Robot requires calibration before teleop.")

        print("[MAIN] Robot is calibrated and ready to use.")
        self.return_motor_states = _load_motor_states(self.return_position_config)
        print(f"[MAIN] Loaded return motor states from {self.return_position_config}")

        self._initialize_targets_from_observation()
        self.sleeping = False

    def cleanup_session(self, return_to_start=True):
        if self.robot is None or not self.robot.is_connected:
            return

        control_loop_error = None
        try:
            self.stop_control_loop()
        except Exception as exc:
            control_loop_error = exc

        try:
            if not self.sleeping:
                self.robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})
            if not self.sleeping and return_to_start and self.return_motor_states is not None:
                self.return_to_recorded_position(
                    self.return_motor_states,
                    duration_s=2.0,
                )
        finally:
            self.robot.disconnect()

        if control_loop_error is not None:
            raise control_loop_error

    def sleep(self, return_to_start=True):
        if self.sleeping:
            return
        self.sleeping = True
        self.stop_control_loop()
        self.robot.stop_base()
        if return_to_start and self.return_motor_states is not None:
            self.return_to_recorded_position(
                self.return_motor_states,
                duration_s=2.0,
            )
        self.robot.bus1.disable_torque()
        self.robot.bus2.disable_torque()
        print("[ROBOT] Sleeping. Base stopped and motor torque disabled.")

    def wake(self):
        if not self.sleeping:
            return
        self.robot.configure()
        self._initialize_targets_from_observation()
        self.sleeping = False
        self.start_control_loop()
        print("[ROBOT] Awake. Motor torque enabled.")

    def start_control_loop(self):
        self._require_targets_initialized()
        if self._control_thread is not None and self._control_thread.is_alive():
            return

        self._stop_event.clear()
        self._control_thread_error = None
        self._control_thread = threading.Thread(
            target=self._run_control_loop,
            daemon=True,
        )
        self._control_thread.start()

    def stop_control_loop(self, timeout=2.0):
        self._stop_event.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=timeout)
        if self._control_thread_error is not None:
            error = self._control_thread_error
            self._control_thread_error = None
            raise error

    def is_control_loop_running(self):
        return self._control_thread is not None and self._control_thread.is_alive()

    def set_quit_status(self, should_quit=True):
        with self._lock:
            self._quit_requested = should_quit

    def get_quit_status(self):
        with self._lock:
            return self._quit_requested

    @property
    def config(self):
        return self.robot.config

    def set_arm_end_effector_target(self, prefix, pose, gripper_state):
        with self._lock:
            self._arm(prefix).set_end_effector_target(pose, gripper_state)

    def set_arm_joint_targets(self, prefix, positions):
        with self._lock:
            arm = self._arm(prefix)
            arm.target_positions.update(
                {joint: value for joint, value in positions.items() if joint in arm.target_positions}
            )

    def set_head_targets(self, positions):
        with self._lock:
            self.head.target_positions.update(
                {motor: value for motor, value in positions.items() if motor in self.head.target_positions}
            )

    def increment_head_target(self, head_motor_1_delta=0.0, head_motor_2_delta=0.0):
        with self._lock:
            self.head.increment_target(head_motor_1_delta, head_motor_2_delta)

    def set_base_velocity_target(self, base_action):
        self.base_controller.set_target(base_action)

    def set_target_state(self, left_arm=None, right_arm=None, head=None, base=None):
        if left_arm is not None:
            self.set_arm_joint_targets("left", left_arm)
        if right_arm is not None:
            self.set_arm_joint_targets("right", right_arm)
        if head is not None:
            self.set_head_targets(head)
        if base is not None:
            self.set_base_velocity_target(base)

    def get_target_state(self):
        with self._lock:
            return {
                "left_arm": self.left_arm.target_positions.copy(),
                "right_arm": self.right_arm.target_positions.copy(),
                "head": self.head.target_positions.copy(),
                "base": self.base_controller.get_target(),
            }

    def reset_targets_to_zero(self):
        with self._lock:
            print("[ROBOT] Reset target state to zero position.")
            self.left_arm.set_zero_target()
            self.right_arm.set_zero_target()
            self.head.set_zero_target()
            self.base_controller.reset()

    def set_recorded_targets(self, motor_states):
        with self._lock:
            self.left_arm.set_recorded_target(motor_states)
            self.right_arm.set_recorded_target(motor_states)
            self.head.set_recorded_target(motor_states)
            self.base_controller.reset()

    def _run_control_loop(self):
        try:
            while not self._stop_event.is_set():
                obs = self.robot.get_observation()
                with self._lock:
                    action = {
                        **self.left_arm.p_control_action(obs),
                        **self.right_arm.p_control_action(obs),
                        **self.head.p_control_action(obs),
                        **self.base_controller.get_target(),
                    }
                self.robot.send_action(action)
                if self.log_rerun:
                    log_rerun_data(obs, action)
                precise_sleep(1.0 / self.fps)
        except Exception as exc:
            self._control_thread_error = exc
            self._stop_event.set()

    def return_to_recorded_position(self, motor_states, duration_s=2.0):
        self._require_targets_initialized()
        print("[ROBOT] Returning arms and head to recorded motor positions before quit.")
        self.set_recorded_targets(motor_states)
        deadline = time.time() + duration_s
        while time.time() < deadline:
            obs = self.robot.get_observation()
            with self._lock:
                action = {
                    **self.left_arm.p_control_action(obs),
                    **self.right_arm.p_control_action(obs),
                    **self.head.p_control_action(obs),
                    "x.vel": 0.0,
                    "theta.vel": 0.0,
                }
            self.robot.send_action(action)
            if self.log_rerun:
                log_rerun_data(obs, action)
            precise_sleep(1.0 / self.fps)

    def _arm(self, prefix):
        if prefix == "left":
            return self.left_arm
        if prefix == "right":
            return self.right_arm
        raise ValueError(f"Unknown arm prefix: {prefix}")

    def _require_targets_initialized(self):
        if self.left_arm is None or self.right_arm is None or self.head is None:
            raise RuntimeError("RobotController is not connected.")

    def _initialize_targets_from_observation(self):
        initial_obs = self.robot.get_observation()
        with self._lock:
            self.left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, initial_obs, SO101Kinematics(), prefix="left")
            self.right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, initial_obs, SO101Kinematics(), prefix="right")
            self.head = SimpleHeadControl(initial_obs)
            self.base_controller.reset()


def _load_motor_states(fpath):
    with open(fpath) as f:
        data = json.load(f)

    if "motor-states" not in data:
        raise ValueError(f"Missing 'motor-states' in {fpath}")
    return data["motor-states"]


def _resolve_initial_motor_position_file(config_path):
    with open(config_path) as f:
        data = json.load(f)

    fpath = Path(data["initial-motor-position-file"]).expanduser()
    if not fpath.is_absolute():
        fpath = config_path.parent / fpath
    return fpath
