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


LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}
BASE_ACCELERATION_RATE = 10.0
BASE_DECELERATION_RATE = 10.0
BASE_TOP_SPEED_LEVELS = [2.0, 4.0, 6.0]
BASE_SPEED_LEVELS = [
    {"linear": 0.1, "angular": 30},
    {"linear": 0.25, "angular": 60},
    {"linear": 0.4, "angular": 90},
]
MIN_VELOCITY_THRESHOLD = 0.02
BASE_FORWARD = 1 << 0
BASE_BACKWARD = 1 << 1
BASE_ROTATE_LEFT = 1 << 2
BASE_ROTATE_RIGHT = 1 << 3


class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        self.target_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        self.zero_pos = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.roll_baseline = None
        self.wrist_roll_anchor = self.target_positions["wrist_roll"]

    def set_zero_target(self):
        print(f"[{self.prefix}] Targeting zero position: {self.zero_pos}")
        self.target_positions = self.zero_pos.copy()
        self.roll_baseline = None
        self.wrist_roll_anchor = self.target_positions["wrist_roll"]

    def set_recorded_target(self, motor_states):
        self.target_positions = {
            joint: motor_states[f"{motor}.pos"] for joint, motor in self.joint_map.items()
        }
        self.roll_baseline = None
        self.wrist_roll_anchor = self.target_positions["wrist_roll"]

    def set_end_effector_target(self, pose, gripper_state):
        """Set arm target from a generic end-effector pose tuple."""
        x, y, z, roll_, pitch_, _yaw = pose
        pitch = -pitch_ * 60 + 10
        current_x = 0.1629 + x
        current_y = 0.1131 + z

        if self.roll_baseline is None:
            self.roll_baseline = roll_
            self.wrist_roll_anchor = self.target_positions["wrist_roll"]

        roll = self.wrist_roll_anchor + (roll_ - self.roll_baseline) * 45
        self.target_positions["shoulder_pan"] = y * 250.0

        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as exc:
            print(f"[{self.prefix}] IK failed: {exc}")

        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch
        )
        self.target_positions["wrist_roll"] = roll
        self.target_positions["gripper"] = gripper_state

    def p_control_action(self, obs):
        action = {}
        for joint, target in self.target_positions.items():
            current = obs[f"{self.prefix}_arm_{joint}.pos"]
            action[f"{self.joint_map[joint]}.pos"] = current + self.kp * (target - current)
        return action


class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def set_zero_target(self):
        print(f"[HEAD] Targeting zero position: {self.zero_pos}")
        self.target_positions = self.zero_pos.copy()

    def set_recorded_target(self, motor_states):
        self.target_positions = {
            motor: motor_states[f"{motor}.pos"] for motor in self.target_positions
        }

    def increment_target(self, head_motor_1_delta=0.0, head_motor_2_delta=0.0):
        self.target_positions["head_motor_1"] += head_motor_1_delta
        self.target_positions["head_motor_2"] += head_motor_2_delta
        if head_motor_1_delta:
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if head_motor_2_delta:
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, obs):
        action = {}
        for motor, target in self.target_positions.items():
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + self.kp * (target - current)
        return action


class SmoothBaseController:
    """Smooth base velocity generator from requested base directions."""

    def __init__(self):
        self.speed_levels = BASE_SPEED_LEVELS
        self.speed_index = 0
        self.base_target = {"x.vel": 0.0, "theta.vel": 0.0}
        self.current_speed = 0.0
        self.last_time = time.time()
        self.last_direction = {"x.vel": 0.0, "theta.vel": 0.0}
        self.is_moving = False
        self._lock = threading.RLock()

    def reset(self):
        with self._lock:
            self.base_target = {"x.vel": 0.0, "theta.vel": 0.0}
            self.current_speed = 0.0
            self.last_time = time.time()
            self.last_direction = {"x.vel": 0.0, "theta.vel": 0.0}
            self.is_moving = False

    def set_target(self, base_action):
        with self._lock:
            self.base_target = {
                "x.vel": base_action.get("x.vel", 0.0),
                "theta.vel": base_action.get("theta.vel", 0.0),
            }

    def get_target(self):
        with self._lock:
            return self.base_target.copy()

    def adjust_speed_level(self, delta):
        with self._lock:
            previous = self.speed_index
            self.speed_index = max(0, min(self.speed_index + delta, len(self.speed_levels) - 1))
            changed = self.speed_index != previous
            if changed:
                direction = "increased" if delta > 0 else "decreased"
                print(
                    f"[BASE] Speed level {direction} to {self.speed_index + 1}/{len(self.speed_levels)} "
                    f"(top multiplier {self.max_speed_multiplier():.1f}x)"
                )
            return changed

    def max_speed_multiplier(self):
        level_index = min(self.speed_index, len(BASE_TOP_SPEED_LEVELS) - 1)
        return BASE_TOP_SPEED_LEVELS[level_index]

    def update(self, directions):
        with self._lock:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            max_speed_multiplier = self.max_speed_multiplier()
            self.current_speed = min(self.current_speed, max_speed_multiplier)

            is_accelerating = directions != 0
            base_action = {"x.vel": 0.0, "theta.vel": 0.0}

            if is_accelerating:
                if not self.is_moving:
                    self.is_moving = True
                    print("[BASE] Starting acceleration")

                speed_setting = self.speed_levels[self.speed_index]
                if directions & BASE_FORWARD:
                    base_action["x.vel"] += speed_setting["linear"]
                if directions & BASE_BACKWARD:
                    base_action["x.vel"] -= speed_setting["linear"]
                if directions & BASE_ROTATE_LEFT:
                    base_action["theta.vel"] += speed_setting["angular"]
                if directions & BASE_ROTATE_RIGHT:
                    base_action["theta.vel"] -= speed_setting["angular"]

                self.last_direction = base_action.copy()
                self.current_speed = min(self.current_speed + BASE_ACCELERATION_RATE * dt, max_speed_multiplier)
            else:
                if self.is_moving:
                    self.is_moving = False
                    print("[BASE] Starting deceleration")
                if self.current_speed > 0.01 and self.last_direction:
                    base_action = self.last_direction.copy()
                self.current_speed = max(self.current_speed - BASE_DECELERATION_RATE * dt, 0.0)

            for key in base_action:
                original_value = base_action[key]
                base_action[key] *= self.current_speed
                if self.current_speed > 0.01 and abs(base_action[key]) < MIN_VELOCITY_THRESHOLD:
                    base_action[key] = MIN_VELOCITY_THRESHOLD if original_value > 0 else -MIN_VELOCITY_THRESHOLD

            self.base_target = base_action
            if is_accelerating:
                print(f"[BASE] ACCEL: Speed={self.current_speed:.2f}/{max_speed_multiplier:.1f}, Action={base_action}")
            return self.base_target.copy()


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
