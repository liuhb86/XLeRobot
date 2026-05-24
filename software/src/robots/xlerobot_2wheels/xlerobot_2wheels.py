#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from itertools import chain
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot_2wheels import XLerobot2WheelsConfig

logger = logging.getLogger(__name__)


class XLerobot2Wheels(Robot):
    """
    The robot includes a two-wheel differential drive mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = XLerobot2WheelsConfig
    name = "xlerobot_2wheels"

    def __init__(self, config: XLerobot2WheelsConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        # Define three speed levels and a current index
        self.speed_levels = [
            {"linear": 0.1, "angular": 30},  # slow
            {"linear": 0.2, "angular": 60},  # medium
            {"linear": 0.3, "angular": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        bus1_motor_names = {
            "left_arm_shoulder_pan",
            "left_arm_shoulder_lift",
            "left_arm_elbow_flex",
            "left_arm_wrist_flex",
            "left_arm_wrist_roll",
            "left_arm_gripper",
            "head_motor_1",
            "head_motor_2",
        }
        calibration1 = {name: cal for name, cal in self.calibration.items() if name in bus1_motor_names}
        
        self.bus1 = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                # left arm
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # head
                "head_motor_1": Motor(7, "sts3215", norm_mode_body),
                "head_motor_2": Motor(8, "sts3215", norm_mode_body),
            },
            calibration= calibration1,
        )
        bus2_motor_names = {
            "right_arm_shoulder_pan",
            "right_arm_shoulder_lift",
            "right_arm_elbow_flex",
            "right_arm_wrist_flex",
            "right_arm_wrist_roll",
            "right_arm_gripper",
            "base_left_wheel",
            "base_right_wheel",
        }
        calibration2 = {name: cal for name, cal in self.calibration.items() if name in bus2_motor_names}
        self.bus2= FeetechMotorsBus(
            port=self.config.port2,
            motors={
                # right arm
                "right_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # base - only 2 wheels for differential drive
                "base_left_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=calibration2,
        )
        self.left_arm_motors = [motor for motor in self.bus1.motors if motor.startswith("left_arm")]
        self.right_arm_motors = [motor for motor in self.bus2.motors if motor.startswith("right_arm")]
        self.head_motors = [motor for motor in self.bus1.motors if motor.startswith("head")]
        self.base_motors = [motor for motor in self.bus2.motors if motor.startswith("base")]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "left_arm_shoulder_pan.pos",
                "left_arm_shoulder_lift.pos",
                "left_arm_elbow_flex.pos",
                "left_arm_wrist_flex.pos",
                "left_arm_wrist_roll.pos",
                "left_arm_gripper.pos",
                "right_arm_shoulder_pan.pos",
                "right_arm_shoulder_lift.pos",
                "right_arm_elbow_flex.pos",
                "right_arm_wrist_flex.pos",
                "right_arm_wrist_roll.pos",
                "right_arm_gripper.pos",
                "head_motor_1.pos",
                "head_motor_2.pos",
                "x.vel",
                "theta.vel",  # Only x and theta for differential drive
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus1.is_connected and self.bus2.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus1.connect()
        self.bus2.connect()
        
        logger.info("Checking motor calibration")

        if not self.bus1.is_calibrated or not self.bus2.is_calibrated:
            self.bus1.disconnect()
            self.bus2.disconnect()
            raise RuntimeError(
                f"{self} is not calibrated. The configured calibration does not match the calibration stored "
                "on the motors. "
                "Run manual calibration before connecting."
            )

        logger.info("Configured calibration matches motor calibration.")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus1.is_calibrated and self.bus2.is_calibrated

    def _resolve_calibration_motors(self, motors: str | list[str] | tuple[str, ...] | None) -> list[str]:
        motor_groups = {
            "all": self.left_arm_motors + self.head_motors + self.right_arm_motors + self.base_motors,
            "left_arm_motors": self.left_arm_motors,
            "right_arm_motors": self.right_arm_motors,
            "head_motors": self.head_motors,
            "base_motors": self.base_motors,
        }
        all_motors = set(motor_groups["all"])

        requested = ["all"] if motors is None else ([motors] if isinstance(motors, str) else list(motors))
        resolved = []
        for item in requested:
            if item in motor_groups:
                candidates = motor_groups[item]
            elif item in all_motors:
                candidates = [item]
            else:
                raise ValueError(
                    f"Unknown calibration target '{item}'. Use one of {sorted(motor_groups)} "
                    f"or one of {sorted(all_motors)}."
                )

            for motor in candidates:
                if motor not in resolved:
                    resolved.append(motor)

        return resolved

    def _calibrate_bus_motors(self, bus: FeetechMotorsBus, motor_names: list[str]) -> dict[str, MotorCalibration]:
        if not motor_names:
            return {}

        bus.disable_torque(motor_names)
        position_motors = [motor for motor in motor_names if "wheel" not in motor]
        full_turn_motors = [motor for motor in motor_names if "wheel" in motor]

        for name in position_motors:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        homing_offsets = dict.fromkeys(full_turn_motors, 0)
        if position_motors:
            input(
                f"Move {', '.join(position_motors)} to the middle of their range of motion and press ENTER...."
            )
            homing_offsets.update(bus.set_half_turn_homings(position_motors))

        range_mins = dict.fromkeys(full_turn_motors, 0)
        range_maxes = dict.fromkeys(full_turn_motors, 4095)
        if position_motors:
            print(
                f"Move {', '.join(position_motors)} sequentially through their entire ranges of motion.\n"
                "Recording positions. Press ENTER to stop..."
            )
            recorded_mins, recorded_maxes = bus.record_ranges_of_motion(position_motors)
            range_mins.update(recorded_mins)
            range_maxes.update(recorded_maxes)

        calibration = {}
        for name in motor_names:
            motor = bus.motors[name]
            calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        bus.write_calibration(calibration, cache=False)
        bus.calibration = {**bus.calibration, **calibration}
        return calibration

    def calibrate(self, motors: str | list[str] | tuple[str, ...] | None = None) -> None:
        """Calibrate all motors, a named motor group, or one or more individual motors."""
        motor_names = self._resolve_calibration_motors(motors)
        logger.info(f"\nRunning calibration of {self}: {motor_names}")

        bus1_motors = [motor for motor in motor_names if motor in self.bus1.motors]
        bus2_motors = [motor for motor in motor_names if motor in self.bus2.motors]

        opened_bus1 = False
        opened_bus2 = False
        try:
            if bus1_motors and not self.bus1.is_connected:
                self.bus1.connect()
                opened_bus1 = True
            if bus2_motors and not self.bus2.is_connected:
                self.bus2.connect()
                opened_bus2 = True

            calibration_updates = {}
            calibration_updates.update(self._calibrate_bus_motors(self.bus1, bus1_motors))
            calibration_updates.update(self._calibrate_bus_motors(self.bus2, bus2_motors))

            self.calibration = {**self.calibration, **calibration_updates}
            self._save_calibration()
            print("Calibration saved to", self.calibration_fpath)
        finally:
            if opened_bus1:
                self.bus1.disconnect()
            if opened_bus2:
                self.bus2.disconnect()
        

    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration        
        self.bus1.disable_torque()
        self.bus2.disable_torque()
        self.bus1.configure_motors()
        self.bus2.configure_motors()
        
        for name in self.left_arm_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus1.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)
            if name == "left_arm_gripper":
                self.bus1.write("Max_Torque_Limit", name, 500)  # 50% of max torque to avoid burnout
                self.bus1.write("Protection_Current", name, 250)  # 50% of max current to avoid burnout
                self.bus1.write("Overload_Torque", name, 25)  # 25% torque when overloaded
        
        for name in self.head_motors:
            self.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus1.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus1.write("I_Coefficient", name, 0)
            self.bus1.write("D_Coefficient", name, 43)
        
        for name in self.right_arm_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus2.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus2.write("I_Coefficient", name, 0)
            self.bus2.write("D_Coefficient", name, 43)
            if name == "right_arm_gripper":
                self.bus2.write("Max_Torque_Limit", name, 500)  # 50% of max torque to avoid burnout
                self.bus2.write("Protection_Current", name, 250)  # 50% of max current to avoid burnout
                self.bus2.write("Overload_Torque", name, 25)  # 25% torque when overloaded
        
        for name in self.base_motors:
            self.bus2.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        
        
        self.bus1.enable_torque()
        self.bus2.enable_torque()
        

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        theta: float,
        wheel_radius: float = None,
        wheelbase: float = None,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands for differential drive.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          wheelbase  : Distance between left and right wheels (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_right_wheel": value}.

        Notes:
          - Differential drive kinematics: only x and theta are controllable
          - y velocity is ignored (differential drive cannot move sideways)
        """
        if wheel_radius is None:
            wheel_radius = self.config.wheel_radius
        if wheelbase is None:
            wheelbase = self.config.wheelbase
            
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        
        # Differential drive kinematics
        # Left wheel speed = (v - ω*L/2) / r
        # Right wheel speed = (v + ω*L/2) / r
        left_wheel_speed = (x - theta_rad * wheelbase / 2) / wheel_radius
        right_wheel_speed = (x + theta_rad * wheelbase / 2) / wheel_radius
        
        # Convert wheel speeds from rad/s to deg/s.
        left_wheel_degps = left_wheel_speed * (180.0 / np.pi)
        right_wheel_degps = right_wheel_speed * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in [left_wheel_degps, right_wheel_degps]]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            left_wheel_degps = left_wheel_degps * scale
            right_wheel_degps = right_wheel_degps * scale

        # Convert each wheel's angular speed (deg/s) to a raw integer.
        # The left drive motor is mounted in the opposite orientation from the right one.
        left_wheel_raw = self._degps_to_raw(-left_wheel_degps)
        right_wheel_raw = self._degps_to_raw(right_wheel_degps)

        return {
            "base_left_wheel": left_wheel_raw,
            "base_right_wheel": right_wheel_raw,
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = None,
        wheelbase: float = None,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities for differential drive.

        Parameters:
          left_wheel_speed  : Raw command for left wheel.
          right_wheel_speed : Raw command for right wheel.
          wheel_radius      : Radius of each wheel (meters).
          wheelbase         : Distance between left and right wheels (meters).

        Returns:
          A dict (x.vel, theta.vel) in m/s and deg/s
        """
        if wheel_radius is None:
            wheel_radius = self.config.wheel_radius
        if wheelbase is None:
            wheelbase = self.config.wheelbase

        # Convert each raw command back to an angular speed in deg/s.
        # Mirror the left-wheel sign flip used when sending commands.
        left_degps = -self._raw_to_degps(left_wheel_speed)
        right_degps = self._raw_to_degps(right_wheel_speed)

        # Convert from deg/s to rad/s.
        left_radps = left_degps * (np.pi / 180.0)
        right_radps = right_degps * (np.pi / 180.0)
        
        # Compute each wheel's linear speed (m/s) from its angular speed.
        left_linear_speed = left_radps * wheel_radius
        right_linear_speed = right_radps * wheel_radius

        # Differential drive inverse kinematics
        # v = (v_left + v_right) / 2
        # ω = (v_right - v_left) / L
        x_vel = (left_linear_speed + right_linear_speed) / 2
        theta_rad = (right_linear_speed - left_linear_speed) / wheelbase
        theta_vel = theta_rad * (180.0 / np.pi)
        
        return {
            "x.vel": x_vel,
            "theta.vel": theta_vel,
        }  # m/s and deg/s
    
    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        linear_speed = speed_setting["linear"]  # e.g. 0.1, 0.2, or 0.3
        angular_speed = speed_setting["angular"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += linear_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= linear_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += angular_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= angular_speed
            
        return {
            "x.vel": x_cmd, 
            "theta.vel": theta_cmd,
        }

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        left_arm_pos = self.bus1.sync_read("Present_Position", self.left_arm_motors)
        right_arm_pos = self.bus2.sync_read("Present_Position", self.right_arm_motors)
        head_pos = self.bus1.sync_read("Present_Position", self.head_motors)
        base_wheel_vel = self.bus2.sync_read("Present_Velocity", self.base_motors)
        
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_right_wheel"],
        )
        
        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_arm_pos.items()}
        head_state = {f"{k}.pos": v for k, v in head_pos.items()}
        # Combine all arm and head states
        obs_dict = {**left_arm_state, **right_arm_state, **head_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command xlerobot_2wheels to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        left_arm_pos = {k: v for k, v in action.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_arm_pos = {k: v for k, v in action.items() if k.startswith("right_arm_") and k.endswith(".pos")}
        head_pos = {k: v for k, v in action.items() if k.startswith("head_") and k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel.get("x.vel", 0.0),
            base_goal_vel.get("theta.vel", 0.0),
        )
        
        
        if self.config.max_relative_target is not None:
            # Read present positions for left arm, right arm, and head
            present_pos_left = self.bus1.sync_read("Present_Position", self.left_arm_motors)
            present_pos_right = self.bus2.sync_read("Present_Position", self.right_arm_motors)
            present_pos_head = self.bus1.sync_read("Present_Position", self.head_motors)

            # Combine all present positions
            present_pos = {**present_pos_left, **present_pos_right, **present_pos_head}

            # Ensure safe goal position for each arm and head
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in chain(left_arm_pos.items(), right_arm_pos.items(), head_pos.items())
            }
            safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            # Update the action with the safe goal positions
            left_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in left_arm_pos}
            right_arm_pos = {k: v for k, v in safe_goal_pos.items() if k in right_arm_pos}
            head_pos = {k: v for k, v in safe_goal_pos.items() if k in head_pos}
        
        left_arm_pos_raw = {k.replace(".pos", ""): v for k, v in left_arm_pos.items()}
        right_arm_pos_raw = {k.replace(".pos", ""): v for k, v in right_arm_pos.items()}
        head_pos_raw = {k.replace(".pos", ""): v for k, v in head_pos.items()}
        
        # Only sync_write if there are motors to write to
        if left_arm_pos_raw:
            self.bus1.sync_write("Goal_Position", left_arm_pos_raw)
        if right_arm_pos_raw:
            self.bus2.sync_write("Goal_Position", right_arm_pos_raw)
        if head_pos_raw:
            self.bus1.sync_write("Goal_Position", head_pos_raw)
        if base_wheel_goal_vel:
            self.bus2.sync_write("Goal_Velocity", base_wheel_goal_vel)
        return {
            **left_arm_pos,
            **right_arm_pos,
            **head_pos,
            **base_goal_vel,
        }

    def stop_base(self):
        self.bus2.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.bus1.disconnect(self.config.disable_torque_on_disconnect)
        self.bus2.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
