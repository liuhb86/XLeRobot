# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host --robot.id=my_xlerobot_2wheels
'''

# To Run the teleop:
'''
PYTHONPATH=src python -m examples.xlerobot_2wheels.teleoperate_joycon
'''

# Base speed control instructions:
# - When holding the left stick away from center, speed will linearly accelerate to maximum speed
# - After releasing the button, speed will linearly decelerate to 0
# - You can adjust the acceleration and deceleration slopes by modifying the following parameters:
#   * BASE_ACCELERATION_RATE: acceleration slope (speed/second)
#   * BASE_DECELERATION_RATE: deceleration slope (speed/second)
#   * BASE_TOP_SPEED_LEVELS: maximum speed multiplier for each speed level

import argparse
from pathlib import Path
import time
import math

from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2Wheels
# from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics
from joyconrobotics import JoyconRobotics
from joyconrobotics.device import get_L_id, get_R_id

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


def find_joycon_id(device):
    if device == "right":
        return get_R_id()
    return get_L_id()


def has_joycon_id(joycon_id):
    return joycon_id is not None and all(value is not None for value in joycon_id)


def check_required_joycons_available():
    joycon_ids = {
        "right": find_joycon_id("right"),
        "left": find_joycon_id("left"),
    }
    missing_devices = [device for device, joycon_id in joycon_ids.items() if not has_joycon_id(joycon_id)]
    if missing_devices:
        details = ", ".join(f"{device}: {joycon_ids[device]}" for device in missing_devices)
        raise RuntimeError(
            f"No usable Joy-Con found for: {', '.join(missing_devices)}. "
            f"Detected ids: {details}. Pair/connect both Joy-Cons, then run this script again."
        )


class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        
        # Set different center values for left and right Joy-Cons
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000
        
        # Gripper control related variables
        self.gripper_speed = 0.4  # Gripper open/close speed (degrees/frame)
        self.gripper_direction = 1  # 1 means open, -1 means close
        self.gripper_min = 0  # Minimum angle (fully closed)
        self.gripper_max = 90  # Maximum angle (fully open)
        self.last_gripper_button_state = 0  # Record previous frame button state for detecting press events
    
    def common_update(self):
        # Modified update logic: joystick only controls fixed axes
        speed_scale = 0.001
        
        # Get current orientation data to print pitch
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        
        def move_servo2(direction):
            self.position[0] += speed_scale * direction * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * direction * self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)

        def move_y(direction):
            self.position[1] += speed_scale * direction * self.dof_speed[1] * self.direction_reverse[1]

        if self.joycon.is_right():
            button_y_left = self.joycon.get_button_y()
            button_a_right = self.joycon.get_button_a()
            button_x_forward = self.joycon.get_button_x()
            button_b_backward = self.joycon.get_button_b()
            button_servo3_up = self.joycon.get_button_r()
            button_servo3_down = self.joycon.get_button_plus()
        else:
            button_y_left = self.joycon.get_button_left()
            button_a_right = self.joycon.get_button_right()
            button_x_forward = self.joycon.get_button_up()
            button_b_backward = self.joycon.get_button_down()
            button_servo3_up = self.joycon.get_button_l()
            button_servo3_down = self.joycon.get_button_minus()

        if button_y_left == 1:
            move_y(-1)
        if button_a_right == 1:
            move_y(1)
        if button_x_forward == 1:
            move_servo2(1)
        if button_b_backward == 1:
            move_servo2(-1)
        if button_servo3_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        if button_servo3_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        # Home button reset logic (simplified version)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # Gripper control logic (hold for linear increase/decrease mode)
        for event_type, status in self.button.events():
            if self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        # Gripper button state detection and direction control
        gripper_button_pressed = False
        if self.joycon.is_right():
            # Right Joy-Con uses ZR button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # Left Joy-Con uses ZL button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        # Detect button press events (from 0 to 1) to change direction
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # Button just pressed, change direction
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        
        # Update button state record
        self.last_gripper_button_state = gripper_button_pressed
        
        # Linear control of gripper open/close when holding gripper button
        if gripper_button_pressed:
            # Check if exceeding limits
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            
            # If exceeding limits, stop moving
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state
            # If exceeding limits, stay at current position, don't change direction

        

        # Button control state
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control
    
class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Set step size
        self.degree_step = 2
        self.xy_step = 0.005
        
        # Start from the recorded startup pose, so quitting can return here.
        self.target_positions = self.joint_positions.copy()
        self.roll_baseline = None
        self.wrist_roll_anchor = self.target_positions["wrist_roll"]
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        
        # Reset kinematics variables to initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        self.roll_baseline = None
        self.wrist_roll_anchor = self.target_positions["wrist_roll"]
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        """Handle Joy-Con input, update arm control - based on 6_so100_joycon_ee_control.py"""
        x, y, z, roll_, pitch_, yaw = joycon_pose
        
        # Calculate pitch control - consistent with 6_so100_joycon_ee_control.py
        pitch = -pitch_ * 60 + 10
        
        # Set coordinates - consistent with 6_so100_joycon_ee_control.py
        current_x = 0.1629 + x
        current_y = 0.1131 + z
        
        if self.roll_baseline is None:
            self.roll_baseline = roll_
            self.wrist_roll_anchor = self.target_positions["wrist_roll"]

        # Calculate roll as a delta from the startup Joy-Con orientation so the wrist does not jump on startup.
        roll = self.wrist_roll_anchor + (roll_ - self.roll_baseline) * 45
        
        # print(f"[{self.prefix}] pitch: {pitch}")

        # Add y value to control shoulder_pan joint - consistent with 6_so100_joycon_ee_control.py
        y_scale = 250.0  # Scaling factor, can be adjusted as needed
        self.target_positions["shoulder_pan"] = y * y_scale
        
        # Use inverse kinematics to calculate joint angles - consistent with 6_so100_joycon_ee_control.py
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")
        
        # Set wrist_flex - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch
        
        # Set wrist_roll - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_roll"] = roll
        
        # Gripper control - now set directly in main loop, no need to handle here
        pass

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.vertical_degree_step = 2  # Move 2 degrees each time
        self.horizontal_degree_step = 1  # Head left/right is more sensitive, so use a smaller step
        # Initialize head motor positions
        self.head_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.target_positions = self.head_positions.copy()
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print(f"[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """Handle right Joy-Con stick input to control head motors."""
        stick_vertical = joycon.joycon.get_stick_right_vertical()
        stick_horizontal = joycon.joycon.get_stick_right_horizontal()
        threshold = 300

        if stick_vertical > joycon.joycon_stick_v_0 + threshold:
            self.target_positions["head_motor_2"] -= self.vertical_degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        elif stick_vertical < joycon.joycon_stick_v_0 - threshold:
            self.target_positions["head_motor_2"] += self.vertical_degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if stick_horizontal < joycon.joycon_stick_h_0 - threshold:
            self.target_positions["head_motor_1"] += self.horizontal_degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        elif stick_horizontal > joycon.joycon_stick_h_0 + threshold:
            self.target_positions["head_motor_1"] -= self.horizontal_degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

def get_joycon_base_pressed_keys(joycon, robot):
    """Get base control commands from the left Joy-Con stick."""
    stick_vertical = joycon.joycon.get_stick_left_vertical()
    stick_horizontal = joycon.joycon.get_stick_left_horizontal()
    threshold = 300

    pressed_keys = set()

    if stick_vertical > joycon.joycon_stick_v_0 + threshold:
        pressed_keys.add(robot.teleop_keys["forward"])
        print("[BASE] Forward")
    elif stick_vertical < joycon.joycon_stick_v_0 - threshold:
        pressed_keys.add(robot.teleop_keys["backward"])
        print("[BASE] Backward")

    if stick_horizontal < joycon.joycon_stick_h_0 - threshold:
        pressed_keys.add(robot.teleop_keys["rotate_left"])
        print("[BASE] Left turn")
    elif stick_horizontal > joycon.joycon_stick_h_0 + threshold:
        pressed_keys.add(robot.teleop_keys["rotate_right"])
        print("[BASE] Right turn")

    return pressed_keys

# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 10.0  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 10.0  # deceleration slope (speed/second)
BASE_TOP_SPEED_LEVELS = [2.0, 4.0, 6.0]  # maximum speed multiplier per speed level
MIN_VELOCITY_THRESHOLD = 0.02 # minimum velocity to send to motors during deceleration

class SmoothBaseController:
    """Simplified smooth base controller with acceleration/deceleration for differential drive"""
    
    def __init__(self):
        self.current_speed = 0.0
        self.last_time = time.time()
        self.last_direction = {"x.vel": 0.0, "theta.vel": 0.0}
        self.is_moving = False
        self.last_speed_down_pressed = False
        self.last_speed_up_pressed = False

    def max_speed_multiplier(self, robot):
        level_index = min(robot.speed_index, len(BASE_TOP_SPEED_LEVELS) - 1)
        return BASE_TOP_SPEED_LEVELS[level_index]

    def update_speed_level(self, joycon_left, joycon_right, robot):
        speed_down_pressed = joycon_left.joycon.get_button_left_sl() == 1
        speed_up_pressed = joycon_right.joycon.get_button_right_sr() == 1

        if speed_down_pressed and not self.last_speed_down_pressed:
            previous_speed_index = robot.speed_index
            robot.speed_index = max(robot.speed_index - 1, 0)
            if robot.speed_index != previous_speed_index:
                print(
                    f"[BASE] Speed level decreased to {robot.speed_index + 1}/{len(robot.speed_levels)} "
                    f"(top multiplier {self.max_speed_multiplier(robot):.1f}x)"
                )
        if speed_up_pressed and not self.last_speed_up_pressed:
            previous_speed_index = robot.speed_index
            robot.speed_index = min(robot.speed_index + 1, len(robot.speed_levels) - 1)
            if robot.speed_index != previous_speed_index:
                print(
                    f"[BASE] Speed level increased to {robot.speed_index + 1}/{len(robot.speed_levels)} "
                    f"(top multiplier {self.max_speed_multiplier(robot):.1f}x)"
                )

        self.current_speed = min(self.current_speed, self.max_speed_multiplier(robot))

        self.last_speed_down_pressed = speed_down_pressed
        self.last_speed_up_pressed = speed_up_pressed
    
    def update(self, pressed_keys, robot):
        """Update smooth control and return base action"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        max_speed_multiplier = self.max_speed_multiplier(robot)
        self.current_speed = min(self.current_speed, max_speed_multiplier)
        
        # Check if any base keys are pressed
        base_keys = [
            robot.teleop_keys['forward'],
            robot.teleop_keys['backward'], 
            robot.teleop_keys['rotate_left'],
            robot.teleop_keys['rotate_right']
        ]
        any_key_pressed = any(key in pressed_keys for key in base_keys)
        
        # Calculate base action directly (bypass robot's built-in speed control)
        base_action = {"x.vel": 0.0, "theta.vel": 0.0}
        
        if any_key_pressed:
            # Keys pressed - calculate direction and accelerate
            if not self.is_moving:
                self.is_moving = True
                print("[BASE] Starting acceleration")
            
            # Get current speed level from robot
            speed_setting = robot.speed_levels[robot.speed_index]
            linear_speed = speed_setting["linear"]  # e.g. 0.1, 0.2, or 0.3
            angular_speed = speed_setting["angular"]  # e.g. 30, 60, or 90
            
            # Calculate direction based on pressed keys
            if robot.teleop_keys["forward"] in pressed_keys:
                base_action["x.vel"] += linear_speed
            if robot.teleop_keys["backward"] in pressed_keys:
                base_action["x.vel"] -= linear_speed
            if robot.teleop_keys["rotate_left"] in pressed_keys:
                base_action["theta.vel"] += angular_speed
            if robot.teleop_keys["rotate_right"] in pressed_keys:
                base_action["theta.vel"] -= angular_speed
            
            # Store current direction for deceleration
            self.last_direction = base_action.copy()
            
            # Accelerate
            self.current_speed += BASE_ACCELERATION_RATE * dt
            self.current_speed = min(self.current_speed, max_speed_multiplier)
                
        else:
            # No keys pressed - decelerate
            if self.is_moving:
                self.is_moving = False
                print("[BASE] Starting deceleration")
            
            # Use last direction for deceleration
            if self.current_speed > 0.01 and self.last_direction:
                base_action = self.last_direction.copy()
            
            # Decelerate
            self.current_speed -= BASE_DECELERATION_RATE * dt
            self.current_speed = max(self.current_speed, 0.0)
        
        # Apply speed multiplier
        if base_action:
            for key in base_action:
                if 'vel' in key:
                    original_value = base_action[key]
                    base_action[key] *= self.current_speed
                    
                    # Ensure minimum velocity during deceleration to prevent motor cutoff
                    if self.current_speed > 0.01 and abs(base_action[key]) < MIN_VELOCITY_THRESHOLD:
                        # During deceleration, maintain minimum velocity to keep motors moving
                        base_action[key] = MIN_VELOCITY_THRESHOLD if original_value > 0 else -MIN_VELOCITY_THRESHOLD
        
        # Debug output
        if any_key_pressed:
            print(f"[BASE] ACCEL: Speed={self.current_speed:.2f}/{max_speed_multiplier:.1f}, Action={base_action}")
        # elif self.current_speed > 0.01:
        #     print(f"[BASE] DECEL: Speed={self.current_speed:.2f}, Action={base_action}")
        # elif self.current_speed <= 0.01:
        #     print(f"[BASE] STOPPED: Speed={self.current_speed:.2f}")
        
        return base_action

# Global smooth controller instance
smooth_controller = SmoothBaseController()


def return_to_recorded_start(left_arm, right_arm, head_control, robot, duration_s=2.0, fps=50, log_rerun=False):
    print("[MAIN] Returning arms and head to recorded startup positions before quit.")
    left_arm.target_positions = left_arm.joint_positions.copy()
    right_arm.target_positions = right_arm.joint_positions.copy()
    head_control.target_positions = head_control.head_positions.copy()

    deadline = time.time() + duration_s
    while time.time() < deadline:
        left_action = left_arm.p_control_action(robot)
        right_action = right_arm.p_control_action(robot)
        head_action = head_control.p_control_action(robot)
        action = {**left_action, **right_action, **head_action, "x.vel": 0.0, "theta.vel": 0.0}
        robot.send_action(action)
        if log_rerun:
            obs = robot.get_observation()
            log_rerun_data(obs, action)
        precise_sleep(1.0 / fps)


def main():
    parser = argparse.ArgumentParser(description="XLeRobot 2Wheels Joy-Con teleoperation")
    parser.add_argument(
        "--log-rerun-data",
        action="store_true",
        help="Enable Rerun visualization logging. Disabled by default.",
    )
    args = parser.parse_args()

    FPS = 30

    joycon_right = None
    joycon_left = None
    robot = None

    # Try to use saved calibration file to avoid recalibrating each time
    # You can modify robot_id here to match your robot configuration
    try:
        check_required_joycons_available()

        # Initialize Joy-Con controllers before robot/Rerun startup so missing controllers fail early.
        print("[MAIN] Initializing right Joy-Con controller...")
        joycon_right = FixedAxesJoyconRobotics(
            "right",
            dof_speed=[2, 2, 2, 1, 1, 1]
        )
        print(f"[MAIN] Right Joy-Con controller connected")
        print("[MAIN] Initializing left Joy-Con controller...")
        joycon_left = FixedAxesJoyconRobotics(
            "left",
            dof_speed=[2, 2, 2, 1, 1, 1]
        )
        print(f"[MAIN] Left Joy-Con controller connected")

        config_path = Path(__file__).resolve().parents[1] / "config" / "xlerobot.json"
        robot_config = XLerobot2WheelsConfig.from_json(config_path)
        robot = XLerobot2Wheels(robot_config)

        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
        if robot.is_calibrated:
            print(f"[MAIN] Robot is calibrated and ready to use!")
        else:
            print(f"[MAIN] Robot requires calibration")
    except Exception as e:
        print(f"[MAIN] Startup failed: {e}")
        if "robot_config" in locals():
            print(f"[MAIN] Robot config: {robot_config}")
        if robot is not None:
            print(f"[MAIN] Robot: {robot}")
        if joycon_right is not None:
            joycon_right.disconnect()
        if joycon_left is not None:
            joycon_left.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()
        return

    if args.log_rerun_data:
        init_rerun(session_name="xlerobot_2wheels_teleop_joycon")

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left")
    right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right")
    head_control = SimpleHeadControl(obs)

    print("[MAIN] Recorded startup arm and head positions. They will return there before quit.")

    # Print comprehensive keymap information based on robot config
    print("\n" + "="*80)
    print("🤖 XLeRobot 2Wheels Joy-Con Control Instructions")
    print("="*80)
    
    print("\n📱 Base Control (Differential Drive):")
    print("    - Left joystick up/down: Forward/backward")
    print("    - Left joystick left/right: Rotate left/right")
    print("    - Left SL: Speed down")
    print("    - Right SR: Speed up")
    print("    🚀 Smooth Control: Linear acceleration when holding, linear deceleration when released")
    
    print("\n🦾 Right Arm Control:")
    print("   Position Control:")
    print("    - Y/A: Y-axis left/right")
    print("    - X/B: Servo 2 forward/backward")
    print("    - R/Plus: Servo 3 up/down")
    print("   Gyro Control:")
    print("    - Joy-Con pitch/tilt controls wrist flex, keeping the wrist angle aligned with the arm pose")
    print("    - Joy-Con roll/rotation controls wrist roll relative to the startup Joy-Con orientation")
    print("   Gripper Control:")
    print("    - ZR Button: Hold to toggle open/close direction, continue holding for linear open/close")
    
    print("\n🦾 Left Arm Control:")
    print("   Position Control:")
    print("    - D-pad left/right: Y-axis left/right")
    print("    - D-pad up/down: Servo 2 forward/backward")
    print("    - L/Minus: Servo 3 up/down")
    print("   Gyro Control:")
    print("    - Joy-Con pitch/tilt controls wrist flex, keeping the wrist angle aligned with the arm pose")
    print("    - Joy-Con roll/rotation controls wrist roll relative to the startup Joy-Con orientation")
    print("   Gripper Control:")
    print("    - ZL Button: Hold to toggle open/close direction, continue holding for linear open/close")
    
    print("\n👁️ Head Control:")
    print("   Right Joy-Con joystick:")
    print("    - Up/Down: Tilt the head down/up (Head Motor 2)")
    print("    - Left/Right: Pan the head left/right with smaller steps (Head Motor 1)")
    
    print(f"\n⚙️ Robot Configuration:")
    print(f"   Wheel Radius: {robot.config.wheel_radius:.3f}m")
    print(f"   Wheelbase: {robot.config.wheelbase:.3f}m")
    print(f"   Speed Levels: {len(robot.speed_levels)} levels")
    for i, level in enumerate(robot.speed_levels):
        top_speed = BASE_TOP_SPEED_LEVELS[min(i, len(BASE_TOP_SPEED_LEVELS) - 1)]
        print(
            f"      Level {i+1}: Linear {level['linear']:.1f}m/s, "
            f"Angular {level['angular']:.0f}°/s, Top Multiplier {top_speed:.1f}x"
        )
    
    print(f"\n🚀 Smooth Control Parameters:")
    print(f"   Acceleration Rate: {BASE_ACCELERATION_RATE:.1f} speed/second")
    print(f"   Deceleration Rate: {BASE_DECELERATION_RATE:.1f} speed/second")
    print(f"   Top Speed Multipliers: {', '.join(f'{speed:.1f}x' for speed in BASE_TOP_SPEED_LEVELS)}")
    
    print("\n" + "="*80)
    print("🎮 Control started! Use Joy-Con to control robot")
    print("="*80 + "\n")

    return_to_start_on_exit = True
    try:
        while True:
            pose_right, gripper_right, control_button_right = joycon_right.get_control()
            # print(f"pose_right: {pose_right}, gripper_right: {gripper_right}, control_button_right: {control_button_right}")
            pose_left, gripper_left, control_button_left = joycon_left.get_control()
            # print(f"pose_left: {pose_left}, gripper_left: {gripper_left}, control_button_left: {control_button_left}")

            if joycon_right.joycon.get_button_home() == 1:
                print("[MAIN] Right Home pressed. Exiting teleoperation...")
                break

            if control_button_right == 8:  # reset button
                print("[MAIN] Reset to zero position!")
                right_arm.move_to_zero_position(robot)
                left_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                continue

            # Handle gripper control - directly use Joy-Con gripper state
            right_arm.target_positions["gripper"] = gripper_right
            left_arm.target_positions["gripper"] = gripper_left
            
            right_arm.handle_joycon_input(pose_right, gripper_right)
            right_action = right_arm.p_control_action(robot)
            left_arm.handle_joycon_input(pose_left, gripper_left)
            left_action = left_arm.p_control_action(robot)
            head_control.handle_joycon_input(joycon_right)
            head_action = head_control.p_control_action(robot)

            smooth_controller.update_speed_level(joycon_left, joycon_right, robot)
            pressed_keys = get_joycon_base_pressed_keys(joycon_left, robot)
            
            # Get smooth base action with linear acceleration/deceleration
            smooth_base_action = smooth_controller.update(pressed_keys, robot)

            # Merge all actions
            action = {**left_action, **right_action, **head_action, **smooth_base_action}
            robot.send_action(action)

            if args.log_rerun_data:
                obs = robot.get_observation()
                log_rerun_data(obs, action)
    except KeyboardInterrupt:
        print("[MAIN] Keyboard interrupt received. Exiting teleoperation...")
    finally:
        if return_to_start_on_exit and robot is not None and robot.is_connected:
            robot.send_action({"x.vel": 0.0, "theta.vel": 0.0})
            return_to_recorded_start(
                left_arm,
                right_arm,
                head_control,
                robot,
                duration_s=2.0,
                fps=FPS,
                log_rerun=args.log_rerun_data,
            )
        if joycon_right is not None:
            joycon_right.disconnect()
        if joycon_left is not None:
            joycon_left.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
