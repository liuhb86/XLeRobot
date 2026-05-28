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
        self.position = [0.0, 0.0, 0.0]
        self.dof_speed = [2, 2, 2, 1, 1, 1]

        self.gripper_state = 1.0
        self.gripper_speed = 0.4
        self.gripper_min = 0
        self.gripper_max = 90


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

    def set_end_effector_target(self, orientation_rad, control_vector):
        """Set arm target from a generic end-effector pose tuple."""
        speed_scale = 0.001
        _roll, pitch, _yaw = orientation_rad

        def move_servo2(direction):
            self.position[0] += speed_scale * direction * self.dof_speed[0] * math.cos(pitch)
            self.position[2] += speed_scale * direction * self.dof_speed[1] * math.sin(pitch)

        def move_y(direction):
            self.position[1] += speed_scale * direction * self.dof_speed[1]

        d1, d2, d3 = control_vector
        move_y(d1)
        move_servo2(d2)
        self.position[2] += d3 * speed_scale * self.dof_speed[2]

        x, y, z = self.position
        pitch = -pitch * 60 + 10
        current_x = 0.1629 + x
        current_y = 0.1131 + z

        if self.roll_baseline is None:
            self.roll_baseline = roll
            self.wrist_roll_anchor = self.target_positions["wrist_roll"]

        roll = self.wrist_roll_anchor + (roll - self.roll_baseline) * 45
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
        

    def increment_gripper_target(self, delta):
        new_gripper_state = self.gripper_state + delta * self.gripper_speed
        new_gripper_state = max(new_gripper_state, self.gripper_min)
        new_gripper_state = min(new_gripper_state, self.gripper_max)
        self.gripper_state = new_gripper_state
        self.target_positions["gripper"] = gripper_state

    def p_control_action(self, obs):
        action = {}
        for joint, target in self.target_positions.items():
            current = obs[f"{self.prefix}_arm_{joint}.pos"]
            action[f"{self.joint_map[joint]}.pos"] = current + self.kp * (target - current)
        return action
