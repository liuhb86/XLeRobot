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
