HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}


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

    def increment_target(self, head_motor_1_delta, head_motor_2_delta):
        self.target_positions["head_motor_1"] += head_motor_1_delta * 2
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
