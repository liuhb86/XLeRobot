from __future__ import annotations

import threading
import time


BASE_ACCELERATION_RATE = 10.0
BASE_DECELERATION_RATE = 10.0
BASE_TOP_SPEED_LEVELS = [2.0, 4.0, 6.0]
BASE_SPEED_LEVELS = [
    {"linear": 0.1, "angular": 30},
    {"linear": 0.25, "angular": 60},
    {"linear": 0.4, "angular": 90},
]
MIN_VELOCITY_THRESHOLD = 0.02

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

    def update_speed(self, linear_delta, rotation_delta):
        with self._lock:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            max_speed_multiplier = self.max_speed_multiplier()
            self.current_speed = min(self.current_speed, max_speed_multiplier)

            is_accelerating = (linear_delta != 0 or rotaion_delta !=0)
            base_action = {"x.vel": 0.0, "theta.vel": 0.0}

            if is_accelerating:
                if not self.is_moving:
                    self.is_moving = True
                    print("[BASE] Starting acceleration")

                speed_setting = self.speed_levels[self.speed_index]
                base_action["x.vel"] += linear_delta * speed_setting["linear"]
                base_action["theta.vel"] += rotation_delta * speed_setting["angular"]

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
