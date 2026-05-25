"""Joy-Con input loop for updating RobotController target state."""

from __future__ import annotations

import math
import threading
import time

from joyconrobotics import JoyconRobotics
from joyconrobotics.device import get_L_id, get_R_id

from robot_controller import BASE_BACKWARD, BASE_FORWARD, BASE_ROTATE_LEFT, BASE_ROTATE_RIGHT

SLEEP_HOLD_SECONDS = 3.0


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

        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000

        self.gripper_speed = 0.4
        self.gripper_direction = 1
        self.gripper_min = 0
        self.gripper_max = 90
        self.last_gripper_button_state = 0

    def common_update(self):
        speed_scale = 0.001
        _roll, pitch, _yaw = self.get_orientation()

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

        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()

        for event_type, status in self.button.events():
            if self.joycon.is_right() and event_type == "a":
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == "y":
                self.restart_episode_button = status

        gripper_button_pressed = False
        if self.joycon.is_right():
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1

        if gripper_button_pressed and self.last_gripper_button_state == 0:
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        self.last_gripper_button_state = gripper_button_pressed

        if gripper_button_pressed:
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            if self.gripper_min <= new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state

        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            else:
                self.button_control = 0

        return self.position, self.gripper_state, self.button_control


class JoyConCommand:
    """Owns the Joy-Con polling loop and updates robot target state."""

    def __init__(self, robot_controller=None, fps=30, log_rerun=False):
        self.robot_controller = robot_controller
        self.joycon_right = None
        self.joycon_left = None
        self.fps = fps
        self.log_rerun = log_rerun
        self.sleep_requested = False
        self.last_speed_down_pressed = False
        self.last_speed_up_pressed = False
        self._stop_event = threading.Event()
        self._thread = None
        self._thread_error = None
        self._sleep_button_pressed_at = None

    def bind_robot_controller(self, robot_controller):
        self.robot_controller = robot_controller

    def connect(self):
        self.joycon_right, self.joycon_left = _initialize_joycons()

    def disconnect(self):
        thread_error = None
        try:
            self.stop()
        except Exception as exc:
            thread_error = exc
        finally:
            _disconnect_joycons(self.joycon_right, self.joycon_left)
            self.joycon_right = None
            self.joycon_left = None

        if thread_error is not None:
            raise thread_error

    def start(self):
        if self.robot_controller is None:
            raise RuntimeError("RobotController is not bound.")
        if self.joycon_right is None or self.joycon_left is None:
            raise RuntimeError("Joy-Cons are not connected.")
        if self._thread is not None and self._thread.is_alive():
            return

        self.sleep_requested = False
        self.last_speed_down_pressed = False
        self.last_speed_up_pressed = False
        self._sleep_button_pressed_at = None
        self._thread_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._thread_error is not None:
            error = self._thread_error
            self._thread_error = None
            raise error

    def wait_for_wake(self, poll_interval_s=1):
        if self.joycon_right is None:
            raise RuntimeError("Joy-Cons are not connected.")
        _wait_for_home_wake(
            self.joycon_right,
            self.joycon_left,
            self.robot_controller,
            self._stop_event,
            poll_interval_s=poll_interval_s,
        )

    def _handle_sleep_button(self):
        if self.joycon_right.joycon.get_button_home() != 1:
            self._sleep_button_pressed_at = None
            return False

        if self._sleep_button_pressed_at is None:
            self._sleep_button_pressed_at = time.time()
            print("[JOYCON] Right Home held. Keep holding for 3 seconds to sleep...")
            return False

        if time.time() - self._sleep_button_pressed_at < SLEEP_HOLD_SECONDS:
            return False

        self._sleep_until_wake()
        return True

    def _sleep_until_wake(self):
        print("[JOYCON] Right Home held for 3 seconds. Entering sleep mode...")
        self.sleep_requested = True
        self.robot_controller.sleep(return_to_start=True)
        self.wait_for_wake()
        if self._stop_event.is_set():
            return
        self.robot_controller.wake()
        self.sleep_requested = False
        self._sleep_button_pressed_at = None
        self.robot_controller.base_controller.reset()

    def _update_base_speed_level(self):
        speed_down_pressed = self.joycon_left.joycon.get_button_left_sl() == 1
        speed_up_pressed = self.joycon_right.joycon.get_button_right_sr() == 1

        if speed_down_pressed and not self.last_speed_down_pressed:
            self.robot_controller.base_controller.adjust_speed_level(-1)
        if speed_up_pressed and not self.last_speed_up_pressed:
            self.robot_controller.base_controller.adjust_speed_level(1)

        self.last_speed_down_pressed = speed_down_pressed
        self.last_speed_up_pressed = speed_up_pressed

    def _run(self):
        self.robot_controller.base_controller.reset()
        try:
            while not self._stop_event.is_set():
                pose_right, gripper_right, _control_button_right = self.joycon_right.get_control()
                pose_left, gripper_left, _control_button_left = self.joycon_left.get_control()

                if self._quit_buttons_pressed():
                    print("[JOYCON] Quit requested by Capture + Home.")
                    self.robot_controller.set_quit_status(True)
                    self._stop_event.set()
                    break

                if self._handle_sleep_button():
                    continue

                self.robot_controller.set_arm_end_effector_target("right", pose_right, gripper_right)
                self.robot_controller.set_arm_end_effector_target("left", pose_left, gripper_left)
                self._update_head_target()
                self._update_base_speed_level()
                directions = get_joycon_base_directions(self.joycon_left)
                self.robot_controller.base_controller.update(directions)
                self._stop_event.wait(1.0 / self.fps)
        except Exception as exc:
            self._thread_error = exc
            self._stop_event.set()

    def _update_head_target(self):
        stick_vertical = self.joycon_right.joycon.get_stick_right_vertical()
        stick_horizontal = self.joycon_right.joycon.get_stick_right_horizontal()
        threshold = 300
        head_motor_1_delta = 0
        head_motor_2_delta = 0

        if stick_vertical > self.joycon_right.joycon_stick_v_0 + threshold:
            head_motor_2_delta = -2
        elif stick_vertical < self.joycon_right.joycon_stick_v_0 - threshold:
            head_motor_2_delta = 2
        if stick_horizontal < self.joycon_right.joycon_stick_h_0 - threshold:
            head_motor_1_delta = 1
        elif stick_horizontal > self.joycon_right.joycon_stick_h_0 + threshold:
            head_motor_1_delta = -1

        if head_motor_1_delta or head_motor_2_delta:
            self.robot_controller.increment_head_target(head_motor_1_delta, head_motor_2_delta)

    def _quit_buttons_pressed(self):
        return (
            self.joycon_left.joycon.get_button_capture() == 1
            and self.joycon_right.joycon.get_button_home() == 1
        )


def get_joycon_base_directions(joycon):
    stick_vertical = joycon.joycon.get_stick_left_vertical()
    stick_horizontal = joycon.joycon.get_stick_left_horizontal()
    threshold = 300
    directions = 0

    if stick_vertical > joycon.joycon_stick_v_0 + threshold:
        directions |= BASE_FORWARD
        print("[BASE] Forward")
    elif stick_vertical < joycon.joycon_stick_v_0 - threshold:
        directions |= BASE_BACKWARD
        print("[BASE] Backward")

    if stick_horizontal < joycon.joycon_stick_h_0 - threshold:
        directions |= BASE_ROTATE_LEFT
        print("[BASE] Left turn")
    elif stick_horizontal > joycon.joycon_stick_h_0 + threshold:
        directions |= BASE_ROTATE_RIGHT
        print("[BASE] Right turn")

    return directions


def _initialize_joycons():
    check_required_joycons_available()
    print("[MAIN] Initializing right Joy-Con controller...")
    joycon_right = FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
    print("[MAIN] Right Joy-Con controller connected")
    print("[MAIN] Initializing left Joy-Con controller...")
    joycon_left = FixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])
    print("[MAIN] Left Joy-Con controller connected")
    return joycon_right, joycon_left


def _disconnect_joycons(joycon_right, joycon_left):
    if joycon_right is not None:
        joycon_right.disconnect()
    if joycon_left is not None:
        joycon_left.disconnect()


def _wait_for_home_wake(joycon_right, joycon_left, robot_controller, stop_event, poll_interval_s=0.25):
    print("[MAIN] Sleep mode active. Press the right Joy-Con Home button to wake.")
    home_was_released = False
    while not stop_event.is_set():
        if joycon_left.joycon.get_button_capture() == 1 and joycon_right.joycon.get_button_home() == 1:
            print("[JOYCON] Quit requested by Capture + Home.")
            robot_controller.set_quit_status(True)
            stop_event.set()
            return

        home_pressed = joycon_right.joycon.get_button_home() == 1
        if not home_pressed:
            home_was_released = True
        elif home_was_released:
            print("[MAIN] Wake requested. Re-initializing teleoperation...")
            return
        stop_event.wait(poll_interval_s)
