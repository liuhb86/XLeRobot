"""Joy-Con input loop for updating RobotController target state."""

from __future__ import annotations

import math
import threading
import time

from joyconrobotics.device import get_L_id, get_R_id
from joyconrobotics.gyro import GyroTrackingJoyCon
from joyconrobotics.joycon import JoyCon
from joyconrobotics.joyconrobotics import AttitudeEstimator

from base_controller import BASE_BACKWARD, BASE_FORWARD, BASE_ROTATE_LEFT, BASE_ROTATE_RIGHT

SLEEP_HOLD_SECONDS = 3.0
ACTIVE_SOLVE_LOOP_SLEEP_SECONDS = 0.01
ASLEEP_SOLVE_LOOP_SLEEP_SECONDS = 1.0


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

class JoyConHalfCommand:
    def __init__(self, device, reset_device=True):
        self.joycon_id = find_joycon_id(device)
        self.joycon = JoyCon(*self.joycon_id)
        self.gyro = GyroTrackingJoyCon(*self.joycon_id)
        self.orientation_sensor = AttitudeEstimator(
            common_rad=True,
            lerobot=False,
            pitch_down_double=False,
            lowpassfilter_alpha_rate=1,
        )

        self.gripper_direction = 1
        self.last_gripper_button_state = 0  

        if reset_device:
            self.reset_joycon()

        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900
            self.joycon_stick_h_0 = 2100
        else:
            self.joycon_stick_v_0 = 2300
            self.joycon_stick_h_0 = 2000

    def disconnect(self):
        self.joycon._close()
        self.gyro._close()

    def reset_joycon(self):
        print("\033[33mcalibrating(2 seconds)..., please place it horizontally on the desktop.\033[0m")
        self.gyro.calibrate()
        time.sleep(2)
        self.gyro.reset_orientation()
        self.orientation_sensor.reset_yaw()
        print("\033[32mJoycon calibrations is complete.\033[0m")

    def update_orientation(self):
        orientation_rad = self.orientation_sensor.update(self.gyro.gyro_in_rad[0], self.gyro.accel_in_g[0])
        return orientation_rad

    def get_control_vector(self):
        if self.joycon.is_right():
            button_left = self.joycon.get_button_y()
            button_right = self.joycon.get_button_a()
            button_up = self.joycon.get_button_x()
            button_down = self.joycon.get_button_b()
            button_sign = self.joycon.get_button_plus()
            button_lr = self.joycon.get_button_r()
        else:
            button_left = self.joycon.get_button_left()
            button_right = self.joycon.get_button_right()
            button_up = self.joycon.get_button_up()
            button_down = self.joycon.get_button_down()
            button_sign = self.joycon.get_button_minus()
            button_lr = self.joycon.get_button_l()
            
        control_vector = (button_right - button_left, button_up - button_down, button_sign - button_lr)
        return control_vector

    def update_gripper(self):
        if self.joycon.is_right():
            button_gripper = self.joycon.get_button_zr()
        else:
            button_gripper = self.joycon.get_button_zl()
        gripper_button_pressed = button_gripper == 1

        gripper_action = 0;
        if gripper_button_pressed:
            if self.last_gripper_button_state == 0:
                self.gripper_direction *= -1
                print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
            gripper_action = self.gripper_direction

        self.last_gripper_button_state = gripper_button_pressed
        return gripper_action

    def get_joycon_stick_directions(self):
        stick_vertical = self.joycon.get_stick_left_vertical()
        stick_horizontal = self.joycon.get_stick_left_horizontal()
        threshold = 300
        v = 0
        h = 0

        if stick_vertical > self.joycon_stick_v_0 + threshold:
            v = 1
            print("[BASE] Forward")
        elif stick_vertical < self.joycon_stick_v_0 - threshold:
            v = -1
            print("[BASE] Backward")

        if stick_horizontal < self.joycon_stick_h_0 - threshold:
            h = -1
            print("[BASE] Left turn")
        elif stick_horizontal > self.joycon_stick_h_0 + threshold:
            h = 1
            print("[BASE] Right turn")

        return (v, h)    

class JoyConCommand:
    """Routes Joy-Con thread updates into RobotController target state."""

    def __init__(self, robot_controller=None, fps=30, log_rerun=False):
        self.robot_controller = robot_controller
        self.joycon_right = None
        self.joycon_left = None
        self.fps = fps
        self.log_rerun = log_rerun

        self._stop_event = threading.Event()
        self._stop_event.set()
        self.sleep_requested = False
        self._sleep_button_pressed_at = None

        self.last_speed_down_pressed = False
        self.last_speed_up_pressed = False

        self._thread = None
        self._thread_error = None

        
    def bind_robot_controller(self, robot_controller):
        self.robot_controller = robot_controller

    def connect(self):
        check_required_joycons_available()
        print("[MAIN] Initializing right Joy-Con controller...")
        self.joycon_right = JoyConHalfCommand("right")
        print("[MAIN] Right Joy-Con controller connected")
        print("[MAIN] Initializing left Joy-Con controller...")
        self.joycon_left = JoyConHalfCommand("left")
        print("[MAIN] Left Joy-Con controller connected")
    
    def disconnect(self):
        stop_error = None
        try:
            self.stop()
        except Exception as exc:
            stop_error = exc
        finally:
            if self.joycon_right is not None:
                self.joycon_right.disconnect()
                self.joycon_right = None
            if self.joycon_left is not None:
                self.joycon_left.disconnect()
                self.joycon_left = None

        if stop_error is not None:
            raise stop_error

    def start(self):
        if self.robot_controller is None:
            raise RuntimeError("RobotController is not bound.")
        if self.joycon_right is None or self.joycon_left is None:
            raise RuntimeError("Joy-Cons are not connected.")
        if self._thread is not None:
            return

        self.sleep_requested = False
        self.last_speed_down_pressed = False
        self.last_speed_up_pressed = False
        self._sleep_button_pressed_at = None
        self._stop_event.clear()
        self._thread_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                self._thread = None
        if self._thread_error is not None:
            error = self._thread_error
            self._thread_error = None
            raise error

    def _run(self):
        try:
            while not self._stop_event.is_set():
                if self._handle_sleep_button():
                    continue
                with self.robot_controller._lock:
                    self._update_arm_target(self.joycon_left, self.robot_controller.left_arm)
                    self._update_arm_target(self.joycon_right, self.robot_controller.right_arm)
                    self._update_head_target()
                    self._update_base_target()
                    
                self._stop_event.wait(1.0 / self.fps)
        except Exception as exc:
            self._thread_error = exc
            self._stop_event.set()

    def _update_arm_target(self, joycon, arm_controller):
        orientation_rad = joycon.update_orientation()
        control_vector = joycon.get_control_vector()
        gripper_action = joycon.update_gripper()
        arm_controller.set_end_effector_target(orientation_rad, control_vector)
        arm_controller.increment_gripper_target(gripper_action)

    def _update_base_target(self):
        self._update_base_speed_level()
        (v, h) = self.joycon_left.get_joycon_stick_directions()
        self.robot_controller.base_controller.update_speed(v, h)

    def _update_base_speed_level(self):
        speed_down_pressed = self.joycon_left.joycon.get_button_left_sl() == 1
        speed_up_pressed = self.joycon_right.joycon.get_button_right_sr() == 1

        if speed_down_pressed and not self.last_speed_down_pressed:
            self.robot_controller.base_controller.adjust_speed_level(-1)
        if speed_up_pressed and not self.last_speed_up_pressed:
            self.robot_controller.base_controller.adjust_speed_level(1)

        self.last_speed_down_pressed = speed_down_pressed
        self.last_speed_up_pressed = speed_up_pressed

    def _update_head_target(self):
        (v, h) = self.joycon_right.get_joycon_stick_directions()
        self.robot_controller.head.increment_target(h, v)

    def _quit_buttons_pressed(self):
        if self.joycon_right.joycon.get_button_home() == 1 and self.joycon_left.joycon.get_button_capture() == 1:
            print("[JOYCON] Quit requested by Capture + Home.")
            self.robot_controller.set_quit_status(True)
            self._stop_event.set()
            return True
        else:
            return False
    
    def _handle_sleep_button(self):
        if self.joycon_right.joycon.get_button_home() != 1:
            self._sleep_button_pressed_at = None
            return False
        
        if self._quit_buttons_pressed():
            return True

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
        self._wait_for_home_wake()
        if self._stop_event.is_set():
            return
        self.robot_controller.wake()
        self.sleep_requested = False
        self._sleep_button_pressed_at = None
        self.robot_controller.base_controller.reset()

    def _wait_for_home_wake(self, poll_interval_s=1):
        print("[MAIN] Sleep mode active. Press the right Joy-Con Home button to wake.")
        home_was_released = False
        while not self._stop_event.is_set():
            if self._quit_buttons_pressed():
                break
            home_pressed = self.joycon_right.joycon.get_button_home() == 1
            if not home_pressed:
                home_was_released = True
            elif home_was_released:
                print("[MAIN] Wake requested. Re-initializing teleoperation...")
                return
            self._stop_event.wait(poll_interval_s)
