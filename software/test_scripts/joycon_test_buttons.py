from joyconrobotics import JoyconRobotics
from joyconrobotics.device import get_L_id, get_R_id
import argparse
import math
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Show pressed Joy-Con buttons in place.")
    parser.add_argument(
        "--device",
        choices=("both", "left", "right"),
        default="both",
        help="Joy-Con side to read. Defaults to both.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=1.0,
        help="Seconds to sample the resting Joy-Con baseline before display starts.",
    )
    return parser.parse_args()


def find_joycon_id(device):
    if device == "right":
        return get_R_id()
    return get_L_id()


def has_joycon_id(joycon_id):
    return joycon_id is not None and all(value is not None for value in joycon_id)


def get_pressed_buttons(joycon):
    """Return the names of currently pressed buttons for either Joy-Con side."""
    if joycon.is_right():
        button_getters = {
            "A": joycon.get_button_a,
            "B": joycon.get_button_b,
            "X": joycon.get_button_x,
            "Y": joycon.get_button_y,
            "R": joycon.get_button_r,
            "ZR": joycon.get_button_zr,
            "SL": joycon.get_button_right_sl,
            "SR": joycon.get_button_right_sr,
            "Plus": joycon.get_button_plus,
            "Home": joycon.get_button_home,
            "Stick": joycon.get_button_r_stick,
        }
    else:
        button_getters = {
            "Up": joycon.get_button_up,
            "Down": joycon.get_button_down,
            "Left": joycon.get_button_left,
            "Right": joycon.get_button_right,
            "L": joycon.get_button_l,
            "ZL": joycon.get_button_zl,
            "SL": joycon.get_button_left_sl,
            "SR": joycon.get_button_left_sr,
            "Minus": joycon.get_button_minus,
            "Capture": joycon.get_button_capture,
            "Stick": joycon.get_button_l_stick,
        }

    pressed_buttons = []
    for button_name, get_button in button_getters.items():
        if get_button() == 1:
            pressed_buttons.append(button_name)
    return pressed_buttons


def format_buttons(pressed_buttons):
    return ", ".join(pressed_buttons) if pressed_buttons else "None"


def get_stick_state(joycon):
    if joycon.is_right():
        horizontal = joycon.get_stick_right_horizontal()
        vertical = joycon.get_stick_right_vertical()
    else:
        horizontal = joycon.get_stick_left_horizontal()
        vertical = joycon.get_stick_left_vertical()
    return horizontal, vertical


def format_stick_direction(horizontal, vertical, center_h, center_v, threshold=300):
    directions = []
    if vertical > center_v + threshold:
        directions.append("Up")
    elif vertical < center_v - threshold:
        directions.append("Down")

    if horizontal > center_h + threshold:
        directions.append("Right")
    elif horizontal < center_h - threshold:
        directions.append("Left")

    return "+".join(directions) if directions else "Center"


def get_motion_state(joycon):
    accel = (
        joycon.get_accel_x(),
        joycon.get_accel_y(),
        joycon.get_accel_z(),
    )
    gyro = (
        joycon.get_gyro_x(),
        joycon.get_gyro_y(),
        joycon.get_gyro_z(),
    )
    return accel, gyro


def subtract_vector(values, baseline):
    return tuple(value - baseline_value for value, baseline_value in zip(values, baseline, strict=True))


def average_vectors(vectors):
    return tuple(sum(vector[axis] for vector in vectors) / len(vectors) for axis in range(3))


def calibrate_motion_baseline(joycons, calibration_seconds):
    print(f"Calibrating motion baseline for {calibration_seconds:.1f}s. Keep Joy-Con(s) still...")
    samples = {device: {"accel": [], "gyro": []} for device, _ in joycons}
    start_time = time.time()
    while time.time() - start_time < calibration_seconds:
        for device, joyconrobotics in joycons:
            accel, gyro = get_motion_state(joyconrobotics.joycon)
            samples[device]["accel"].append(accel)
            samples[device]["gyro"].append(gyro)
        time.sleep(0.02)

    baselines = {}
    for device, device_samples in samples.items():
        baselines[device] = {
            "accel": average_vectors(device_samples["accel"]),
            "gyro": average_vectors(device_samples["gyro"]),
        }
    print("Calibration complete.")
    return baselines


def format_tilt(accel, threshold=2500):
    ax, ay, _ = accel
    directions = []
    if ay > threshold:
        directions.append("Nose up")
    elif ay < -threshold:
        directions.append("Nose down")

    if ax > threshold:
        directions.append("Tilt right")
    elif ax < -threshold:
        directions.append("Tilt left")

    return "+".join(directions) if directions else "Level"


def format_shake(accel):
    magnitude = math.sqrt(sum(value * value for value in accel))
    if magnitude > 9000:
        return "Hard shake"
    if magnitude > 4500:
        return "Shake"
    return "Still"


def format_rotation(gyro, threshold=2000):
    gx, gy, gz = gyro
    rotations = []
    if gx > threshold:
        rotations.append("Roll+")
    elif gx < -threshold:
        rotations.append("Roll-")

    if gy > threshold:
        rotations.append("Pitch+")
    elif gy < -threshold:
        rotations.append("Pitch-")

    if gz > threshold:
        rotations.append("Yaw+")
    elif gz < -threshold:
        rotations.append("Yaw-")

    return "+".join(rotations) if rotations else "No turn"


STICK_CENTERS = {
    "left": (2000, 2300),
    "right": (2100, 1900),
}


args = parse_args()
devices = ("left", "right") if args.device == "both" else (args.device,)
joycon_ids = {device: find_joycon_id(device) for device in devices}
missing_devices = [device for device, joycon_id in joycon_ids.items() if not has_joycon_id(joycon_id)]
if missing_devices:
    for device in missing_devices:
        print(f"No usable {device} Joy-Con found. Detected id: {joycon_ids[device]}")
    print("Pair/connect the missing Joy-Con or run with --device left/right to test only one side.")
    raise SystemExit(1)

joycons = []
for device in devices:
    joycons.append((device, JoyconRobotics(device=device, without_rest_init=True)))

motion_baselines = calibrate_motion_baseline(joycons, args.calibration_seconds)

print(f"Joy-Con button test ({args.device}). Press Ctrl+C to stop.")
for device, _ in joycons:
    print(
        f"{device.capitalize():>5} buttons: None | stick: h=0000 v=0000 Center | "
        "motion: Level, Still, No turn"
    )
    print("      accel delta: x=      0 y=      0 z=      0")
    print("       gyro delta: x=      0 y=      0 z=      0")

try:
    while True:
        lines = []
        for device, joyconrobotics in joycons:
            pressed_buttons = get_pressed_buttons(joyconrobotics.joycon)
            stick_h, stick_v = get_stick_state(joyconrobotics.joycon)
            center_h, center_v = STICK_CENTERS[device]
            stick_direction = format_stick_direction(stick_h, stick_v, center_h, center_v)
            accel, gyro = get_motion_state(joyconrobotics.joycon)
            accel_delta = subtract_vector(accel, motion_baselines[device]["accel"])
            gyro_delta = subtract_vector(gyro, motion_baselines[device]["gyro"])
            motion_text = f"{format_tilt(accel_delta)}, {format_shake(accel_delta)}, {format_rotation(gyro_delta)}"
            lines.append(
                f"{device.capitalize():>5} buttons: {format_buttons(pressed_buttons)} | "
                f"stick: h={stick_h:04d} v={stick_v:04d} {stick_direction} | "
                f"motion: {motion_text}"
            )
            accel_x, accel_y, accel_z = accel_delta
            gyro_x, gyro_y, gyro_z = gyro_delta
            lines.append(f"      accel delta: x={accel_x:7.0f} y={accel_y:7.0f} z={accel_z:7.0f}")
            lines.append(f"       gyro delta: x={gyro_x:7.0f} y={gyro_y:7.0f} z={gyro_z:7.0f}")
        print(f"\033[{len(lines)}F\033[J" + "\n".join(lines), end="", flush=True)
        time.sleep(0.02)
except KeyboardInterrupt:
    print()
finally:
    for _, joyconrobotics in joycons:
        joyconrobotics.disconnect()
