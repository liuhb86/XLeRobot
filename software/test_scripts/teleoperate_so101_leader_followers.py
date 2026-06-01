#!/usr/bin/env python3
"""Teleoperate SO101 followers from SO101 leaders defined in xlerobot.json."""

import argparse
import json
import logging
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate SO101 followers from configured SO101 leaders.")
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR.parent / "config" / "xlerobot.json",
        help="Path to xlerobot.json.",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=SCRIPT_DIR.parent / "config" / "calibration",
        help="Calibration root. Defaults to ../config/calibration relative to this script.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Teleoperation loop frequency.")
    parser.add_argument(
        "--teleop-time-s",
        type=float,
        default=None,
        help="Optional teleoperation duration in seconds. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Optional SO101Follower max_relative_target safety clamp.",
    )
    parser.add_argument(
        "--skip-calibration-check",
        action="store_true",
        help="Connect without asking LeRobot to write calibration files to motors.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_calibration_dir(calibration_root, device_kind, device_name, device_id):
    flat_file = calibration_root / f"{device_id}.json"
    nested_dir = calibration_root / device_kind / device_name
    nested_file = nested_dir / f"{device_id}.json"

    if nested_file.is_file():
        return nested_dir
    if flat_file.is_file():
        return calibration_root
    return nested_dir if device_kind == "robots" else calibration_root


def make_pairs(config_data, calibration_root, max_relative_target):
    leader_arms = config_data.get("leader-arms", {})
    if not leader_arms:
        raise ValueError("No leader arms found in xlerobot.json under 'leader-arms'.")

    pairs = []
    for side, leader_data in leader_arms.items():
        if side not in config_data:
            raise ValueError(f"Leader arm '{side}' has no matching follower entry in xlerobot.json.")

        follower_data = config_data[side]
        leader_id = leader_data["robot-id"]
        follower_id = follower_data["robot-id"]

        leader_config = SO101LeaderConfig(
            id=leader_id,
            port=leader_data["port"],
            calibration_dir=resolve_calibration_dir(
                calibration_root, "teleoperators", "so_leader", leader_id
            ),
        )
        follower_config = SO101FollowerConfig(
            id=follower_id,
            port=follower_data["port"],
            calibration_dir=resolve_calibration_dir(
                calibration_root, "robots", "so_follower", follower_id
            ),
            max_relative_target=max_relative_target,
        )
        pairs.append((side, SO101Leader(leader_config), SO101Follower(follower_config)))

    return pairs


def connect_pairs(pairs, calibrate):
    connected = []
    try:
        for side, leader, follower in pairs:
            print(f"Connecting {side}: leader {leader.id} on {leader.config.port}")
            leader.connect(calibrate=calibrate)
            connected.append(leader)

            print(f"Connecting {side}: follower {follower.id} on {follower.config.port}")
            follower.connect(calibrate=calibrate)
            connected.append(follower)
    except Exception:
        disconnect_devices(reversed(connected))
        raise


def disconnect_devices(devices):
    for device in devices:
        try:
            if device.is_connected:
                device.disconnect()
        except Exception as exc:
            logging.warning("Failed to disconnect %s: %s", device, exc)


def teleop_loop(pairs, fps, duration):
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        for _, leader, follower in pairs:
            action = leader.get_action()
            follower.send_action(action)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))

        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)", end="\r")

        if duration is not None and time.perf_counter() - start >= duration:
            print()
            return


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config_data = load_json(args.config)
    calibration_root = args.calibration_root.expanduser()
    pairs = make_pairs(config_data, calibration_root, args.max_relative_target)

    print(f"Config: {args.config}")
    print(f"Calibration root: {calibration_root}")
    for side, leader, follower in pairs:
        print(f"{side}: {leader.id} -> {follower.id}")
        print(f"  leader calibration: {leader.calibration_fpath}")
        print(f"  follower calibration: {follower.calibration_fpath}")

    try:
        connect_pairs(pairs, calibrate=not args.skip_calibration_check)
        teleop_loop(pairs, fps=args.fps, duration=args.teleop_time_s)
    except KeyboardInterrupt:
        print("\nStopping teleoperation.")
    finally:
        devices = []
        for _, leader, follower in pairs:
            devices.extend([follower, leader])
        disconnect_devices(devices)


if __name__ == "__main__":
    main()
