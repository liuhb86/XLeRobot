#!/usr/bin/env python3
"""Calibrate XLeRobot 2-wheels motors from the shared JSON config."""

import argparse
from pathlib import Path

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate XLeRobot 2-wheels motors.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "xlerobot.json",
        help="Path to xlerobot.json. Relative calibration-dir values are resolved from this file.",
    )
    parser.add_argument(
        "--motors",
        nargs="+",
        default=None,
        help=(
            "Calibration targets. Omit for all motors. Supported groups: all, left_arm_motors, "
            "right_arm_motors, head_motors, base_motors. Individual motor names are also accepted."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_config = XLerobot2WheelsConfig.from_json(args.config)
    robot = XLerobot2Wheels(robot_config)

    targets = args.motors
    if targets is not None and len(targets) == 1:
        targets = targets[0]

    print(f"Using config: {args.config}")
    print(f"Calibration file: {robot.calibration_fpath}")
    print(f"Calibration targets: {targets or 'all'}")
    robot.calibrate(targets)


if __name__ == "__main__":
    main()
