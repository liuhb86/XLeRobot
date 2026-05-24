#!/usr/bin/env python3
"""Record XLeRobot 2-wheels motor positions to a JSON config file."""

import argparse
import json
from pathlib import Path

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig


def resolve_initial_motor_position_file(config_path):
    with open(config_path) as f:
        data = json.load(f)

    fpath = Path(data["initial-motor-position-file"]).expanduser()
    if not fpath.is_absolute():
        fpath = config_path.parent / fpath
    return fpath


def parse_args():
    parser = argparse.ArgumentParser(description="Record XLeRobot 2-wheels motor states.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "xlerobot.json",
        help="Path to xlerobot.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for recorded motor positions. Defaults to initial-motor-position-file in config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_config = XLerobot2WheelsConfig.from_json(args.config)
    output = args.output or resolve_initial_motor_position_file(args.config)
    robot = XLerobot2Wheels(robot_config)

    robot.connect()
    try:
        obs = robot.get_observation()
        motor_states = {key: value for key, value in obs.items() if key.endswith(".pos")}
        payload = {
            "robot-id": robot_config.id,
            "motor-states": motor_states,
        }

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(payload, f, indent=4)

        print(f"Recorded {len(motor_states)} motor states to {output}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
