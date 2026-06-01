#!/usr/bin/env python3
"""Calibrate an SO101 leader arm and save calibration under software/config."""

import argparse
import logging
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate an SO101 leader arm.")
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port for the SO101 leader arm, for example COM5 or /dev/ttyACM0.",
    )
    parser.add_argument(
        "--id",
        default="so101_leader",
        help="Calibration id. The output file is <calibration-dir>/<id>.json.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "config" / "calibration",
        help="Directory where the calibration JSON is saved.",
    )
    parser.add_argument(
        "--range-mode",
        action="store_true",
        help="Use LeRobot's -100..100 normalized range instead of degrees.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    config = SO101LeaderConfig(
        id=args.id,
        port=args.port,
        calibration_dir=args.calibration_dir,
        use_degrees=not args.range_mode,
    )
    leader = SO101Leader(config)

    print(f"Calibration directory: {leader.calibration_dir}")
    print(f"Calibration file: {leader.calibration_fpath}")
    print("Starting fresh SO101 leader calibration.")

    try:
        leader.connect(calibrate=False)
        leader.calibration = {}
        leader.calibrate()
    finally:
        if leader.is_connected:
            leader.disconnect()


if __name__ == "__main__":
    main()
