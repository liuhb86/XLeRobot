#!/usr/bin/env python3
"""Copy XLerobot 2-wheels arm calibration into SOFollower calibration files."""

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def default_calibration_root():
    return SCRIPT_DIR.parent / "config" / "calibration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split one xlerobot_2wheels calibration file into left/right SOFollower calibration files."
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=default_calibration_root(),
        help="LeRobot calibration root. Defaults to ../config/calibration relative to this script.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="xlerobot_2wheels calibration id. If omitted, the only JSON in robots/xlerobot_2wheels is used.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "xlerobot.json",
        help="Example robot config used to get left/right SOFollower ids.",
    )
    parser.add_argument("--left-id", type=str, default=None, help="Target SOFollower id for the left arm.")
    parser.add_argument("--right-id", type=str, default=None, help="Target SOFollower id for the right arm.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written without writing files.")
    return parser.parse_args()


def resolve_source_path(calibration_root, source_id):
    source_dir = calibration_root / "robots" / "xlerobot_2wheels"
    if source_id:
        return source_dir / f"{source_id}.json"

    candidates = sorted(source_dir.glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No xlerobot_2wheels calibration files found in {source_dir}")
    candidate_names = ", ".join(path.stem for path in candidates)
    raise ValueError(f"Multiple xlerobot_2wheels calibration files found. Use --source-id. Available: {candidate_names}")


def load_target_ids(robot_config_path, left_id, right_id):
    if left_id and right_id:
        return {"left": left_id, "right": right_id}

    with open(robot_config_path) as f:
        robot_config = json.load(f)

    return {
        "left": left_id or robot_config["left"]["robot-id"],
        "right": right_id or robot_config["right"]["robot-id"],
    }


def extract_arm_calibration(source_calibration, arm):
    prefix = f"{arm}_arm_"
    calibration = {}
    missing_keys = []
    for joint in ARM_JOINTS:
        source_key = f"{prefix}{joint}"
        if source_key not in source_calibration:
            missing_keys.append(source_key)
            continue
        calibration[joint] = source_calibration[source_key]

    if missing_keys:
        raise KeyError(f"Missing calibration keys for {arm} arm: {missing_keys}")
    return calibration


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def main():
    args = parse_args()
    source_path = resolve_source_path(args.calibration_root, args.source_id)
    target_ids = load_target_ids(args.robot_config, args.left_id, args.right_id)
    target_dir = args.calibration_root / "robots" / "so_follower"

    with open(source_path) as f:
        source_calibration = json.load(f)

    outputs = {
        "left": (
            target_dir / f"{target_ids['left']}.json",
            extract_arm_calibration(source_calibration, "left"),
        ),
        "right": (
            target_dir / f"{target_ids['right']}.json",
            extract_arm_calibration(source_calibration, "right"),
        ),
    }

    print(f"Source: {source_path}")
    for arm, (target_path, calibration) in outputs.items():
        print(f"{arm.capitalize()} target: {target_path}")
        print(f"  joints: {', '.join(calibration)}")
        if not args.dry_run:
            write_json(target_path, calibration)

    if args.dry_run:
        print("Dry run complete. No files were written.")
    else:
        print("SOFollower calibration files updated.")


if __name__ == "__main__":
    main()
