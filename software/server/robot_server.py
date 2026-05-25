"""Run XLeRobot target control with input controllers."""

from __future__ import annotations

import argparse
import time

from lerobot.utils.visualization_utils import init_rerun

from joycon_command import JoyConCommand
from robot_controller import RobotController


def main():
    parser = argparse.ArgumentParser(description="XLeRobot 2Wheels target-state server")
    parser.add_argument(
        "--log-rerun-data",
        action="store_true",
        help="Enable Rerun visualization logging. Disabled by default.",
    )
    args = parser.parse_args()
    fps = 30
    monitor_interval_s = 1.0

    if args.log_rerun_data:
        init_rerun(session_name="xlerobot_2wheels_teleop_joycon")

    robot_controller = None
    joycon_command = None
    try:
        joycon_command = JoyConCommand(fps=fps, log_rerun=args.log_rerun_data)
        try:
            joycon_command.connect()
        except Exception as exc:
            print(f"[MAIN] Joy-Con startup failed: {exc}")
            return

        robot_controller = RobotController(fps=fps, log_rerun=args.log_rerun_data)
        robot_controller.connect()
        joycon_command.bind_robot_controller(robot_controller)
        print("[MAIN] Loaded recorded arm and head positions. They will return there before sleep/quit.")

        try:
            robot_controller.start_control_loop()
            joycon_command.start()
            while not robot_controller.get_quit_status():
                time.sleep(monitor_interval_s)
        except Exception as exc:
            print(f"[MAIN] Startup/control failed: {exc}")
            if robot_controller is not None:
                print(f"[MAIN] Robot config: {robot_controller.robot_config}")
                print(f"[MAIN] Robot: {robot_controller.robot}")
            return
    except KeyboardInterrupt:
        print("[MAIN] Keyboard interrupt received. Exiting teleoperation...")
    finally:
        if joycon_command is not None:
            joycon_command.disconnect()
        if robot_controller is not None:
            robot_controller.cleanup_session(
                return_to_start=True,
            )
    print("Teleoperation ended.")


if __name__ == "__main__":
    main()
