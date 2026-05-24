"""
Discord control server for XLeRobot.

Listens to one Discord channel and responds to robot control commands.
Initial commands:
    sleep  - stop the base and disable motor torque
    wake   - enable/configure motor torque
    status - report whether the robot is awake or sleeping

Example:
    PYTHONPATH=software/src python software/server/discord_control.py \
        --token YOUR_DISCORD_BOT_TOKEN \
        --channel-id 123456789012345678
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import discord

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig


logger = logging.getLogger("discord_control")


@dataclass
class DiscordControlConfig:
    token: str
    channel_id: int
    robot_type: str
    robot_id: str
    port1: str
    port2: str
    command_prefix: str

    @classmethod
    def from_args(cls) -> "DiscordControlConfig":
        parser = argparse.ArgumentParser(description="Control XLeRobot from a Discord channel")
        parser.add_argument(
            "--token",
            default=os.getenv("DISCORD_APP_TOKEN"),
            help="Discord bot/app token. Defaults to DISCORD_APP_TOKEN.",
        )
        parser.add_argument(
            "--channel-id",
            type=int,
            default=_optional_int_env("DISCORD_CHANNEL_ID"),
            help="Discord channel ID to listen to. Defaults to DISCORD_CHANNEL_ID.",
        )
        parser.add_argument(
            "--robot-type",
            choices=("xlerobot_2wheels", "xlerobot"),
            default=os.getenv("XLE_DISCORD_ROBOT_TYPE", "xlerobot_2wheels"),
            help="Robot class to control.",
        )
        parser.add_argument(
            "--robot-id",
            default=os.getenv("XLE_DISCORD_ROBOT_ID", "my_xlerobot_2wheels"),
            help="Robot calibration/config ID.",
        )
        parser.add_argument(
            "--port1",
            default=os.getenv("XLE_DISCORD_PORT1", "/dev/ttyACM0"),
            help="First motor bus serial port.",
        )
        parser.add_argument(
            "--port2",
            default=os.getenv("XLE_DISCORD_PORT2", "/dev/ttyACM1"),
            help="Second motor bus serial port.",
        )
        parser.add_argument(
            "--command-prefix",
            default=os.getenv("DISCORD_COMMAND_PREFIX", ""),
            help="Optional command prefix, for example '!'. Defaults to no prefix.",
        )
        args = parser.parse_args()

        if not args.token:
            raise ValueError("Discord token is required via --token or DISCORD_APP_TOKEN")
        if args.channel_id is None:
            raise ValueError("Discord channel ID is required via --channel-id or DISCORD_CHANNEL_ID")

        return cls(
            token=args.token,
            channel_id=args.channel_id,
            robot_type=args.robot_type,
            robot_id=args.robot_id,
            port1=args.port1,
            port2=args.port2,
            command_prefix=args.command_prefix.strip().lower(),
        )


def _optional_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


class RobotSleepController:
    def __init__(self, config: DiscordControlConfig):
        self.config = config
        self.robot = self._make_robot(config)
        self.sleeping = False

    def _make_robot(self, config: DiscordControlConfig):
        if config.robot_type == "xlerobot":
            robot_config = XLerobotConfig(id=config.robot_id, port1=config.port1, port2=config.port2)
            return XLerobot(robot_config)

        robot_config = XLerobot2WheelsConfig(id=config.robot_id, port1=config.port1, port2=config.port2)
        return XLerobot2Wheels(robot_config)

    async def connect(self) -> None:
        await asyncio.to_thread(self.robot.connect)
        self.sleeping = False

    async def sleep(self) -> str:
        if self.sleeping:
            return "Already sleeping."

        await asyncio.to_thread(self._sleep_robot)
        self.sleeping = True
        return "Sleeping. Base stopped and motor torque disabled."

    async def wake(self) -> str:
        if not self.sleeping:
            return "Already awake."

        await asyncio.to_thread(self._wake_robot)
        self.sleeping = False
        return "Awake. Motor torque enabled."

    async def status(self) -> str:
        mode = "sleeping" if self.sleeping else "awake"
        connected = "connected" if self.robot.is_connected else "disconnected"
        return f"Status: {mode}, robot {connected}, type {self.config.robot_type}."

    async def disconnect(self) -> None:
        if self.robot.is_connected:
            await asyncio.to_thread(self.robot.disconnect)

    def _sleep_robot(self) -> None:
        self.robot.stop_base()
        self.robot.bus1.disable_torque()
        self.robot.bus2.disable_torque()

    def _wake_robot(self) -> None:
        self.robot.configure()


class DiscordRobotClient(discord.Client):
    def __init__(self, config: DiscordControlConfig, controller: RobotSleepController):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config = config
        self.controller = controller
        self._command_lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        await self.controller.connect()

    async def close(self) -> None:
        await self.controller.disconnect()
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s", self.user)
        channel = self.get_channel(self.config.channel_id)
        if channel is None:
            logger.warning("Channel %s is not visible to this bot", self.config.channel_id)
            return
        await channel.send("XLeRobot Discord control online. Commands: sleep, wake, status.")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return
        if message.channel.id != self.config.channel_id:
            return

        command = self._parse_command(message.content)
        if command is None:
            return

        async with self._command_lock:
            if command == "sleep":
                response = await self.controller.sleep()
            elif command == "wake":
                response = await self.controller.wake()
            elif command == "status":
                response = await self.controller.status()
            else:
                response = "Unknown command. Supported commands: sleep, wake, status."

        await message.channel.send(response)

    def _parse_command(self, content: str) -> Optional[str]:
        normalized = content.strip().lower()
        if not normalized:
            return None

        if self.config.command_prefix:
            if not normalized.startswith(self.config.command_prefix):
                return None
            normalized = normalized[len(self.config.command_prefix) :].strip()

        return normalized.split(maxsplit=1)[0] if normalized else None


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = DiscordControlConfig.from_args()
    controller = RobotSleepController(config)
    client = DiscordRobotClient(config, controller)
    await client.start(config.token)


if __name__ == "__main__":
    asyncio.run(main())
