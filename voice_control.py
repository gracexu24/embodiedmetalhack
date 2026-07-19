#!/usr/bin/env python3
"""Voice-command controller for staged house building."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from house_builder.builder import HouseBuilder
from house_builder.models import HouseRequest, Layer
from house_builder.policy import MolmoAct2Policy
from house_builder.robot import SO101Robot
from house_builder.verifier import PlacementVerifier
from human_builder import capture_model_house, request_to_sentence
from run import load_config


class VoiceCommand(str, Enum):
    BUILD_THIS = "build this"
    START = "start"
    BUILD_WALL = "build wall"
    BUILD_ROOF = "build roof"
    STOP = "stop"
    UNKNOWN = "unknown"


def parse_voice_command(transcript: str) -> VoiceCommand:
    """Normalize a transcript and select one supported command."""
    normalized = re.sub(r"[^a-z]+", " ", transcript.lower()).strip()
    aliases = {
        "build this": VoiceCommand.BUILD_THIS,
        "start": VoiceCommand.START,
        "build wall": VoiceCommand.BUILD_WALL,
        "build the wall": VoiceCommand.BUILD_WALL,
        "build roof": VoiceCommand.BUILD_ROOF,
        "build the roof": VoiceCommand.BUILD_ROOF,
        "stop": VoiceCommand.STOP,
        "quit": VoiceCommand.STOP,
        "exit": VoiceCommand.STOP,
    }
    return aliases.get(normalized, VoiceCommand.UNKNOWN)


class VoiceBuildController:
    """Apply voice commands to one staged HouseBuilder session."""

    def __init__(
        self,
        builder: HouseBuilder,
        request_provider: Callable[[], HouseRequest],
        output: Callable[[str], None] = print,
    ) -> None:
        self.builder = builder
        self.request_provider = request_provider
        self.output = output
        self.request: HouseRequest | None = None

    def handle(self, transcript: str) -> bool:
        """Handle one transcript; return False when listening should stop."""
        command = parse_voice_command(transcript)
        self.output(f'Heard: "{transcript}"')

        if command is VoiceCommand.BUILD_THIS:
            if self.builder.session_active:
                self.output("A build is already active. Say stop before scanning again.")
                return True
            self.request = self.request_provider()
            self.output(f"Ready: {request_to_sentence(self.request)}")
            self.output('Say "start" to build the door layer.')
            return True

        if command is VoiceCommand.START:
            if self.request is None:
                self.output('Say "build this" before starting.')
                return True
            if not self.builder.session_active:
                self.builder.prepare(self.request)
            result = self.builder.build_layer(Layer.DOOR)
            self.output(result.message)
            if result.success:
                self.output('Say "build wall" to continue.')
            return True

        if command is VoiceCommand.BUILD_WALL:
            if not self._ready_for(Layer.DOOR, "start"):
                return True
            result = self.builder.build_layer(Layer.WALL)
            self.output(result.message)
            if result.success:
                self.output('Say "build roof" to continue.')
            return True

        if command is VoiceCommand.BUILD_ROOF:
            if not self._ready_for(Layer.WALL, "build wall"):
                return True
            result = self.builder.build_layer(Layer.ROOF)
            self.output(result.message)
            return True

        if command is VoiceCommand.STOP:
            self.builder.close()
            self.output("Build stopped safely.")
            return False

        self.output('Unknown command. Say "build this", "start", "build wall", or "build roof".')
        return True

    def _ready_for(self, required_layer: Layer, required_command: str) -> bool:
        if not self.builder.session_active or required_layer not in self.builder.completed_layers:
            self.output(f'Say "{required_command}" first.')
            return False
        return True


def _create_builder(config_path: Path) -> tuple[HouseBuilder, dict[str, Any]]:
    config = load_config(config_path)
    robot = SO101Robot(config["robot"])
    verifier = PlacementVerifier(config["verification"], config["cameras"])
    policy = MolmoAct2Policy(
        config["policy"],
        robot,
        verifier.camera_observations,
    )
    builder = HouseBuilder(
        robot,
        policy,
        verifier,
        float(config["policy"]["skill_duration_seconds"]),
        float(config["policy"].get("check_interval_seconds", 3.0)),
    )
    return builder, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Control staged house building by voice.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--text", action="store_true", help="Type commands instead of using a mic")
    parser.add_argument("--microphone-index", type=int)
    args = parser.parse_args()

    builder, config = _create_builder(args.config)
    controller = VoiceBuildController(
        builder,
        lambda: capture_model_house(config),
    )
    try:
        if args.text:
            while controller.handle(input("Command: ")):
                pass
            return 0

        try:
            import speech_recognition as sr
        except ImportError:
            print('Voice dependencies are missing. Install with: pip install -e ".[voice]"')
            return 1

        recognizer = sr.Recognizer()
        microphone = sr.Microphone(device_index=args.microphone_index)
        with microphone as source:
            print("Calibrating microphone noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

        print('Listening. Say "build this", "start", "build wall", "build roof", or "stop".')
        while True:
            try:
                with microphone as source:
                    audio = recognizer.listen(source, phrase_time_limit=4)
                transcript = recognizer.recognize_google(audio)
                if not controller.handle(transcript):
                    return 0
            except sr.UnknownValueError:
                print("Could not understand that command.")
            except sr.RequestError as exc:
                print(f"Speech recognition service failed: {exc}")
                return 1
    except (EOFError, KeyboardInterrupt):
        print("\nStopping.")
        return 130
    except Exception as exc:
        print(f"Voice build aborted safely: {exc}")
        return 1
    finally:
        builder.close()


if __name__ == "__main__":
    raise SystemExit(main())
