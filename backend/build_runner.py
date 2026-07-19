"""Runs staged HouseBuilder commands in a background thread and fans out
state-machine transitions to WebSocket subscribers.

Commands mirror voice_control.py: build_this, start, build_wall, build_roof,
retry_last_step, and stop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import rerun as rr

from house_builder.builder import HouseBuilder
from house_builder.models import HouseRequest, Layer
from house_builder.policy import MolmoAct2Policy
from house_builder.robot import SO101Robot
from house_builder.state_machine import BuildState
from house_builder.verifier import PlacementVerifier
from human_builder import detect_model_house, request_to_sentence

from .highlights import RECORDINGS_DIR, recording_path
from .rerun_service import RerunEndpoints

log = logging.getLogger(__name__)

_BUSY_STATES = {
    BuildState.CONNECTING.value,
    BuildState.HOMING.value,
    BuildState.EXECUTING.value,
    BuildState.VERIFYING.value,
}


class BuildAlreadyRunningError(RuntimeError):
    pass


class BuildRunner:
    """Owns at most one staged build session at a time."""

    def __init__(self, config: dict[str, Any], rerun_endpoints: RerunEndpoints) -> None:
        self.config = config
        self._rerun_endpoints = rerun_endpoints
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="house-build")
        self._lock = threading.Lock()
        self.run_id: str | None = None
        self.state: str = BuildState.IDLE.value
        self.history: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.request: HouseRequest | None = None
        self.request_sentence: str | None = None
        self.completed_layers: list[str] = []
        self.failed_layer: str | None = None
        self._builder: HouseBuilder | None = None
        self._busy = False
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        queue.put_nowait(self.status_event())
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def status_event(self) -> dict[str, Any]:
        return {
            "type": "status",
            "run_id": self.run_id,
            "state": self.state,
            "history": list(self.history),
            "result": self.result,
            "request_sentence": self.request_sentence,
            "completed_layers": list(self.completed_layers),
            "failed_layer": self.failed_layer,
            "session_active": self._builder is not None and self._builder.session_active,
            "busy": self._busy,
        }

    def _broadcast(self, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        for queue in list(self._subscribers):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)

    def _on_state_change(self, previous: BuildState, next_state: BuildState) -> None:
        self.state = next_state.value
        entry = {"from": previous.value, "to": next_state.value, "step": len(self.history)}
        self.history.append(entry)
        self._broadcast({"type": "transition", "run_id": self.run_id, **entry})
        self._broadcast(self.status_event())

    def set_request(self, request: HouseRequest) -> dict[str, Any]:
        """Store a house request from Build this / reference scan (no robot motion)."""
        with self._lock:
            if self._busy or self.state in _BUSY_STATES:
                raise BuildAlreadyRunningError("A build command is already running.")
            if self._builder is not None and self._builder.session_active:
                raise BuildAlreadyRunningError(
                    "A build session is already active. Stop before scanning again."
                )
            self._loop = asyncio.get_running_loop()
            self.request = request
            self.request_sentence = request_to_sentence(request)
            self.result = None
            self.failed_layer = None
            self.completed_layers = []
            if self.run_id is None:
                self.run_id = uuid.uuid4().hex[:12]
                self.history = []
                self.state = BuildState.IDLE.value
        print(f"[ui] request stored (run {self.run_id}): {self.request_sentence}", flush=True)
        self._broadcast(self.status_event())
        return {
            "run_id": self.run_id,
            "request_sentence": self.request_sentence,
            "door": request.door.value,
            "wall": request.wall.value,
            "roof": request.roof.value,
        }

    def handle_command(self, command: str) -> dict[str, Any]:
        """Queue one staged voice-equivalent command."""
        normalized = command.strip().lower().replace("_", " ")
        aliases = {
            "build this": "build_this",
            "start": "start",
            "build wall": "build_wall",
            "build the wall": "build_wall",
            "build roof": "build_roof",
            "build the roof": "build_roof",
            "retry last step": "retry_last_step",
            "retry the last step": "retry_last_step",
            "retry": "retry_last_step",
            "stop": "stop",
        }
        action = aliases.get(normalized)
        if action is None:
            print(f"[ui] rejected unknown command: {command!r}", flush=True)
            return {"error": f"Unknown command: {command}"}

        print(
            f"[ui] command {action!r} received "
            f"(state={self.state}, completed={self.completed_layers}, "
            f"failed={self.failed_layer}, busy={self._busy})",
            flush=True,
        )
        with self._lock:
            if self._busy and action != "stop":
                print(f"[ui] command {action!r} rejected: a command is already running", flush=True)
                raise BuildAlreadyRunningError("A build command is already running.")
            self._loop = asyncio.get_running_loop()
            if action == "stop":
                self._busy = False
                self._executor.submit(self._stop)
                return {"accepted": True, "command": "stop"}
            self._busy = True

        self._executor.submit(self._run_command, action)
        self._broadcast(self.status_event())
        return {"accepted": True, "command": action}

    def start_build(self, request: HouseRequest) -> str:
        """One-shot convenience: set request then run all layers."""
        self.set_request(request)
        with self._lock:
            if self._busy:
                raise BuildAlreadyRunningError("A build command is already running.")
            self._loop = asyncio.get_running_loop()
            self._busy = True
            run_id = self.run_id or uuid.uuid4().hex[:12]
            self.run_id = run_id
        self._executor.submit(self._run_oneshot)
        self._broadcast(self.status_event())
        return run_id

    def _ensure_builder(self) -> HouseBuilder:
        if self._builder is None or not self._builder.session_active:
            if self.request is None:
                raise RuntimeError('Say "build this" / scan the model house before starting.')
            self._attach_recording_sink(self.run_id or "session")
            robot = SO101Robot(self.config["robot"])
            verifier = PlacementVerifier(self.config["verification"], self.config["cameras"])
            policy = MolmoAct2Policy(
                self.config["policy"],
                robot,
                verifier.camera_observations,
            )
            self._builder = HouseBuilder(
                robot,
                policy,
                verifier,
                float(self.config["policy"]["skill_duration_seconds"]),
                float(self.config["policy"].get("check_interval_seconds", 3.0)),
                on_state_change=self._on_state_change,
            )
            self._builder.prepare(self.request)
        return self._builder

    def _result_payload(self, result: Any) -> dict[str, Any]:
        payload = {
            "success": result.success,
            "completed_layers": [layer.value for layer in result.completed_layers],
            "failed_layer": result.failed_layer.value if result.failed_layer else None,
            "message": result.message,
        }
        self.result = payload
        self.completed_layers = payload["completed_layers"]
        self.failed_layer = payload["failed_layer"]
        if self._builder is not None:
            self.failed_layer = (
                self._builder.failed_layer.value if self._builder.failed_layer else None
            )
            self.completed_layers = [layer.value for layer in self._builder.completed_layers]
        return payload

    def _run_command(self, action: str) -> None:
        try:
            if action == "build_this":
                raise RuntimeError(
                    "build_this must be handled via /api/build/command after a scan, "
                    "or POST /api/cam2/scan."
                )
            if action == "start":
                if self.request is None:
                    raise RuntimeError('Scan the model house / set a request before "start".')
                builder = self._ensure_builder()
                result = builder.build_layer(Layer.DOOR)
                self._result_payload(result)
            elif action == "build_wall":
                builder = self._ensure_builder()
                result = builder.build_layer(Layer.WALL)
                self._result_payload(result)
            elif action == "build_roof":
                builder = self._ensure_builder()
                result = builder.build_layer(Layer.ROOF)
                self._result_payload(result)
            elif action == "retry_last_step":
                if self._builder is None:
                    raise RuntimeError("There is no failed step to retry.")
                print(
                    f"[ui] retrying failed layer: {self.failed_layer}",
                    flush=True,
                )
                result = self._builder.retry_last_step()
                self._result_payload(result)
            if self.result is not None:
                print(
                    f"[ui] command {action!r} finished: "
                    f"success={self.result['success']}, "
                    f"completed={self.result['completed_layers']}, "
                    f"failed={self.result['failed_layer']}, "
                    f"message={self.result['message']!r}",
                    flush=True,
                )
            self._broadcast({"type": "result", "run_id": self.run_id, "result": self.result})
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the dashboard
            log.exception("Command %s failed", action)
            print(f"[ui] command {action!r} raised: {exc}", flush=True)
            self.result = {
                "success": False,
                "completed_layers": list(self.completed_layers),
                "failed_layer": self.failed_layer,
                "message": str(exc),
            }
            self._broadcast({"type": "result", "run_id": self.run_id, "result": self.result})
        finally:
            self._busy = False
            if self._builder is not None and not self._builder.session_active:
                self._builder = None
            self._broadcast(self.status_event())

    def _run_oneshot(self) -> None:
        try:
            assert self.request is not None
            self._attach_recording_sink(self.run_id or "oneshot")
            robot = SO101Robot(self.config["robot"])
            verifier = PlacementVerifier(self.config["verification"], self.config["cameras"])
            policy = MolmoAct2Policy(
                self.config["policy"],
                robot,
                verifier.camera_observations,
            )
            builder = HouseBuilder(
                robot,
                policy,
                verifier,
                float(self.config["policy"]["skill_duration_seconds"]),
                float(self.config["policy"].get("check_interval_seconds", 3.0)),
                on_state_change=self._on_state_change,
            )
            self._builder = builder
            result = builder.build(self.request)
            self._result_payload(result)
            self._broadcast({"type": "result", "run_id": self.run_id, "result": self.result})
        except Exception as exc:  # noqa: BLE001
            log.exception("One-shot build aborted")
            self.result = {
                "success": False,
                "completed_layers": [],
                "failed_layer": None,
                "message": str(exc),
            }
            self._broadcast({"type": "result", "run_id": self.run_id, "result": self.result})
        finally:
            self._busy = False
            self._builder = None
            self._broadcast(self.status_event())

    def _stop(self) -> None:
        print("[ui] stop requested: closing build session", flush=True)
        try:
            if self._builder is not None:
                self._builder.close()
        finally:
            self._builder = None
            self._busy = False
            self.state = BuildState.IDLE.value
            self.failed_layer = None
            self.result = {
                "success": False,
                "completed_layers": list(self.completed_layers),
                "failed_layer": None,
                "message": "Build stopped safely.",
            }
            self._broadcast({"type": "result", "run_id": self.run_id, "result": self.result})
            self._broadcast(self.status_event())

    def detect_from_frame(self, frame: Any) -> HouseRequest:
        return detect_model_house(frame, self.config["human_builder"])

    def _attach_recording_sink(self, run_id: str) -> None:
        try:
            rr.set_sinks(
                rr.GrpcSink(url=self._rerun_endpoints.grpc_uri),
                rr.FileSink(str(recording_path(run_id))),
            )
        except Exception:
            log.warning(
                "Could not attach a per-run recording sink for %s; highlights will be "
                "unavailable for this run.",
                run_id,
                exc_info=True,
            )
