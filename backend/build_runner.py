"""Runs HouseBuilder.build() in a background thread and fans out state-machine
transitions to WebSocket subscribers, while also saving a per-run .rrd recording
for the highlights (Query API) endpoint to read afterward.
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
from house_builder.models import HouseRequest
from house_builder.policy import MockPolicy, MolmoAct2Policy, Policy
from house_builder.robot import MockRobot, Robot, SO101Robot
from house_builder.state_machine import BuildState
from house_builder.verifier import PlacementVerifier

from .highlights import RECORDINGS_DIR, recording_path
from .rerun_service import RerunEndpoints

log = logging.getLogger(__name__)

_TERMINAL_STATES = {BuildState.COMPLETED.value, BuildState.FAILED.value}


class BuildAlreadyRunningError(RuntimeError):
    pass


class BuildRunner:
    """Owns at most one build at a time -- run.py already assumes single-build hardware use."""

    def __init__(self, config: dict[str, Any], rerun_endpoints: RerunEndpoints) -> None:
        self.config = config
        self._rerun_endpoints = rerun_endpoints
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="house-build")
        self._lock = threading.Lock()
        self.run_id: str | None = None
        self.state: str = BuildState.IDLE.value
        self.history: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
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
        }

    def _broadcast(self, event: dict[str, Any]) -> None:
        if self._loop is None:
            return
        for queue in list(self._subscribers):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)

    def start_build(self, request: HouseRequest) -> str:
        with self._lock:
            if self.run_id is not None and self.state not in _TERMINAL_STATES | {
                BuildState.IDLE.value
            }:
                raise BuildAlreadyRunningError("A build is already in progress.")
            self._loop = asyncio.get_running_loop()
            run_id = uuid.uuid4().hex[:12]
            self.run_id = run_id
            self.state = BuildState.IDLE.value
            self.history = []
            self.result = None
        self._executor.submit(self._run, run_id, request)
        self._broadcast(self.status_event())
        return run_id

    def _on_state_change(self, previous: BuildState, next_state: BuildState) -> None:
        self.state = next_state.value
        entry = {"from": previous.value, "to": next_state.value, "step": len(self.history)}
        self.history.append(entry)
        self._broadcast({"type": "transition", "run_id": self.run_id, **entry})

    def _run(self, run_id: str, request: HouseRequest) -> None:
        self._attach_recording_sink(run_id)

        mock = bool(self.config.get("mock_mode", False))
        robot: Robot = MockRobot() if mock else SO101Robot(self.config["robot"])
        verifier = PlacementVerifier(self.config["verification"], self.config["cameras"], mock=mock)
        policy: Policy
        if mock:
            policy = MockPolicy()
        else:
            policy = MolmoAct2Policy(self.config["policy"], robot, verifier.camera_observations)
        builder = HouseBuilder(
            robot,
            policy,
            verifier,
            float(self.config["policy"]["skill_duration_seconds"]),
            on_state_change=self._on_state_change,
        )
        try:
            result = builder.build(request)
            self.result = {
                "success": result.success,
                "completed_layers": [layer.value for layer in result.completed_layers],
                "failed_layer": result.failed_layer.value if result.failed_layer else None,
                "message": result.message,
            }
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the dashboard
            log.exception("Build %s aborted with an unhandled exception", run_id)
            self.result = {
                "success": False,
                "completed_layers": [],
                "failed_layer": None,
                "message": str(exc),
            }
        finally:
            self._broadcast({"type": "result", "run_id": run_id, "result": self.result})

    def _attach_recording_sink(self, run_id: str) -> None:
        """Tee live logging to a per-run file, alongside the already-running gRPC
        server the dashboard's Rerun iframe is connected to, so the highlights
        endpoint has something to query once the build finishes.
        """
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
