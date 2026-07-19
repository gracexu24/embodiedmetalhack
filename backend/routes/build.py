"""Build control: staged voice-equivalent commands, status, and WebSocket stream."""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, model_validator

from house_builder.models import Color, HouseRequest
from house_builder.parser import parse_house_request

from ..build_runner import BuildAlreadyRunningError, BuildRunner

router = APIRouter()


class BuildRequestBody(BaseModel):
    """Either a natural-language sentence, or explicit door/wall/roof colors."""

    sentence: str | None = None
    door: Color | None = None
    wall: Color | None = None
    roof: Color | None = None

    @model_validator(mode="after")
    def _require_one_input(self) -> BuildRequestBody:
        structured = (self.door, self.wall, self.roof)
        if self.sentence is None and any(value is None for value in structured):
            raise ValueError(
                "Provide either 'sentence', or all three of 'door'/'wall'/'roof'."
            )
        return self


class CommandBody(BaseModel):
    command: str


def _to_house_request(body: BuildRequestBody) -> HouseRequest:
    if body.sentence is not None:
        return parse_house_request(body.sentence)
    assert body.door is not None and body.wall is not None and body.roof is not None
    return HouseRequest(door=body.door, wall=body.wall, roof=body.roof)


@router.post("")
async def start_build(body: BuildRequestBody, request: Request) -> dict[str, str]:
    """One-shot: run the full house build in the background."""
    runner: BuildRunner = request.app.state.build_runner
    try:
        house_request = _to_house_request(body)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        run_id = runner.start_build(house_request)
    except BuildAlreadyRunningError as exc:
        return {"error": str(exc)}
    return {"run_id": run_id}


@router.post("/request")
async def set_request(body: BuildRequestBody, request: Request) -> dict[str, object]:
    """Build this (without camera): store a request so Start can build the door."""
    runner: BuildRunner = request.app.state.build_runner
    try:
        house_request = _to_house_request(body)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        return runner.set_request(house_request)
    except BuildAlreadyRunningError as exc:
        return {"error": str(exc)}


@router.post("/command")
async def run_command(body: CommandBody, request: Request) -> dict[str, object]:
    """UI / voice fallback: start, build wall, build roof, retry last step, stop."""
    runner: BuildRunner = request.app.state.build_runner
    try:
        return runner.handle_command(body.command)
    except BuildAlreadyRunningError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@router.get("/status")
def get_status(request: Request) -> dict[str, object]:
    runner: BuildRunner = request.app.state.build_runner
    return runner.status_event()


@router.websocket("/ws")
async def build_status_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    runner: BuildRunner = websocket.app.state.build_runner
    queue = runner.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        runner.unsubscribe(queue)
