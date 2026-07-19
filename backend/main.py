"""FastAPI app for the house-builder web dashboard.

Run with: uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import rerun_service
from .build_runner import BuildRunner
from .camera_hub import CameraHub
from .config import load_config
from .routes import build, cam2, highlights
from .routes import rerun as rerun_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DEV_ORIGIN = "http://localhost:5173"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    app.state.config = config

    rerun_endpoints = rerun_service.start()
    app.state.rerun_endpoints = rerun_endpoints

    camera_hub = CameraHub(config["cameras"].get("camera3"))
    camera_hub.start()
    app.state.camera_hub = camera_hub

    app.state.build_runner = BuildRunner(config, rerun_endpoints)

    print("[ui] dashboard backend ready", flush=True)
    print("[ui]   API:          http://localhost:8000 (docs at /docs)", flush=True)
    print(f"[ui]   frontend dev: {FRONTEND_DEV_ORIGIN} (npm run dev in frontend/)", flush=True)
    print(f"[ui]   Rerun viewer: {rerun_endpoints.viewer_url}", flush=True)
    print(f"[ui]   Rerun gRPC:   {rerun_endpoints.grpc_uri}", flush=True)
    print(
        f"[ui]   camera3:      {'available' if camera_hub.available else 'UNAVAILABLE'} "
        f"(config: {config['cameras'].get('camera3')})",
        flush=True,
    )
    print(f"[ui]   robot dry_run: {config['robot'].get('dry_run', False)}", flush=True)
    print(f"[ui]   policy server: {config['policy'].get('server', 'n/a')}", flush=True)

    yield

    camera_hub.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="House Builder Dashboard", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_DEV_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(build.router, prefix="/api/build", tags=["build"])
    app.include_router(cam2.router, prefix="/api/cam2", tags=["cam2"])
    app.include_router(rerun_routes.router, prefix="/api/rerun", tags=["rerun"])
    app.include_router(highlights.router, prefix="/api/highlights", tags=["highlights"])
    return app


app = create_app()
