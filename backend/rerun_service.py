"""Starts Rerun's gRPC log sink and web viewer once per process.

The dashboard embeds the resulting web-viewer URL in an <iframe>, reusing the same
rr.log(...) calls and Blueprint (src/house_builder/rr_blueprint.py) that already power
the native desktop viewer used by run.py -- cam0, cam1, arm joints, and the state/
verification text log all show up for free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote

import rerun as rr

from house_builder.rr_blueprint import build_blueprint

log = logging.getLogger(__name__)

# The gRPC server is tied to a RecordingStream and shuts down if that stream's sinks are
# replaced. build_runner calls rr.set_sinks(...) per run (to tee a file sink alongside the
# live feed), which would kill a server owned by the global recording -- so we give the
# server its own dedicated recording and keep a module-level reference so it isn't GC'd.
_server_recording: "rr.RecordingStream | None" = None

WEB_VIEWER_PORT = 9090
# gRPC log-sink port. Rerun defaults to 9876; we pin a different port so a stray
# Rerun instance from another project (e.g. so100-hackathon) can't collide with
# ours and silently steal the feed.
GRPC_PORT = 9878


@dataclass(frozen=True)
class RerunEndpoints:
    grpc_uri: str
    viewer_url: str


def start() -> RerunEndpoints:
    """Initialize Rerun and serve it over gRPC + a web viewer.

    Returns both the gRPC URI (so build_runner can attach a per-run file sink
    alongside it) and the browser-facing viewer URL (for the dashboard iframe).
    """
    global _server_recording
    blueprint = build_blueprint()
    rr.init("house_builder", spawn=False, default_blueprint=blueprint)
    # Own the gRPC server with a dedicated recording so it survives build_runner's per-run
    # rr.set_sinks() on the global recording. Tying the server to the global recording (as
    # before) meant the first build's set_sinks shut the server down -- the viewer would
    # then show "Welcome to Rerun" and the client sink logged a gRPC transport error.
    _server_recording = rr.RecordingStream("house_builder")
    # Pass the blueprint to serve_grpc as well: a default_blueprint set only on rr.init is
    # not reliably buffered for late-connecting viewers, so the web viewer would otherwise
    # open with no panels until data arrives.
    grpc_uri = rr.serve_grpc(
        grpc_port=GRPC_PORT, default_blueprint=blueprint, recording=_server_recording
    )
    # open_browser=False: we only want the static web-viewer page hosted, not an
    # OS browser tab opened server-side. The dashboard iframe supplies its own
    # `?url=` to connect, which is why we build viewer_url ourselves below.
    rr.serve_web_viewer(web_port=WEB_VIEWER_PORT, open_browser=False, connect_to=grpc_uri)
    viewer_url = f"http://localhost:{WEB_VIEWER_PORT}/?url={quote(grpc_uri, safe='')}"
    log.info("Rerun web viewer ready at %s (grpc: %s)", viewer_url, grpc_uri)
    return RerunEndpoints(grpc_uri=grpc_uri, viewer_url=viewer_url)
