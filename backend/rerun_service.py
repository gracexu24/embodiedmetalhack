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

WEB_VIEWER_PORT = 9090


@dataclass(frozen=True)
class RerunEndpoints:
    grpc_uri: str
    viewer_url: str


def start() -> RerunEndpoints:
    """Initialize Rerun and serve it over gRPC + a web viewer.

    Returns both the gRPC URI (so build_runner can attach a per-run file sink
    alongside it) and the browser-facing viewer URL (for the dashboard iframe).
    """
    rr.init("house_builder", spawn=False, default_blueprint=build_blueprint())
    grpc_uri = rr.serve_grpc()
    # open_browser=False: we only want the static web-viewer page hosted, not an
    # OS browser tab opened server-side. The dashboard iframe supplies its own
    # `?url=` to connect, which is why we build viewer_url ourselves below.
    rr.serve_web_viewer(web_port=WEB_VIEWER_PORT, open_browser=False, connect_to=grpc_uri)
    viewer_url = f"http://localhost:{WEB_VIEWER_PORT}/?url={quote(grpc_uri, safe='')}"
    log.info("Rerun web viewer ready at %s (grpc: %s)", viewer_url, grpc_uri)
    return RerunEndpoints(grpc_uri=grpc_uri, viewer_url=viewer_url)
