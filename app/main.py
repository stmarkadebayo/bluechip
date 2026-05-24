from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.middleware.rate_limit import RateLimitMiddleware, rate_limit_config, rate_limit_enabled

app = FastAPI(
    title="Bluechip User Intelligence Agent",
    description="Review simulation and personalized recommendation API.",
    version="0.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("BLUECHIP_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

if rate_limit_enabled():
    requests_per_window, window_seconds = rate_limit_config()
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )

app.include_router(router, prefix="/api")

ui_dir = Path(__file__).resolve().parents[1] / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False, response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_file = ui_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UI Index File Not Found</h1>", status_code=404)
