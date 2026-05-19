from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Bluechip User Intelligence Agent",
    description="Review simulation and personalized recommendation API.",
    version="0.1.0",
)

app.include_router(router, prefix="/api")

ui_dir = Path(__file__).resolve().parents[1] / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")
