from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Bluechip User Intelligence Agent",
    description="Review simulation and personalized recommendation API.",
    version="0.1.0",
)

app.include_router(router, prefix="/api")
