from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import RLock
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Small in-process sliding-window limiter for public API protection."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_window: int = 120,
        window_seconds: int = 60,
        key_func: Callable[[Request], str] | None = None,
    ) -> None:
        super().__init__(app)
        self.requests_per_window = max(1, requests_per_window)
        self.window_seconds = max(1, window_seconds)
        self.key_func = key_func or client_ip_key
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()
        self._exempt_paths = _exempt_paths()

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._is_exempt(request):
            return await call_next(request)

        now = time.monotonic()
        key = self.key_func(request)
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.requests_per_window:
                retry_after = max(1, int(self.window_seconds - (now - hits[0])))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "rate limit exceeded",
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.requests_per_window),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            hits.append(now)
            remaining = max(0, self.requests_per_window - len(hits))

        response = await call_next(request)
        response.headers.setdefault("X-RateLimit-Limit", str(self.requests_per_window))
        response.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        return response

    def _is_exempt(self, request: Request) -> bool:
        path = request.url.path
        if not path.startswith("/api"):
            return True
        return any(path == exempt or path.startswith(f"{exempt}/") for exempt in self._exempt_paths)


def client_ip_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_enabled() -> bool:
    configured = os.getenv("BLUECHIP_RATE_LIMIT_ENABLED", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}
    return os.getenv("APP_ENV", "").strip().lower() == "production"


def rate_limit_config() -> tuple[int, int]:
    return (
        _int_env("BLUECHIP_RATE_LIMIT_REQUESTS", 120),
        _int_env("BLUECHIP_RATE_LIMIT_WINDOW_SECONDS", 60),
    )


def _exempt_paths() -> tuple[str, ...]:
    raw = os.getenv("BLUECHIP_RATE_LIMIT_EXEMPT_PATHS", "/api/health")
    paths = tuple(path.strip().rstrip("/") for path in raw.split(",") if path.strip())
    return paths or ("/api/health",)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
