from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.models.schemas import AgentTraceStep, RuntimeMetrics, TraceRecord


def runtime_db_path() -> Path:
    configured = os.getenv("BLUECHIP_RUNTIME_DB_PATH")
    if configured:
        return Path(configured)
    return Path("runs/bluechip_runtime.sqlite")


def runtime_database_url() -> str:
    return (os.getenv("BLUECHIP_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip()


def _is_postgres_url(database_url: str) -> bool:
    lowered = database_url.lower()
    return lowered.startswith(("postgres://", "postgresql://"))


def _load_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set to Postgres, but psycopg is not installed. "
            "Install the project dependencies or unset DATABASE_URL to use SQLite."
        ) from exc
    return psycopg, dict_row


def _build_trace_record(
    endpoint: str,
    latency_ms: float,
    steps: list[AgentTraceStep],
    generation_provider: str,
    estimated_generation_tokens: int,
    estimated_generation_cost_usd: float = 0.0,
    model_versions: dict[str, str] | None = None,
    index_versions: dict[str, str] | None = None,
    retrieval_source_counts: dict[str, int] | None = None,
    validation_status: str = "",
    fallback_reason: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        trace_id=str(uuid4()),
        endpoint=endpoint,
        created_at=datetime.now(UTC).isoformat(),
        latency_ms=round(latency_ms, 2),
        generation_provider=generation_provider,
        estimated_generation_tokens=estimated_generation_tokens,
        estimated_generation_cost_usd=round(estimated_generation_cost_usd, 6),
        model_versions=model_versions or {},
        index_versions=index_versions or {},
        retrieval_source_counts=retrieval_source_counts or {},
        validation_status=validation_status,
        fallback_reason=fallback_reason,
        steps=steps,
    )


def _metrics_from_records(records: list[TraceRecord]) -> RuntimeMetrics:
    if not records:
        return RuntimeMetrics(requests=0)
    endpoint_counts = Counter(record.endpoint for record in records)
    provider_counts = Counter(record.generation_provider for record in records)
    retrieval_source_counts: Counter[str] = Counter()
    model_version_counts: Counter[str] = Counter()
    index_version_counts: Counter[str] = Counter()
    for record in records:
        retrieval_source_counts.update(record.retrieval_source_counts)
        model_version_counts.update(
            f"{name}={value}" for name, value in record.model_versions.items()
        )
        index_version_counts.update(
            f"{name}={value}" for name, value in record.index_versions.items()
        )
    avg_latency = sum(record.latency_ms for record in records) / len(records)
    tokens = sum(record.estimated_generation_tokens for record in records)
    cost = sum(record.estimated_generation_cost_usd for record in records)
    validation_failures = sum(
        1
        for record in records
        if record.validation_status and record.validation_status not in {"ok", "pass", "passed"}
    )
    fallback_count = sum(1 for record in records if record.fallback_reason)
    return RuntimeMetrics(
        requests=len(records),
        by_endpoint=dict(endpoint_counts),
        by_generation_provider=dict(provider_counts),
        average_latency_ms=round(avg_latency, 2),
        estimated_generation_tokens=tokens,
        estimated_generation_cost_usd=round(cost, 6),
        validation_failures=validation_failures,
        validation_failure_rate=round(validation_failures / len(records), 4),
        fallback_count=fallback_count,
        fallback_rate=round(fallback_count / len(records), 4),
        retrieval_source_counts=dict(sorted(retrieval_source_counts.items())),
        model_version_counts=dict(sorted(model_version_counts.items())),
        index_version_counts=dict(sorted(index_version_counts.items())),
    )


class SQLiteTraceStore:
    """SQLite-backed trace store for persistent request observability."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.getenv("BLUECHIP_TRACE_SQLITE_PATH") or runtime_db_path())
        self._lock = Lock()
        self._init_db()

    def append(
        self,
        endpoint: str,
        latency_ms: float,
        steps: list[AgentTraceStep],
        generation_provider: str,
        estimated_generation_tokens: int,
        estimated_generation_cost_usd: float = 0.0,
        model_versions: dict[str, str] | None = None,
        index_versions: dict[str, str] | None = None,
        retrieval_source_counts: dict[str, int] | None = None,
        validation_status: str = "",
        fallback_reason: str | None = None,
    ) -> TraceRecord:
        record = _build_trace_record(
            endpoint=endpoint,
            latency_ms=latency_ms,
            steps=steps,
            generation_provider=generation_provider,
            estimated_generation_tokens=estimated_generation_tokens,
            estimated_generation_cost_usd=estimated_generation_cost_usd,
            model_versions=model_versions,
            index_versions=index_versions,
            retrieval_source_counts=retrieval_source_counts,
            validation_status=validation_status,
            fallback_reason=fallback_reason,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces(trace_id, created_at, endpoint, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.trace_id,
                    record.created_at,
                    record.endpoint,
                    record.model_dump_json(),
                ),
            )
        return record

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        bounded_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM traces
                ORDER BY id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        records = []
        for row in rows:
            try:
                records.append(TraceRecord(**json.loads(row["payload"])))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
        return records

    def metrics(self) -> RuntimeMetrics:
        return _metrics_from_records(self.recent(limit=10_000))

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_endpoint ON traces(endpoint)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn


class PostgresTraceStore:
    """Postgres-backed trace store for durable hosted observability."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or runtime_database_url()
        if not self.database_url:
            raise ValueError("PostgresTraceStore requires DATABASE_URL or BLUECHIP_DATABASE_URL")
        self._lock = Lock()
        self._psycopg, self._row_factory = _load_psycopg()
        self._init_db()

    def append(
        self,
        endpoint: str,
        latency_ms: float,
        steps: list[AgentTraceStep],
        generation_provider: str,
        estimated_generation_tokens: int,
        estimated_generation_cost_usd: float = 0.0,
        model_versions: dict[str, str] | None = None,
        index_versions: dict[str, str] | None = None,
        retrieval_source_counts: dict[str, int] | None = None,
        validation_status: str = "",
        fallback_reason: str | None = None,
    ) -> TraceRecord:
        record = _build_trace_record(
            endpoint=endpoint,
            latency_ms=latency_ms,
            steps=steps,
            generation_provider=generation_provider,
            estimated_generation_tokens=estimated_generation_tokens,
            estimated_generation_cost_usd=estimated_generation_cost_usd,
            model_versions=model_versions,
            index_versions=index_versions,
            retrieval_source_counts=retrieval_source_counts,
            validation_status=validation_status,
            fallback_reason=fallback_reason,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces(trace_id, created_at, endpoint, payload)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    record.trace_id,
                    record.created_at,
                    record.endpoint,
                    record.model_dump_json(),
                ),
            )
        return record

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        bounded_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM traces
                ORDER BY id DESC
                LIMIT %s
                """,
                (bounded_limit,),
            ).fetchall()
        records = []
        for row in rows:
            try:
                records.append(TraceRecord(**json.loads(row["payload"])))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
        return records

    def metrics(self) -> RuntimeMetrics:
        return _metrics_from_records(self.recent(limit=10_000))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id BIGSERIAL PRIMARY KEY,
                    trace_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_endpoint ON traces(endpoint)")

    def _connect(self) -> Any:
        return self._psycopg.connect(self.database_url, row_factory=self._row_factory)


def build_trace_store() -> SQLiteTraceStore | PostgresTraceStore:
    database_url = runtime_database_url()
    if _is_postgres_url(database_url):
        return PostgresTraceStore(database_url)
    return SQLiteTraceStore()


trace_store = build_trace_store()
