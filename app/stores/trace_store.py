from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.models.schemas import AgentTraceStep, RuntimeMetrics, TraceRecord


def runtime_db_path() -> Path:
    configured = os.getenv("BLUECHIP_RUNTIME_DB_PATH")
    if configured:
        return Path(configured)
    return Path("runs/bluechip_runtime.sqlite")


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
        record = TraceRecord(
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
        records = self.recent(limit=10_000)
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


trace_store = SQLiteTraceStore()
