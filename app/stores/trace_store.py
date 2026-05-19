from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.models.schemas import AgentTraceStep, RuntimeMetrics, TraceRecord


class JsonlTraceStore:
    """Append-only local trace store for demo observability.

    Production deployments should send these records to tracing/analytics
    infrastructure. The local JSONL format keeps the submission inspectable.
    """

    def __init__(self, path: Path = Path("runs/traces/requests.jsonl")) -> None:
        self.path = path
        self._lock = Lock()

    def append(
        self,
        endpoint: str,
        latency_ms: float,
        steps: list[AgentTraceStep],
        generation_provider: str,
        estimated_generation_tokens: int,
        estimated_generation_cost_usd: float = 0.0,
    ) -> TraceRecord:
        record = TraceRecord(
            trace_id=str(uuid4()),
            endpoint=endpoint,
            created_at=datetime.now(UTC).isoformat(),
            latency_ms=round(latency_ms, 2),
            generation_provider=generation_provider,
            estimated_generation_tokens=estimated_generation_tokens,
            estimated_generation_cost_usd=round(estimated_generation_cost_usd, 6),
            steps=steps,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        return record

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        rows = self.path.read_text(encoding="utf-8").splitlines()
        records = [TraceRecord(**json.loads(row)) for row in rows[-limit:] if row.strip()]
        records.reverse()
        return records

    def metrics(self) -> RuntimeMetrics:
        records = self.recent(limit=10_000)
        if not records:
            return RuntimeMetrics(requests=0)
        endpoint_counts = Counter(record.endpoint for record in records)
        avg_latency = sum(record.latency_ms for record in records) / len(records)
        tokens = sum(record.estimated_generation_tokens for record in records)
        cost = sum(record.estimated_generation_cost_usd for record in records)
        return RuntimeMetrics(
            requests=len(records),
            by_endpoint=dict(endpoint_counts),
            average_latency_ms=round(avg_latency, 2),
            estimated_generation_tokens=tokens,
            estimated_generation_cost_usd=round(cost, 6),
        )


trace_store = JsonlTraceStore()
