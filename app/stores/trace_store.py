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


trace_store = JsonlTraceStore()
