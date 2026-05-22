from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from app.models.schemas import RecommendationItem, UserProfile
from app.stores.trace_store import runtime_db_path


@dataclass
class TurnRecord:
    turn_index: int
    user_message: str
    agent_response: str
    recommendations: list[RecommendationItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "agent_response": self.agent_response,
            "recommendations": [
                recommendation.model_dump(mode="json") for recommendation in self.recommendations
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TurnRecord":
        recommendations = []
        for row in payload.get("recommendations") or []:
            try:
                recommendations.append(RecommendationItem(**row))
            except (TypeError, ValueError):
                continue
        return cls(
            turn_index=int(payload.get("turn_index") or 0),
            user_message=str(payload.get("user_message") or ""),
            agent_response=str(payload.get("agent_response") or ""),
            recommendations=recommendations,
            metadata=dict(payload.get("metadata") or {}),
            timestamp=float(payload.get("timestamp") or time.time()),
        )


@dataclass
class ConversationState:
    conversation_id: str
    user_profile: UserProfile | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    context: str = ""
    accepted_recommendations: list[str] = field(default_factory=list)
    rejected_recommendations: list[str] = field(default_factory=list)
    preference_refinements: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    _lock: RLock = field(default_factory=RLock, repr=False)
    _on_change: Callable[["ConversationState"], None] | None = field(default=None, repr=False)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def bind_persistence(self, callback: Callable[["ConversationState"], None]) -> None:
        self._on_change = callback

    def add_turn(
        self,
        user_message: str,
        agent_response: str,
        recommendations: list[RecommendationItem] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnRecord:
        with self._lock:
            turn = TurnRecord(
                turn_index=len(self.turns),
                user_message=user_message,
                agent_response=agent_response,
                recommendations=recommendations or [],
                metadata=metadata or {},
            )
            self.turns.append(turn)
            self.last_updated = time.time()
            self._persist()
            return turn

    def refine_context(self, new_context: str) -> str:
        with self._lock:
            if self.turns:
                previous = set(self.context.split())
                current = set(new_context.split())
                additions = current - previous
                removals = current.intersection(
                    {w for w in previous if w.lower() in {"cheap", "expensive", "loud", "quiet"}}
                )
                combined = (previous - removals) | additions
                self.context = " ".join(sorted(combined))
            else:
                self.context = new_context
            self.last_updated = time.time()
            self._persist()
            return self.context

    def record_feedback(self, item_id: str, accepted: bool) -> None:
        with self._lock:
            if accepted:
                if item_id not in self.accepted_recommendations:
                    self.accepted_recommendations.append(item_id)
                if item_id in self.rejected_recommendations:
                    self.rejected_recommendations.remove(item_id)
            else:
                if item_id not in self.rejected_recommendations:
                    self.rejected_recommendations.append(item_id)
                if item_id in self.accepted_recommendations:
                    self.accepted_recommendations.remove(item_id)
            self.last_updated = time.time()
            self._persist()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "conversation_id": self.conversation_id,
                "turn_count": self.turn_count,
                "context": self.context,
                "accepted_count": len(self.accepted_recommendations),
                "rejected_count": len(self.rejected_recommendations),
                "last_turn": self.turns[-1].user_message[:100] if self.turns else "",
                "last_updated": self.last_updated,
            }

    def to_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "conversation_id": self.conversation_id,
                "user_profile": (
                    self.user_profile.model_dump(mode="json") if self.user_profile else None
                ),
                "turns": [turn.to_payload() for turn in self.turns],
                "context": self.context,
                "accepted_recommendations": self.accepted_recommendations,
                "rejected_recommendations": self.rejected_recommendations,
                "preference_refinements": self.preference_refinements,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
            }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConversationState":
        user_profile = None
        if payload.get("user_profile"):
            try:
                user_profile = UserProfile(**payload["user_profile"])
            except (TypeError, ValueError):
                user_profile = None
        return cls(
            conversation_id=str(payload.get("conversation_id") or ""),
            user_profile=user_profile,
            turns=[TurnRecord.from_payload(row) for row in payload.get("turns") or []],
            context=str(payload.get("context") or ""),
            accepted_recommendations=[
                str(value) for value in payload.get("accepted_recommendations") or []
            ],
            rejected_recommendations=[
                str(value) for value in payload.get("rejected_recommendations") or []
            ],
            preference_refinements=dict(payload.get("preference_refinements") or {}),
            created_at=float(payload.get("created_at") or time.time()),
            last_updated=float(payload.get("last_updated") or time.time()),
        )

    def _persist(self) -> None:
        if self._on_change:
            self._on_change(self)


class ConversationManager:
    """SQLite-backed multi-turn conversation state store."""

    _instance: ConversationManager | None = None
    _max_conversations: int = 1000

    def __new__(cls) -> "ConversationManager":
        db_path = _conversation_db_path()
        if cls._instance is None or getattr(cls._instance, "_db_path", None) != db_path:
            cls._instance = super().__new__(cls)
            cls._instance._db_path = db_path
            cls._instance._conversations = OrderedDict()
            cls._instance._lock = RLock()
            cls._instance._init_db()
            cls._instance._load_conversations()
        return cls._instance

    def create(self, user_profile: UserProfile | None = None, context: str = "") -> ConversationState:
        with self._lock:
            conversation_id = str(uuid.uuid4())[:8]
            state = ConversationState(
                conversation_id=conversation_id,
                user_profile=user_profile,
                context=context,
            )
            state.bind_persistence(self._persist_state)
            self._conversations[conversation_id] = state
            self._persist_state(state)
            self._evict_if_needed()
            return state

    def get(self, conversation_id: str) -> ConversationState | None:
        with self._lock:
            state = self._conversations.get(conversation_id)
            if state is not None:
                state.last_updated = time.time()
                self._conversations.move_to_end(conversation_id)
            return state

    def get_or_create(
        self,
        conversation_id: str,
        user_profile: UserProfile | None = None,
        context: str = "",
    ) -> ConversationState:
        state = self.get(conversation_id)
        if state is None:
            state = self.create(user_profile=user_profile, context=context)
        return state

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            deleted = self._conversations.pop(conversation_id, None) is not None
            if deleted:
                with self._connect() as conn:
                    conn.execute(
                        "DELETE FROM conversations WHERE conversation_id = ?",
                        (conversation_id,),
                    )
            return deleted

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            states = sorted(
                self._conversations.values(),
                key=lambda state: state.last_updated,
                reverse=True,
            )[:limit]
        return [state.summary() for state in states]

    def _evict_if_needed(self) -> None:
        while len(self._conversations) > self._max_conversations:
            conversation_id, _ = self._conversations.popitem(last=False)
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_last_updated
                ON conversations(last_updated DESC)
                """
            )

    def _load_conversations(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload FROM conversations
                ORDER BY last_updated ASC
                LIMIT ?
                """,
                (self._max_conversations,),
            ).fetchall()
        for row in rows:
            try:
                state = ConversationState.from_payload(json.loads(row["payload"]))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
            if not state.conversation_id:
                continue
            state.bind_persistence(self._persist_state)
            self._conversations[state.conversation_id] = state

    def _persist_state(self, state: ConversationState) -> None:
        payload = json.dumps(state.to_payload(), ensure_ascii=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(conversation_id, created_at, last_updated, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    last_updated = excluded.last_updated,
                    payload = excluded.payload
                """,
                (state.conversation_id, state.created_at, state.last_updated, payload),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn


def _conversation_db_path() -> Path:
    return Path(os.getenv("BLUECHIP_CONVERSATION_SQLITE_PATH") or runtime_db_path())


def get_conversation_manager() -> ConversationManager:
    return ConversationManager()
