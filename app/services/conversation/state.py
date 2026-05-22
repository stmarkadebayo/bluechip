from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import RecommendationItem, UserProfile


@dataclass
class TurnRecord:
    turn_index: int
    user_message: str
    agent_response: str
    recommendations: list[RecommendationItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


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

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def add_turn(
        self,
        user_message: str,
        agent_response: str,
        recommendations: list[RecommendationItem] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnRecord:
        turn = TurnRecord(
            turn_index=len(self.turns),
            user_message=user_message,
            agent_response=agent_response,
            recommendations=recommendations or [],
            metadata=metadata or {},
        )
        self.turns.append(turn)
        self.last_updated = time.time()
        return turn

    def refine_context(self, new_context: str) -> str:
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
        return self.context

    def record_feedback(self, item_id: str, accepted: bool) -> None:
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

    def summary(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_count": self.turn_count,
            "context": self.context,
            "accepted_count": len(self.accepted_recommendations),
            "rejected_count": len(self.rejected_recommendations),
            "last_turn": self.turns[-1].user_message[:100] if self.turns else "",
            "last_updated": self.last_updated,
        }


class ConversationManager:
    """In-memory multi-turn conversation state store.

    Manages conversation sessions for multi-turn recommendation scenarios.
    Each conversation maintains user profile, turn history, context refinement,
    and feedback tracking.  State persists in-memory (for the hackathon;
    production would use a persistent store).
    """

    _instance: ConversationManager | None = None
    _max_conversations: int = 1000

    def __new__(cls) -> "ConversationManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conversations: OrderedDict[str, ConversationState] = OrderedDict()
        return cls._instance

    def create(self, user_profile: UserProfile | None = None, context: str = "") -> ConversationState:
        conversation_id = str(uuid.uuid4())[:8]
        state = ConversationState(
            conversation_id=conversation_id,
            user_profile=user_profile,
            context=context,
        )
        self._conversations[conversation_id] = state
        self._evict_if_needed()
        return state

    def get(self, conversation_id: str) -> ConversationState | None:
        state = self._conversations.get(conversation_id)
        if state is not None:
            state.last_updated = time.time()
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
        return self._conversations.pop(conversation_id, None) is not None

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            state.summary()
            for state in list(self._conversations.values())[-limit:]
        ]

    def _evict_if_needed(self) -> None:
        while len(self._conversations) > self._max_conversations:
            self._conversations.popitem(last=False)


def get_conversation_manager() -> ConversationManager:
    return ConversationManager()
