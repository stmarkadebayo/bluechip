from __future__ import annotations

from app.models.schemas import UserProfile


class InMemoryProfileStore:
    """Development-only profile store.

    Production should replace this with a feature store or low-latency database.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def put(self, user_id: str, profile: UserProfile) -> None:
        self._profiles[user_id] = profile
