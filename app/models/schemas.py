from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class UserHistoryItem(BaseModel):
    item_id: str
    item_name: str
    rating: float = Field(ge=1, le=5)
    review: str
    category: Optional[str] = None
    timestamp: Optional[int] = None


class Item(BaseModel):
    item_id: str
    name: str
    category: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    average_rating: Optional[float] = Field(default=None, ge=1, le=5)


class UserProfile(BaseModel):
    locale: Optional[str] = None
    average_rating: float
    rating_strictness: str
    seen_item_ids: list[str] = Field(default_factory=list)
    preferred_terms: list[str]
    disliked_terms: list[str]
    preferred_categories: list[str]
    category_affinity: dict[str, float] = Field(default_factory=dict)
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    recent_terms: list[str] = Field(default_factory=list)
    review_length_mean: float = 0.0
    evidence_count: int = 0
    confidence: float = Field(default=0.0, ge=0, le=1)
    voice_style: str
    signals: list[str]


class ItemProfile(BaseModel):
    item_id: str
    name: str
    category: str
    quality_score: float = Field(ge=0, le=1)
    terms: list[str]
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    average_rating: Optional[float] = None
    popularity: int = 0
    signals: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RatingPrediction(BaseModel):
    predicted_rating: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    user_signals: list[str]
    item_signals: list[str]


class ValidationResult(BaseModel):
    is_consistent: bool
    issues: list[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    step: str
    status: str
    detail: str = ""


class SimulateReviewRequest(BaseModel):
    user_persona: str
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    target_item: Item
    locale: Optional[str] = None


class ProfileUserRequest(BaseModel):
    user_persona: str
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    locale: Optional[str] = None


class SimulateReviewResponse(BaseModel):
    predicted_rating: int
    review: str
    confidence: float
    user_signals: list[str]
    item_signals: list[str]
    validation: ValidationResult
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)


class UserProfileResponse(BaseModel):
    profile: UserProfile


class RecommendationRequest(BaseModel):
    user_persona: str
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    context: str = ""
    candidate_items: list[Item]
    locale: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=20)


class RecommendationItem(BaseModel):
    rank: int
    item_id: str
    name: str
    score: float = Field(ge=0, le=1)
    reason: str
    tradeoffs: str
    signals: list[str]
    matched_signals: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)


class ReviewRecord(BaseModel):
    review_id: str
    user_id: str
    item_id: str
    item_name: str
    rating: float = Field(ge=1, le=5)
    review: str
    category: str = "unknown"
    timestamp: int = 0


class EvalMetric(BaseModel):
    name: str
    value: float


class EvalReport(BaseModel):
    task: str
    dataset: str
    examples: int
    metrics: list[EvalMetric]
    notes: list[str] = Field(default_factory=list)
