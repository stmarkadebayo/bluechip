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
    recent_average_rating: float = 3.5
    rating_std: float = 0.0
    positive_rating_share: float = 0.0
    negative_rating_share: float = 0.0
    rating_trend: float = 0.0
    rating_strictness: str
    seen_item_ids: list[str] = Field(default_factory=list)
    preferred_terms: list[str]
    disliked_terms: list[str]
    preferred_categories: list[str]
    category_affinity: dict[str, float] = Field(default_factory=dict)
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    aspect_scores: dict[str, float] = Field(default_factory=dict)
    nigerian_context: list[str] = Field(default_factory=list)
    recent_terms: list[str] = Field(default_factory=list)
    review_length_mean: float = 0.0
    embedding: list[float] = Field(default_factory=list)
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
    aspect_scores: dict[str, float] = Field(default_factory=dict)
    nigerian_context: list[str] = Field(default_factory=list)
    average_rating: Optional[float] = None
    popularity: int = 0
    embedding: list[float] = Field(default_factory=list)
    signals: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RatingPrediction(BaseModel):
    predicted_rating: int = Field(ge=1, le=5)
    predicted_score: float | None = Field(default=None, ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    model_name: str | None = None
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
    user_id: Optional[str] = None
    user_persona: str
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    target_item: Item
    locale: Optional[str] = None


class ProfileUserRequest(BaseModel):
    user_id: Optional[str] = None
    user_persona: str
    user_history: list[UserHistoryItem] = Field(default_factory=list)
    locale: Optional[str] = None


class SimulateReviewResponse(BaseModel):
    trace_id: str | None = None
    predicted_rating: int
    predicted_score: float | None = None
    review: str
    confidence: float
    model_name: str | None = None
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
    candidate_sources: list[str] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)


class CandidateDiagnostics(BaseModel):
    strategy: str
    input_count: int
    candidate_count: int
    source_counts: dict[str, int] = Field(default_factory=dict)
    used_collaborative: bool = False


class RuntimeMetrics(BaseModel):
    requests: int
    by_endpoint: dict[str, int] = Field(default_factory=dict)
    by_generation_provider: dict[str, int] = Field(default_factory=dict)
    average_latency_ms: float = 0.0
    estimated_generation_tokens: int = 0
    estimated_generation_cost_usd: float = 0.0
    validation_failures: int = 0
    validation_failure_rate: float = 0.0
    fallback_count: int = 0
    fallback_rate: float = 0.0
    retrieval_source_counts: dict[str, int] = Field(default_factory=dict)
    model_version_counts: dict[str, int] = Field(default_factory=dict)
    index_version_counts: dict[str, int] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str
    endpoint: str
    created_at: str
    latency_ms: float
    generation_provider: str
    estimated_generation_tokens: int
    estimated_generation_cost_usd: float
    model_versions: dict[str, str] = Field(default_factory=dict)
    index_versions: dict[str, str] = Field(default_factory=dict)
    retrieval_source_counts: dict[str, int] = Field(default_factory=dict)
    validation_status: str = ""
    fallback_reason: str | None = None
    steps: list[AgentTraceStep] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    trace_id: str | None = None
    recommendations: list[RecommendationItem]
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    candidate_diagnostics: CandidateDiagnostics | None = None


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
