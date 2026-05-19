from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from app.models.schemas import Item
from app.services.generation import generator
from app.services.generation.generator import generate_review, generate_review_result
from app.services.generation.providers import (
    DEEPSEEK_DEFAULT_MODEL,
    OPENROUTER_DEFAULT_MODEL,
    DeepSeekChatProvider,
    MockGenerationProvider,
    OpenRouterChatProvider,
    get_generation_provider,
)
from app.services.profiling.item_profile import build_item_profile
from app.services.profiling.user_profile import build_user_profile
from app.services.validation.critic import validate_review_simulation


def test_review_generation_fallback_is_rating_conditioned_and_grounded(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "template")
    user_profile = build_user_profile(
        persona="A Lagos student who likes quiet affordable restaurants.",
        history=[],
        locale="Nigeria",
    )
    item_profile = build_item_profile(
        Item(
            item_id="r1",
            name="Calm Grill",
            category="restaurant",
            summary="Quiet affordable grill with fast service.",
            average_rating=4.4,
        )
    )

    review = generate_review(user_profile, item_profile, predicted_rating=4)
    validation = validate_review_simulation(4, review, user_profile, item_profile)

    assert "Calm Grill" in review
    assert "4 out of 5" in review
    assert validation.is_consistent


def test_review_generation_prompt_contains_llm_contract(monkeypatch) -> None:
    captured = {}

    class CapturingProvider:
        def generate(self, instructions: str, prompt: str) -> str:
            captured["instructions"] = instructions
            captured["prompt"] = prompt
            return "I would rate Calm Grill 4 out of 5. Quiet and affordable enough for me."

    monkeypatch.setattr(generator, "get_generation_provider", lambda: CapturingProvider())
    user_profile = build_user_profile(
        persona="A practical diner who likes quiet affordable restaurants.",
        history=[],
    )
    item_profile = build_item_profile(
        Item(
            item_id="r1",
            name="Calm Grill",
            category="restaurant",
            summary="Quiet affordable grill with fast service.",
            average_rating=4.4,
        )
    )

    review = generate_review(user_profile, item_profile, predicted_rating=4)

    assert review.startswith("I would rate Calm Grill")
    assert "Do not invent item facts" in captured["instructions"]
    assert "Predicted rating: 4/5" in captured["prompt"]
    assert "Required first sentence: I would rate Calm Grill 4 out of 5." in captured["prompt"]
    assert "Calm Grill" in captured["prompt"]


def test_review_generation_falls_back_when_provider_fails(monkeypatch) -> None:
    class FailingProvider:
        def generate(self, instructions: str, prompt: str) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(generator, "get_generation_provider", lambda: FailingProvider())
    user_profile = build_user_profile(
        persona="A strict buyer who dislikes noisy products.",
        history=[],
    )
    item_profile = build_item_profile(
        Item(
            item_id="p1",
            name="Quiet Fan",
            category="appliance",
            summary="Quiet compact fan.",
            average_rating=4.0,
        )
    )

    review = generate_review(user_profile, item_profile, predicted_rating=2)

    assert "Quiet Fan" in review
    assert "2 out of 5" in review


def test_review_generation_strict_provider_surfaces_failure(monkeypatch) -> None:
    class FailingProvider:
        def generate(self, instructions: str, prompt: str) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(generator, "get_generation_provider", lambda: FailingProvider())
    user_profile = build_user_profile(persona="A strict buyer.", history=[])
    item_profile = build_item_profile(
        Item(item_id="p1", name="Quiet Fan", category="appliance", summary="Quiet fan.")
    )

    try:
        generate_review_result(user_profile, item_profile, predicted_rating=2, strict_provider=True)
    except RuntimeError as exc:
        assert "provider unavailable" in str(exc)
    else:
        raise AssertionError("strict provider mode should raise generation failures")


def test_review_generation_repairs_missing_rating_contract(monkeypatch) -> None:
    class LooseProvider:
        def generate(self, instructions: str, prompt: str) -> str:
            return "Pepper House is affordable and fast enough for me."

    monkeypatch.setattr(generator, "get_generation_provider", lambda: LooseProvider())
    user_profile = build_user_profile(
        persona="A Lagos student who likes affordable meals.",
        history=[],
        locale="Nigeria",
    )
    item_profile = build_item_profile(
        Item(item_id="r1", name="Pepper House", category="restaurant", summary="Fast meals.")
    )

    result = generate_review_result(user_profile, item_profile, predicted_rating=4)
    validation = validate_review_simulation(4, result.text, user_profile, item_profile)

    assert result.text.startswith("I would rate Pepper House 4 out of 5.")
    assert validation.is_consistent


def test_mock_provider_exercises_prompt_mode_generation(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    user_profile = build_user_profile(
        persona="A Lagos student who likes quiet affordable restaurants.",
        history=[],
        locale="Nigeria",
    )
    item_profile = build_item_profile(
        Item(
            item_id="r1",
            name="Calm Grill",
            category="restaurant",
            summary="Quiet affordable grill with fast service.",
            average_rating=4.4,
        )
    )

    result = generate_review_result(user_profile, item_profile, predicted_rating=4)

    assert isinstance(get_generation_provider(), MockGenerationProvider)
    assert result.provider == "mock"
    assert "Calm Grill" in result.text
    assert "4 out of 5" in result.text
    assert "Nigerian English" in result.text


def test_openrouter_provider_uses_deepseek_v4_flash_free_by_default(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    provider = get_generation_provider()

    assert isinstance(provider, OpenRouterChatProvider)
    assert provider.model == OPENROUTER_DEFAULT_MODEL


def test_deepseek_provider_is_preferred_when_key_is_configured(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "other-test-key")

    provider = get_generation_provider()

    assert isinstance(provider, DeepSeekChatProvider)
    assert provider.model == DEEPSEEK_DEFAULT_MODEL


def test_openrouter_provider_parses_chat_completion_response(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"Grounded generated review."}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenRouterChatProvider(api_key="test-key", timeout_seconds=7)

    text = provider.generate("Follow facts.", "Write the review.")

    assert text == "Grounded generated review."
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert b"deepseek/deepseek-v4-flash:free" in captured["body"]
    assert captured["timeout"] == 7


def test_deepseek_provider_parses_chat_completion_response(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"DeepSeek generated review."}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = DeepSeekChatProvider(api_key="test-key", timeout_seconds=7)

    text = provider.generate("Follow facts.", "Write the review.")

    assert text == "DeepSeek generated review."
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert b"deepseek-v4-flash" in captured["body"]
    assert captured["timeout"] == 7


def test_openrouter_provider_reports_http_error_body(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=BytesIO(b'{"error":{"message":"credits required"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenRouterChatProvider(api_key="test-key", max_retries=0)

    try:
        provider.generate("Follow facts.", "Write the review.")
    except RuntimeError as exc:
        assert "HTTP 429" in str(exc)
        assert "credits required" in str(exc)
    else:
        raise AssertionError("HTTP errors should surface as RuntimeError")
