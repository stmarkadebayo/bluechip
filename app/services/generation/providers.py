from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import sleep


OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-v4-flash:free"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


class GenerationProvider:
    def generate(self, instructions: str, prompt: str) -> str:
        raise NotImplementedError


class TemplateGenerationProvider(GenerationProvider):
    def generate(self, instructions: str, prompt: str) -> str:
        del instructions
        return prompt


class MockGenerationProvider(GenerationProvider):
    """Local deterministic provider that exercises prompt-mode generation."""

    def generate(self, instructions: str, prompt: str) -> str:
        del instructions
        if "Predicted rating:" in prompt and "Item:" in prompt:
            return _mock_review(prompt)
        if "Candidate:" in prompt:
            return _mock_recommendation_reason(prompt)
        return "Generated locally from the provided prompt."


@dataclass
class OpenAIResponsesProvider(GenerationProvider):
    """Minimal OpenAI Responses API client using stdlib HTTP.

    This keeps the submission runnable without requiring the OpenAI Python SDK.
    """

    api_key: str
    model: str = "gpt-5"
    timeout_seconds: int = 30
    max_retries: int = 1

    def generate(self, instructions: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "store": False,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = _open_json(request, timeout_seconds=self.timeout_seconds, max_retries=self.max_retries)
        return _extract_output_text(data)


@dataclass
class OpenRouterChatProvider(GenerationProvider):
    """Minimal OpenRouter Chat Completions client using stdlib HTTP."""

    api_key: str
    model: str = OPENROUTER_DEFAULT_MODEL
    timeout_seconds: int = 30
    app_title: str = "Bluechip User Intelligence Agent"
    max_retries: int = 1

    def generate(self, instructions: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": self.app_title,
            },
            method="POST",
        )
        data = _open_json(request, timeout_seconds=self.timeout_seconds, max_retries=self.max_retries)
        return _extract_chat_completion_text(data)


@dataclass
class DeepSeekChatProvider(GenerationProvider):
    """Minimal DeepSeek Chat Completions client using the OpenAI-compatible API."""

    api_key: str
    model: str = DEEPSEEK_DEFAULT_MODEL
    base_url: str = DEEPSEEK_DEFAULT_BASE_URL
    timeout_seconds: int = 30
    max_retries: int = 1

    def generate(self, instructions: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
            "thinking": {"type": "disabled"},
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        data = _open_json(request, timeout_seconds=self.timeout_seconds, max_retries=self.max_retries)
        return _extract_chat_completion_text(data)


def get_generation_provider() -> GenerationProvider:
    provider = (_env("LLM_PROVIDER") or "").strip().lower()
    deepseek_api_key = _env("DEEPSEEK_API_KEY")
    openrouter_api_key = _env("OPENROUTER_API_KEY")
    openai_api_key = _env("OPENAI_API_KEY")
    if provider in {"mock", "local", "fixture"}:
        return MockGenerationProvider()
    if provider == "template":
        return TemplateGenerationProvider()
    if (provider == "deepseek" or (not provider and deepseek_api_key)) and deepseek_api_key:
        return DeepSeekChatProvider(
            api_key=deepseek_api_key,
            model=_deepseek_model(),
            base_url=_env("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE_URL,
        )
    if (provider == "openrouter" or (not provider and openrouter_api_key)) and openrouter_api_key:
        return OpenRouterChatProvider(
            api_key=openrouter_api_key,
            model=_openrouter_model(),
        )
    if provider == "openai" and openai_api_key:
        return OpenAIResponsesProvider(
            api_key=openai_api_key,
            model=_env("LLM_MODEL") or "gpt-5",
        )
    return TemplateGenerationProvider()


def generation_provider_name() -> str:
    provider = (_env("LLM_PROVIDER") or "").strip().lower()
    deepseek_api_key = _env("DEEPSEEK_API_KEY")
    openrouter_api_key = _env("OPENROUTER_API_KEY")
    if provider in {"mock", "local", "fixture"}:
        return "mock"
    if provider == "template":
        return "template"
    if provider == "deepseek" or (not provider and deepseek_api_key):
        return "deepseek" if deepseek_api_key else "template"
    if provider == "openrouter" or (not provider and openrouter_api_key):
        return "openrouter" if openrouter_api_key else "template"
    if provider == "openai":
        return "openai" if _env("OPENAI_API_KEY") else "template"
    return "template"


def _openrouter_model() -> str:
    model = (_env("OPENROUTER_MODEL") or _env("LLM_MODEL") or "").strip()
    if not model or model == "gpt-5":
        return OPENROUTER_DEFAULT_MODEL
    return model


def _deepseek_model() -> str:
    model = (_env("DEEPSEEK_MODEL") or _env("LLM_MODEL") or "").strip()
    if not model or model == "gpt-5":
        return DEEPSEEK_DEFAULT_MODEL
    return model


def _extract_output_text(data: dict) -> str:
    if data.get("output_text"):
        return str(data["output_text"])

    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


def _extract_chat_completion_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
        return "\n".join(chunks).strip()
    return ""


def _open_json(
    request: urllib.request.Request,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
    attempts = max(1, max_retries + 1)
    last_error = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            status = f"HTTP {exc.code}"
            last_error = f"{status}: {body}" if body else status
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(f"LLM generation failed: {last_error}") from exc
            sleep(_retry_delay(exc, attempt))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt == attempts - 1:
                raise RuntimeError(f"LLM generation failed: {last_error}") from exc
            sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"LLM generation failed: {last_error}")


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:500]
    message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else None
    return str(message or data)[:500]


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return min(float(retry_after), 3.0)
        except ValueError:
            pass
    return min(0.5 * (2 ** attempt), 3.0)


def _mock_review(prompt: str) -> str:
    rating = _line_value(prompt, "Predicted rating") or "3/5"
    rating_number = rating.split("/", 1)[0].strip()
    item = _line_value(prompt, "Item") or "the item"
    locale = (_line_value(prompt, "Locale") or "").lower()
    signals = _line_value(prompt, "Item signals") or "the provided item details"
    locale_clause = " In natural Nigerian English," if "nigeria" in locale else ""
    if rating_number in {"4", "5"}:
        verdict = "it matches the things I usually value."
    elif rating_number == "3":
        verdict = "it has useful strengths, but I would still be selective."
    else:
        verdict = "it misses enough of my preferences that I would be cautious."
    return (
        f"I would rate {item} {rating_number} out of 5."
        f"{locale_clause} Based on the provided signals, {signals}. For me, {verdict}"
    )


def _mock_recommendation_reason(prompt: str) -> str:
    candidate = _line_value(prompt, "Candidate") or "This item"
    context = _line_value(prompt, "Context")
    context_clause = f" for {context}" if context else ""
    return (
        f"{candidate} ranks well{context_clause} because its candidate signals match "
        "the user's stated preferences, with only the listed tradeoffs considered."
    )


def _line_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return _env_file_values().get(name)


@lru_cache(maxsize=1)
def _env_file_values() -> dict[str, str]:
    path = Path(".env")
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        values[key] = value
    return values
