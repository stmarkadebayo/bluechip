from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class GenerationProvider:
    def generate(self, instructions: str, prompt: str) -> str:
        raise NotImplementedError


class TemplateGenerationProvider(GenerationProvider):
    def generate(self, instructions: str, prompt: str) -> str:
        del instructions
        return prompt


@dataclass
class OpenAIResponsesProvider(GenerationProvider):
    """Minimal OpenAI Responses API client using stdlib HTTP.

    This keeps the submission runnable without requiring the OpenAI Python SDK.
    """

    api_key: str
    model: str = "gpt-5"
    timeout_seconds: int = 30

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI generation failed: {exc}") from exc
        return _extract_output_text(data)


def get_generation_provider() -> GenerationProvider:
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY")
    if provider == "openai" and api_key:
        return OpenAIResponsesProvider(
            api_key=api_key,
            model=os.getenv("LLM_MODEL") or "gpt-5",
        )
    return TemplateGenerationProvider()


def _extract_output_text(data: dict) -> str:
    if data.get("output_text"):
        return str(data["output_text"])

    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()
