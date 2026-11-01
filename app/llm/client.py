import json
from typing import Any

import httpx

from app.core.config import settings


class LLMError(Exception):
    """Raised when an LLM provider cannot return usable output."""


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.api_key = api_key if api_key is not None else (
            settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        )
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(settings.llm_enabled and self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str, *, timeout_seconds: float | None = None) -> str:
        if not self.api_key:
            raise LLMError("LLM API key is not configured.")
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response did not contain a message.") from exc
        if not isinstance(content, str):
            raise LLMError("LLM response content was not text.")
        return content


def get_llm_client() -> LLMClient | None:
    client = LLMClient()
    return client if client.enabled else None