"""Minimal client for a local Ollama server."""

import json
from typing import Any, Optional

import httpx

from app.config import get_settings


class LLMError(Exception):
    """Raised when the LLM is unreachable or returns unusable output."""


class OllamaClient:
    """Calls Ollama's chat endpoint. Temperature 0 so the same question gives the same query."""

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], schema: Optional[dict] = None) -> str:
        """Send a conversation, return the assistant text. `schema` forces JSON matching that schema."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        if schema is not None:
            payload["format"] = schema
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return response.json()["message"]["content"]

    def chat_json(self, messages: list[dict[str, str]], schema: dict) -> dict[str, Any]:
        """Like chat() but parses and returns the JSON object."""
        text = self.chat(messages, schema)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]}") from exc
        if not isinstance(data, dict):
            raise LLMError("LLM JSON output is not an object")
        return data

    def is_available(self) -> bool:
        """True if the server answers and the configured model is installed."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            names = {m["name"] for m in response.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, ValueError, KeyError):
            return False


def get_llm() -> OllamaClient:
    """Build the client from settings."""
    settings = get_settings()
    return OllamaClient(settings.ollama_url, settings.llm_model, settings.llm_timeout_seconds)
