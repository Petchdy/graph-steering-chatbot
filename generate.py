"""Swappable response generators. All return dict: {response, technique, phase}.

EchoGenerator        - deterministic offline stub for tests.
LocalLLMGenerator    - Ollama native /api/chat (qwen3.5-nothink). Primary path.
OpenRouterGenerator  - Claude via OpenRouter. Optional alternative.
"""

import json
import os
import re


def _build_messages(system: str, history: list[tuple[str, str]]) -> list[dict]:
    messages = [{"role": "system", "content": str(system)}]
    for user_msg, assistant_msg in history[:-1]:
        messages.append({"role": "user", "content": str(user_msg)})
        if assistant_msg:
            messages.append({"role": "assistant", "content": str(assistant_msg)})
    if history:
        messages.append({"role": "user", "content": str(history[-1][0])})
    return messages


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON. Returns fallback dict on failure.

    qwen3.5 with format=json sometimes over-escapes string delimiters as `\\"`
    instead of `"`. We retry with those collapsed when the first parse fails.
    """
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        fixed = cleaned.replace('\\"', '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Last-ditch: pull out a "response" string by regex so the chat doesn't break.
    m = re.search(r'"response"\s*:\s*\\?"(.+?)\\?"\s*,', cleaned, re.DOTALL)
    if m:
        return {"response": m.group(1), "technique": "Rapport Building", "phase": "Rapport"}
    print(f"[generate] JSON parse failed, using raw text as response. raw={raw[:120]!r}")
    return {"response": raw, "technique": "Rapport Building", "phase": "Rapport"}


_FALLBACK = {"response": "", "technique": "Rapport Building", "phase": "Rapport"}


class EchoGenerator:
    """Offline stub — returns a dict with a canned echo. Used by tests."""

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        last = history[-1][0] if history else ""
        return {"response": f"[echo] {last}", "technique": "Rapport Building", "phase": "Rapport"}


class LocalLLMGenerator:
    """Calls Ollama's native /api/chat endpoint (NOT the OpenAI-compatible /v1).

    Thinking models like qwen3 only reliably honor 'think': false on the native
    API; the /v1 endpoint can stall or return empty content.
    """

    def __init__(self, model: str = "qwen3.5-nothink", base_url: str = "http://localhost:11434/v1"):
        self._model = model
        self._host = base_url.removesuffix("/v1")

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        import requests

        messages = _build_messages(system, history)
        response = requests.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {"temperature": 0.4},
            },
            timeout=180,
        )
        if not response.ok:
            raise RuntimeError(f"Ollama /api/chat {response.status_code}: {response.text}")
        raw = response.json()["message"]["content"]
        return _parse_json(raw)


class OpenRouterGenerator:
    """Calls a model through OpenRouter's chat completions API."""

    def __init__(self, model: str = "anthropic/claude-sonnet-4-6", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY required for OpenRouterGenerator")

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        import requests

        messages = _build_messages(system, history)
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "messages": messages},
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return _parse_json(raw)
