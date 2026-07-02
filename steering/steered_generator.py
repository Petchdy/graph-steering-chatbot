"""SteeredRemoteGenerator — a Generator that overlays activation steering on the reply.

Composes the existing Ollama LocalLLMGenerator so the CBT pipeline is untouched:
  * active_strategy == "none"  -> delegate entirely to LocalLLMGenerator (zero behaviour change).
  * else -> LocalLLMGenerator supplies CBT `technique`/`phase` (drives the graph); the steering
    microservice supplies the steered `response` text. On any service error, fall back to the
    Ollama reply so the chat never breaks.

The strategy is per-turn generator state (set via set_strategy) so the Generator Protocol
(`generate(system, history)`) is unchanged. The chatbot's /chat + UI set it before each turn.
"""

from __future__ import annotations

import os

from cbt_kg.interfaces import Generator
from cbt_kg.generate import LocalLLMGenerator


class SteeredRemoteGenerator(Generator):
    def __init__(self, steer_url: str | None = None, fallback: LocalLLMGenerator | None = None,
                 default_strategy: str = "none"):
        self._url = (steer_url or os.environ.get("STEER_URL", "http://localhost:8100")).rstrip("/")
        self._fallback = fallback or LocalLLMGenerator()
        self.active_strategy = default_strategy

    def set_strategy(self, strategy: str) -> None:
        self.active_strategy = strategy or "none"

    def _messages(self, history: list[tuple[str, str]]) -> list[dict]:
        """history [(client, therapist), ...] -> chat messages (client=user, therapist=assistant)."""
        msgs = []
        for user_msg, assistant_msg in history[:-1]:
            msgs.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                msgs.append({"role": "assistant", "content": str(assistant_msg)})
        if history:
            msgs.append({"role": "user", "content": str(history[-1][0])})
        return msgs

    def _service(self, strategy: str, history: list[tuple[str, str]]) -> str:
        import requests
        resp = requests.post(
            f"{self._url}/generate",
            json={"messages": self._messages(history), "strategy": strategy},
            timeout=180,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        # CBT metadata (technique/phase, drives the graph) comes from Ollama. If Ollama is absent
        # (e.g. steering-only demo), default the metadata so steering still works without it.
        try:
            base = self._fallback.generate(system, history)
        except Exception as exc:  # noqa: BLE001 — Ollama down / not installed
            print(f"[steered] Ollama unavailable ({exc}); using default technique/phase")
            base = {"response": "", "technique": "Rapport Building", "phase": "Rapport"}

        strat = self.active_strategy
        # "none" baseline: prefer Ollama's real CBT reply; if Ollama is absent, use the steering
        # service with NO hook (strategy="none") — same model + same prompt, so it's a clean A/B
        # baseline whose ONLY difference from a steered reply is the vector.
        if strat == "none":
            if base.get("response"):
                return base
            try:
                txt = self._service("none", history)
                if txt:
                    return {**base, "response": txt}
            except Exception as exc:  # noqa: BLE001
                print(f"[steered] service error on none baseline ({exc})")
            base["response"] = base.get("response") or "(steering service unavailable)"
            return base

        try:
            steered = self._service(strat, history)
            if steered:
                return {**base, "response": steered}
        except Exception as exc:  # noqa: BLE001 — never break the chat on a steering hiccup
            print(f"[steered] service error ({exc}); falling back to base reply")
        return base
