"""SteeredRemoteGenerator — a Generator that overlays activation steering on the reply.

Composes the existing Ollama LocalLLMGenerator so the CBT pipeline is untouched:
  * active_strategy == "none"  -> delegate entirely to LocalLLMGenerator (zero behaviour change).
  * else -> LocalLLMGenerator supplies CBT `technique`/`phase` (drives the graph); the steering
    microservice supplies the steered `response` text. On any service error, fall back to the
    Ollama reply so the chat never breaks.

GPU-lean single-model mode: construct with `fallback=None` (factory sets this when STEER_NO_OLLAMA=1)
to run WITHOUT Ollama entirely — both the baseline ("none") and steered replies come from the one HF
model behind the steering service, and technique/phase fall back to defaults (therapy.py still
advances phases deterministically via PHASE_ORDER + node-grounded validate_phase). The baseline reply
passes the CBT `system` prompt through to the service so it stays on-task without a steering vector.

The strategy is per-turn generator state (set via set_strategy) so the Generator Protocol
(`generate(system, history)`) is unchanged. The chatbot's /chat + UI set it before each turn.
"""

from __future__ import annotations

import os

from cbt_kg.interfaces import Generator
from cbt_kg.generate import LocalLLMGenerator

_MISSING = object()  # distinguishes "fallback omitted -> default to Ollama" from "fallback=None -> no Ollama"


class SteeredRemoteGenerator(Generator):
    def __init__(self, steer_url: str | None = None, fallback=_MISSING,
                 default_strategy: str = "none"):
        self._url = (steer_url or os.environ.get("STEER_URL", "http://localhost:8100")).rstrip("/")
        # Omitted -> back-compat default (Ollama). Explicit None -> Ollama-free single-model mode.
        self._fallback = LocalLLMGenerator() if fallback is _MISSING else fallback
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

    def _service(self, strategy: str, history: list[tuple[str, str]],
                 system: str | None = None) -> str:
        import requests
        resp = requests.post(
            f"{self._url}/generate",
            json={"messages": self._messages(history), "strategy": strategy, "system": system},
            timeout=180,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        # CBT metadata (technique/phase, drives the graph) comes from Ollama when present. In
        # single-model mode (fallback is None) we skip Ollama entirely and default the metadata —
        # therapy.py still advances phases deterministically via PHASE_ORDER + validate_phase.
        if self._fallback is None:
            base = {"response": "", "technique": "Rapport Building", "phase": "Rapport"}
        else:
            try:
                base = self._fallback.generate(system, history)
            except Exception as exc:  # noqa: BLE001 — Ollama down / not installed
                print(f"[steered] Ollama unavailable ({exc}); using default technique/phase")
                base = {"response": "", "technique": "Rapport Building", "phase": "Rapport"}

        strat = self.active_strategy
        # "none" baseline: prefer Ollama's real CBT reply; if Ollama is absent, use the steering
        # service with NO hook (strategy="none") — same model, same CBT `system` prompt (passed
        # through), so it's a clean A/B baseline whose ONLY difference from a steered reply is the vector.
        if strat == "none":
            if base.get("response"):
                return {**base, "steer_status": "none"}
            try:
                txt = self._service("none", history, system=system)
                if txt:
                    return {**base, "response": txt, "steer_status": "none"}
            except Exception as exc:  # noqa: BLE001
                print(f"[steered] service error on none baseline ({exc})")
            base["response"] = base.get("response") or "(steering service unavailable)"
            return {**base, "steer_status": "none"}

        # A strategy is selected but the service errored or returned nothing — this is silent to
        # the chat itself (never breaks the turn), but the UI needs to know so it can show the
        # reply is NOT actually steered, rather than leaving the user to guess from tone alone.
        try:
            steered = self._service(strat, history)
            if steered:
                return {**base, "response": steered, "steer_status": "steered"}
        except Exception as exc:  # noqa: BLE001 — never break the chat on a steering hiccup
            print(f"[steered] service error ({exc}); falling back to base reply")
        return {**base, "steer_status": "fallback"}
