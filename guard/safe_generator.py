"""SafeGenerator — a Generator that sanitizes whatever the inner Generator produced.

Composes any `cbt_kg.interfaces.Generator` (the same pattern steering/steered_generator.py uses to
compose LocalLLMGenerator), so wrapping happens once in factory.make_generator and covers EVERY
path: echo, local Ollama, OpenRouter, the steering service, and — the one the service-side guard
structurally cannot reach — SteeredRemoteGenerator's fall-back-to-Ollama reply when the steering
service errors.

Contract:
  * `response` is replaced by the sanitized text. This is what reaches session.history, the
    transcript and the graph, so no fabricated number can compound across turns.
  * every other key is passed through untouched — `technique` and `phase` drive phase gating and
    the knowledge graph, `steer_status` drives the UI chip.
  * a new `guard` key reports {needs_footer, crisis, removed}; therapy.py appends the verified
    footer to the OUTWARD reply only, never to history (see guard/NOTES.md).
  * the chat never hard-fails on a guard bug: internal errors fail OPEN, matching the steering
    fallback philosophy. The one exception is a reply that already looked referral-shaped when the
    failure happened — that returns the canonical sentence rather than unfiltered text.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .sanitize import (_ORG_FALLBACK_NOUNS, _PHONE_STRONG_RE, _PLACEHOLDER_RE, _URL_RE,
                       load_config, sanitize)

LOG_PATH = Path(__file__).parent / "logs" / "guard_log.jsonl"


class SafeGenerator:
    """Wraps an inner Generator and filters its replies. See module docstring."""

    def __init__(self, inner, config: dict | None = None,
                 log_path: str | Path | None = None):
        self._inner = inner
        self._config = config
        self._log_path = Path(log_path) if log_path else LOG_PATH

    def __getattr__(self, name):
        """Delegate everything else to the wrapped generator.

        Load-bearing: api.py and ui.py call `generator.set_strategy(...)` behind a `hasattr` check,
        so without this the steering strategy dropdown would silently stop working the moment the
        guard is enabled.
        """
        return getattr(self._inner, name)

    # ─────────────────────────── Generator protocol ────────────────────────

    def generate(self, system: str, history: list[tuple[str, str]]) -> dict:
        result = self._inner.generate(system, history)
        if not isinstance(result, dict):
            return result

        raw = str(result.get("response") or "")
        try:
            cfg = self._config if self._config is not None else load_config()
            user_message = str(history[-1][0]) if history else ""
            res = sanitize(raw, cfg, user_message=user_message)

            out = dict(result)
            out["response"] = res.clean
            out["guard"] = {
                "needs_footer": res.needs_footer,
                "crisis": res.crisis,
                "removed": list(res.removed),
            }
            if res.removed or res.crisis:
                self._log({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "raw": raw,
                    "clean": res.clean,
                    "removed": res.removed,
                    "entities": res.entities,
                    "crisis": res.crisis,
                    "steer_status": result.get("steer_status"),
                })
            return out
        except Exception as exc:  # noqa: BLE001 — a guard bug must never break the chat
            print(f"[guard] sanitize failed, failing open: {type(exc).__name__}: {exc}")
            fallback = self._panic_text(raw)
            self._log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "raw": raw,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_open": fallback is None,
            })
            if fallback is None:
                return result
            out = dict(result)
            out["response"] = fallback
            out["guard"] = {"needs_footer": True, "crisis": False, "removed": [raw]}
            return out

    # ─────────────────────────── internals ─────────────────────────────────

    def _panic_text(self, raw: str) -> str | None:
        """Canonical sentence if the reply looked referral-shaped, else None (= fail open).

        Used only when sanitize() itself raised. Returning raw text that we already have reason to
        believe contains contact details is the one failure this guard must not commit, so a coarse
        second look is worth the few microseconds.
        """
        try:
            looks_referral = bool(
                _URL_RE.search(raw) or _PHONE_STRONG_RE.search(raw)
                or _PLACEHOLDER_RE.search(raw)
                or any(n in raw for n in _ORG_FALLBACK_NOUNS))
            if not looks_referral:
                return None
            cfg = self._config if self._config is not None else load_config()
            return (cfg.get("canonical_sentence") or "").strip() or None
        except Exception:  # noqa: BLE001 — the panic path itself must not raise
            return None

    def _log(self, record: dict) -> None:
        """Append one JSON line. This is the research record of the deployment-side fabrication
        rate, so it is written for every removal — but a logging failure never breaks a turn."""
        try:
            path = self._log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[guard] could not write {self._log_path}: {type(exc).__name__}: {exc}")


def guard_enabled() -> bool:
    """REFERRAL_GUARD=0 disables the guard entirely (research mode — reproduces the pure
    studied condition, byte-identical to pre-guard behaviour)."""
    return os.environ.get("REFERRAL_GUARD", "1") != "0"
