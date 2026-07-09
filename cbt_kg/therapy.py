"""Part 1 orchestrator: async turn loop + node-grounded phase gates.

The therapy reply is generated against the PRE-turn graph context, so the
client's question never blocks on extraction. Extraction runs concurrently
under a per-session asyncio.Lock; the result is merged back into the graph
before the function returns. Every CONSOLIDATE_EVERY turns, Tier B fires
in a detached background task that does not block the response.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from .interfaces import Extractor, Generator, GraphStore, Schema
from .prompts import THERAPIST_SYSTEM

PHASE_ORDER = ["Rapport", "Exploration", "Technique", "Consolidation"]

PHASE_MINIMUMS: dict[str, dict] = {
    "Exploration":   {"requires": ["Problem"],                       "min_turns": 2},
    "Technique":     {"requires": ["AutomaticThought", "Situation"], "min_turns": 5},
    "Consolidation": {"requires": ["AdaptiveResponse"],              "min_turns": 12},
}

EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "8"))
CONSOLIDATE_EVERY = int(os.environ.get("CONSOLIDATE_EVERY", "6"))


def validate_phase(proposed: str, current: str, graph: GraphStore,
                   turn_count: int) -> str:
    """Allow the proposed phase only if its V4_flat node-class minimums hold
    AND turn_count >= the required minimum. Going backwards / staying put
    is always allowed."""
    try:
        if PHASE_ORDER.index(proposed) <= PHASE_ORDER.index(current):
            return proposed
    except ValueError:
        return current
    mins = PHASE_MINIMUMS.get(proposed, {})
    classes_met = all(graph.count_found(c) >= 1 for c in mins.get("requires", []))
    turns_met = turn_count >= mins.get("min_turns", 0)
    return proposed if (classes_met and turns_met) else current


@dataclass
class Session:
    schema: Schema
    graph: GraphStore
    extractor: Extractor
    generator: Generator
    history: list[tuple[str, str]] = field(default_factory=list)
    turn_count: int = 0
    extraction_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transcript: list[tuple[int, str, str]] = field(default_factory=list)


# ─────────────────────── Sync wrapper (tests / Gradio) ────────────────────

def turn(session: Session, user_message: str) -> dict:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, async_turn(session, user_message))
            return future.result()
    return asyncio.run(async_turn(session, user_message))


# ─────────────────────── Async core ───────────────────────────────────────

async def async_turn(session: Session, user_message: str) -> dict:
    session.turn_count += 1
    turn_index = session.turn_count

    # Clean any half-finished slot from a prior crash.
    while session.history and session.history[-1][1] == "":
        session.history.pop()

    window = _build_window(session.history, n=2)
    pre_turn_context = session.graph.cbt_context()
    snap = session.graph.snapshot()
    current_phase = snap.get("session_phase") or "Rapport"

    system_prompt = THERAPIST_SYSTEM.format(cbt_context=pre_turn_context)

    session.history.append((user_message, ""))
    session.transcript.append((turn_index, "client", user_message))

    extraction_task = asyncio.create_task(
        _run_extraction(session, user_message, window, turn_index)
    )
    generate_task = asyncio.create_task(
        _run_generate(session.generator, system_prompt, session.history)
    )

    try:
        result = await generate_task
    except Exception:
        if session.history and session.history[-1][1] == "":
            session.history.pop()
        extraction_task.cancel()
        raise

    current_technique = snap.get("active_technique") or "Rapport Building"
    reply     = result.get("response", "")
    technique = result.get("technique") or current_technique
    # The model's own "phase" self-report is not trustworthy as the sole
    # advancement signal: it's hard-coded into its own context ("CURRENT
    # SESSION STATE: Session phase: Rapport") and in practice it mostly just
    # echoes that value back rather than reasoning about whether to graduate
    # — confirmed live: turns with a Problem node + turn_count >= 2 (i.e.
    # Exploration's gate already satisfied) still got "phase": "Rapport" back
    # on turns where JSON parsing succeeded, so relying on the model to ever
    # *propose* "Exploration" left the session stuck even though the real
    # (deterministic, node-grounded) gate in validate_phase would have granted
    # it. So we always propose at least the next phase in PHASE_ORDER every
    # turn — validate_phase's node-count/turn-count gate is the actual
    # authority and will reject it if unearned — while still respecting the
    # model if it ever proposes something further ahead than that.
    idx_current = PHASE_ORDER.index(current_phase) if current_phase in PHASE_ORDER else 0
    next_phase = PHASE_ORDER[min(idx_current + 1, len(PHASE_ORDER) - 1)]
    model_phase = result.get("phase")
    if model_phase in PHASE_ORDER and PHASE_ORDER.index(model_phase) > PHASE_ORDER.index(next_phase):
        proposed = model_phase
    else:
        proposed = next_phase
    session.history[-1] = (user_message, reply)
    session.transcript.append((turn_index, "therapist", reply))

    extraction_mode = "sync" if extraction_task.done() else "async"
    try:
        extraction_result = await extraction_task or {}
    except Exception as exc:
        print(f"[therapy] extraction failed: {type(exc).__name__}: {exc}")
        extraction_result = {}

    validated_phase = validate_phase(proposed, current_phase,
                                     session.graph, session.turn_count)
    session.graph.apply_session_state(validated_phase, technique)

    # Fire Tier B every CONSOLIDATE_EVERY turns (detached — does NOT block).
    if (session.turn_count % CONSOLIDATE_EVERY) == 0:
        asyncio.create_task(_run_consolidate(session))

    return {
        "reply": reply,
        "technique": technique,
        "phase": validated_phase,
        "extraction_mode": extraction_mode,
        "new_nodes": extraction_result.get("new_nodes", []),
        "new_edges": extraction_result.get("edges", []),
        "graph_snapshot": session.graph.snapshot(),
        # Only set by SteeredRemoteGenerator ("steered"/"fallback"/"none"); absent for
        # EchoGenerator/LocalLLMGenerator, which don't have a steering concept.
        "steer_status": result.get("steer_status"),
    }


async def _run_generate(generator: Generator, system: str,
                         history: list[tuple[str, str]]) -> dict:
    return await asyncio.to_thread(generator.generate, system, history)


async def _run_extraction(session: Session, message: str,
                           window: list[tuple[str, str]],
                           turn_index: int) -> dict:
    async with session.extraction_lock:
        try:
            return await asyncio.to_thread(
                session.extractor.process_turn,
                message, window, session.graph, turn_index,
            )
        except Exception as exc:
            print(f"[therapy] process_turn raised: {type(exc).__name__}: {exc}")
            return {"new_nodes": [], "edges": [], "error": str(exc)}


async def _run_consolidate(session: Session) -> None:
    async with session.extraction_lock:
        try:
            await asyncio.to_thread(
                session.extractor.consolidate, list(session.transcript), session.graph,
            )
        except Exception as exc:
            print(f"[therapy] consolidate raised: {type(exc).__name__}: {exc}")


def _build_window(history: list[tuple[str, str]], n: int = 2) -> list[tuple[str, str]]:
    completed = [(u, a) for u, a in history if a]
    recent = completed[-n:]
    window = []
    for user, assistant in recent:
        window.append(("client", user))
        window.append(("therapist", assistant))
    return window
