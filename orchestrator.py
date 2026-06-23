"""The turn loop. Depends only on interfaces.py — never on concrete implementations."""

import asyncio
import os
from dataclasses import dataclass, field

from interfaces import Extractor, GraphStore, Generator, Schema
from prompts import CBT_SYSTEM_PROMPT

PHASE_ORDER = ["Rapport", "Exploration", "Technique", "Consolidation"]

PHASE_MINIMUMS: dict[str, dict] = {
    "Exploration":   {"fields": ["presenting_problem"],                     "min_turns": 2},
    "Technique":     {"fields": ["presenting_problem", "negative_thought"],  "min_turns": 5},
    "Consolidation": {"fields": ["negative_thought", "cognitive_pattern"],   "min_turns": 12},
}

EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "8"))


def validate_phase(proposed: str, current: str, snapshot: dict, turn_count: int) -> str:
    """Accept proposed phase only if the minimum field/turn requirements are met."""
    try:
        if PHASE_ORDER.index(proposed) <= PHASE_ORDER.index(current):
            return proposed
    except ValueError:
        return current
    mins = PHASE_MINIMUMS.get(proposed, {})
    fields_met = all(snapshot.get(f, {}).get("acquired") for f in mins.get("fields", []))
    turns_met = turn_count >= mins.get("min_turns", 0)
    return proposed if (fields_met and turns_met) else current


@dataclass
class Session:
    schema: Schema
    graph: GraphStore
    extractor: Extractor
    generator: Generator
    history: list[tuple[str, str]] = field(default_factory=list)
    turn_count: int = 0
    extraction_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── Sync wrapper (keeps existing tests + Gradio UI working) ───────────────────

def turn(session: Session, user_message: str) -> dict:
    """Sync entry point. Calls async_turn in a new event loop or via thread executor."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, async_turn(session, user_message))
            return future.result()
    else:
        return asyncio.run(async_turn(session, user_message))


# ── Async core ────────────────────────────────────────────────────────────────

async def async_turn(session: Session, user_message: str) -> dict:
    """
    Main turn logic. Extraction and generation run concurrently.

    Option 1 (sync-opportunistic): extraction finishes within EXTRACTION_TIMEOUT →
      graph is updated before generate completes → snapshot reflects new state.
    Option 2 (async fallback): generate returns first → extraction continues in
      background → graph updates eventually.
    """
    session.turn_count += 1

    window = _build_window(session.history, n=2)

    # Defensive: clean up half-finished turn from a prior crash.
    while session.history and session.history[-1][1] == "":
        session.history.pop()

    pre_turn_context = session.graph.cbt_context()
    pre_turn_snapshot = session.graph.snapshot()
    current_phase = (pre_turn_snapshot.get("session_phase") or {}).get("value") or "Rapport"

    system_prompt = CBT_SYSTEM_PROMPT.format(cbt_context=pre_turn_context)

    session.history.append((user_message, ""))

    schema_text = session.schema.render()
    ontology_text = session.schema.render_ontology()

    extraction_task = asyncio.create_task(
        _run_extraction(session, user_message, window, schema_text, ontology_text)
    )
    generate_task = asyncio.create_task(
        _run_generate(session.generator, system_prompt, session.history)
    )

    try:
        result = await generate_task
    except Exception as exc:
        # Drop the pending history slot so the next turn isn't poisoned.
        if session.history and session.history[-1][1] == "":
            session.history.pop()
        extraction_task.cancel()
        raise

    reply     = result.get("response", "")
    technique = result.get("technique", "Rapport Building")
    proposed  = result.get("phase", "Rapport")
    session.history[-1] = (user_message, reply)

    extraction_mode = "sync" if extraction_task.done() else "async"
    try:
        deltas = await extraction_task or {}
    except Exception as exc:
        print(f"[orchestrator] extraction failed: {type(exc).__name__}: {exc}")
        deltas = {}

    validated_phase = validate_phase(proposed, current_phase,
                                     session.graph.snapshot(), session.turn_count)
    session.graph.apply_session_state(validated_phase, technique)

    return {
        "reply": reply,
        "technique": technique,
        "phase": validated_phase,
        "deltas": deltas,
        "slots": session.graph.snapshot(),
        "extraction_mode": extraction_mode,
    }


async def _run_generate(generator: Generator, system: str,
                        history: list[tuple[str, str]]) -> dict:
    return await asyncio.to_thread(generator.generate, system, history)


async def _run_extraction(session: Session, message: str,
                          window: list[tuple[str, str]],
                          schema_text: str, ontology_text: str) -> dict:
    """Full extraction pipeline. Runs under per-session lock.

    Per-node edge resolution was an LLM call per node — easily 5+ extra Ollama
    requests per turn that blew past the timeout. For now we only upsert nodes
    and auto-resolve any placeholder edges between two found nodes whose
    (subj_label, pred, obj_label) tuple is in the ontology.
    """
    async with session.extraction_lock:
        try:
            node_candidates = await asyncio.to_thread(
                session.extractor.extract_nodes, message, window, ontology_text
            )
        except Exception as exc:
            print(f"[extract_nodes] failed: {type(exc).__name__}: {exc}")
            node_candidates = []

        try:
            flat_deltas = await asyncio.to_thread(
                session.extractor.extract, message, schema_text
            )
        except Exception as exc:
            print(f"[extract] failed: {type(exc).__name__}: {exc}")
            flat_deltas = {}

        session.graph.apply_deltas(flat_deltas, session.turn_count)

        for candidate in node_candidates:
            label = candidate.get("label", "")
            props = candidate.get("props", {})
            if not label or not props:
                continue
            session.graph.upsert_node(label, props, session.turn_count)

        # Auto-resolve placeholder edges between any two found nodes.
        _auto_resolve_edges(session.graph, session.schema, session.turn_count)

        return flat_deltas


def _auto_resolve_edges(graph, schema, turn_id: int) -> None:
    """Mark a placeholder edge 'found' when both endpoint placeholders are now
    'found'. Conservative: only fires on the canonical placeholder pair (i.e.
    the edges pre-created in graph.reset()), not on every multi-instance pair."""
    found_ids = {n.node_id for n in graph.nodes() if n.status == "found"}
    for e in graph.edges():
        if e.status == "found":
            continue
        if e.subject_id in found_ids and e.object_id in found_ids:
            graph.resolve_edge(e.subject_id, e.predicate, e.object_id, turn_id)


def _build_window(history: list[tuple[str, str]], n: int = 2) -> list[tuple[str, str]]:
    """Return last n complete exchanges as [(role, text), ...] pairs."""
    completed = [(u, a) for u, a in history if a]
    recent = completed[-n:]
    window = []
    for user, assistant in recent:
        window.append(("client", user))
        window.append(("therapist", assistant))
    return window
