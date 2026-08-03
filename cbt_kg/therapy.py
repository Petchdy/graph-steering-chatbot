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
import uuid
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


MAX_CHECKPOINTS = int(os.environ.get("MAX_CHECKPOINTS", "40"))


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
    # Branching: state captured *before* each turn, so a client message can be
    # rewritten and replayed against the context it originally had.
    checkpoints: dict[int, dict] = field(default_factory=dict)
    branches: list[dict] = field(default_factory=list)
    active_branch: str | None = None


# ─────────────────────── Snapshot / restore ───────────────────────────────

def snapshot_session(session: Session) -> dict:
    """Everything that makes a timeline: dialogue, counters, and graph.

    `checkpoints` are carried along by reference — they are written once and
    never mutated in place, so branches can share them without copying.
    """
    return {
        "history": list(session.history),
        "transcript": list(session.transcript),
        "turn_count": session.turn_count,
        "graph_state": session.graph.export_state(),
        "checkpoints": dict(session.checkpoints),
    }


def restore_session(session: Session, snap: dict) -> None:
    session.history = list(snap["history"])
    session.transcript = list(snap["transcript"])
    session.turn_count = snap["turn_count"]
    session.checkpoints = dict(snap.get("checkpoints") or {})
    session.graph.import_state(snap["graph_state"])


def apply_graph(graph: GraphStore, nodes: list, edges: list) -> None:
    """Replace `graph` with a corrected one WITHOUT losing the placeholder scaffold.

    `replace_all` is literal — it swaps in exactly what it is given. That is right
    for `import_state`, whose snapshots carry their own placeholders, but wrong for
    an externally corrected graph: both sources of one (LiveGraphReader and the
    canvas `saveJSON` export) deliberately emit `found` items only. Applying that
    directly deleted every 'not yet discovered' placeholder, so a load-then-apply
    round trip silently erased the greyed-out scaffold from the therapy panel.

    Placeholders are kept by id, not by label: `upsert_node` flips a placeholder to
    'found' in place, reusing its id, so an id present in `nodes` means that class's
    placeholder was legitimately consumed and must not come back.

    Order is preserved as well as membership. The canvas lays nodes out by their
    position in this list — slot within a layer comes from list index — so simply
    concatenating the incoming nodes ahead of the kept ones re-sorted the graph and
    made the drawing jump around after an apply.
    """
    old_nodes = graph.nodes()
    old_edges = graph.edges()
    node_rank = {n.node_id: i for i, n in enumerate(old_nodes)}
    edge_rank = {e.edge_id: i for i, e in enumerate(old_edges)}

    incoming_node_ids = {n.node_id for n in nodes}
    kept_nodes = [n for n in old_nodes
                  if n.status == "missing" and n.node_id not in incoming_node_ids]

    merged_nodes = list(nodes) + kept_nodes
    live_ids = {n.node_id for n in merged_nodes}
    incoming_edge_ids = {e.edge_id for e in edges}
    # A placeholder edge whose endpoint was dropped by the correction would dangle.
    kept_edges = [e for e in old_edges
                  if e.status == "missing" and e.edge_id not in incoming_edge_ids
                  and e.subject_id in live_ids and e.object_id in live_ids]
    merged_edges = list(edges) + kept_edges

    # Stable sort: anything already in the graph keeps its position, genuinely new
    # items keep their incoming order at the end.
    tail = len(node_rank)
    merged_nodes.sort(key=lambda n: node_rank.get(n.node_id, tail))
    tail_e = len(edge_rank)
    merged_edges.sort(key=lambda e: edge_rank.get(e.edge_id, tail_e))

    graph.replace_all(merged_nodes, merged_edges)


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
    # Checkpoint BEFORE any mutation, so this turn can later be replayed with a
    # different client message against exactly the context it had here.
    session.checkpoints[session.turn_count + 1] = snapshot_session(session)
    if len(session.checkpoints) > MAX_CHECKPOINTS:
        for stale in sorted(session.checkpoints)[:-MAX_CHECKPOINTS]:
            del session.checkpoints[stale]

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
    # Record the turn as an Utterance node before extraction runs, so the turn
    # indices extraction writes into node.evidence already have something to
    # resolve against. Part 2 needs this to quote dialogue rather than just cite
    # turn numbers; Session.transcript is process-local and never reaches it.
    session.graph.record_utterance(turn_index, "client", user_message)

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
    session.graph.record_utterance(turn_index, "therapist", reply)

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

    # Referral guard: `reply` is ALREADY sanitized (SafeGenerator cleaned result["response"] before
    # this function saw it), so history/transcript/graph above are clean with no change here. The
    # verified footer is attached to the OUTWARD reply only — it must never enter session.history,
    # because serve_steer.py generates with no_repeat_ngram_size=3 over the whole sequence, so
    # footer n-grams in an assistant turn would ban the model from producing that phrasing itself,
    # and it starts imitating and mutating the format. See guard/NOTES.md.
    guard_info = result.get("guard")
    outward_reply = reply
    if guard_info and guard_info.get("needs_footer"):
        try:
            from guard.sanitize import footer_text
            footer = footer_text()
            if footer:
                outward_reply = f"{reply}\n\n{footer}" if reply else footer
        except Exception as exc:  # noqa: BLE001 — a guard bug must never break the turn
            print(f"[therapy] referral footer unavailable: {type(exc).__name__}: {exc}")

    return {
        "reply": outward_reply,
        "technique": technique,
        "phase": validated_phase,
        "extraction_mode": extraction_mode,
        "new_nodes": extraction_result.get("new_nodes", []),
        "new_edges": extraction_result.get("edges", []),
        "graph_snapshot": session.graph.snapshot(),
        # Only set by SteeredRemoteGenerator ("steered"/"fallback"/"none"); absent for
        # EchoGenerator/LocalLLMGenerator, which don't have a steering concept.
        "steer_status": result.get("steer_status"),
        # Only set when the referral guard is enabled (REFERRAL_GUARD != 0):
        # {needs_footer, crisis, removed}. Absent => guard off, and "reply" is byte-identical
        # to the pre-guard behaviour.
        "guard": guard_info,
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


# ─────────────────────── Branching (alternate client turns) ───────────────

def _client_message_at(session: Session, turn_index: int) -> str:
    for ti, speaker, text in session.transcript:
        if ti == turn_index and speaker == "client":
            return text
    idx = turn_index - 1
    if 0 <= idx < len(session.history):
        return session.history[idx][0]
    return ""


def _branch_label(turn_index: int, message: str) -> str:
    trimmed = " ".join((message or "").split())
    if len(trimmed) > 48:
        trimmed = trimmed[:48] + "…"
    return f"turn {turn_index}: {trimmed}" if trimmed else f"turn {turn_index}"


def _archive_current(session: Session, turn_index: int) -> None:
    """Persist the timeline as it stands so switching back returns to it."""
    snap = snapshot_session(session)
    if session.active_branch:
        for b in session.branches:
            if b["id"] == session.active_branch:
                b["state"] = snap
                return
    message = _client_message_at(session, turn_index)
    bid = uuid.uuid4().hex[:8]
    session.branches.append({
        "id": bid, "turn_index": turn_index, "message": message,
        "label": _branch_label(turn_index, message), "state": snap,
    })
    session.active_branch = bid


def edit_turn(session: Session, turn_index: int, new_message: str) -> dict:
    """Replay `turn_index` with a different client message.

    The context before that turn is preserved exactly (from its checkpoint);
    everything the original timeline said from that turn onward is kept as a
    switchable branch rather than discarded.
    """
    checkpoint = session.checkpoints.get(turn_index)
    if checkpoint is None:
        raise ValueError(
            f"no checkpoint for turn {turn_index} — it is either beyond the "
            f"session or older than the last {MAX_CHECKPOINTS} turns"
        )
    _archive_current(session, turn_index)
    restore_session(session, checkpoint)
    session.active_branch = None
    result = turn(session, new_message)
    bid = uuid.uuid4().hex[:8]
    session.branches.append({
        "id": bid, "turn_index": turn_index, "message": new_message,
        "label": _branch_label(turn_index, new_message),
        "state": snapshot_session(session),
    })
    session.active_branch = bid
    return {**result, "branch_id": bid, "branches": list_branches(session)}


def switch_branch(session: Session, branch_id: str) -> dict:
    """Restore a previously recorded timeline, graph and all."""
    target = next((b for b in session.branches if b["id"] == branch_id), None)
    if target is None:
        raise ValueError(f"unknown branch {branch_id}")
    if session.active_branch and session.active_branch != branch_id:
        # Keep whatever has been said since this branch was made current.
        for b in session.branches:
            if b["id"] == session.active_branch:
                b["state"] = snapshot_session(session)
                break
    restore_session(session, target["state"])
    session.active_branch = branch_id
    return {"branch_id": branch_id, "branches": list_branches(session)}


def list_branches(session: Session) -> list[dict]:
    return [{"id": b["id"], "turn_index": b["turn_index"],
             "message": b["message"], "label": b["label"],
             "active": b["id"] == session.active_branch}
            for b in session.branches]


def editable_turns(session: Session) -> list[int]:
    """Client turns that still have a checkpoint, so can be rewritten."""
    return sorted(t for t in session.checkpoints if t <= session.turn_count)
