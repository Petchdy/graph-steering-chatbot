"""
CBT KG — Stage 3 (V4): subject-anchored chain extraction.

Anchor on EVERY subject node (not just Situation) so nothing is missed — the
laddering case (at_1..at_5 all stemsFrom one core belief) proves a Situation-only
anchor is too narrow. For each subject node, ONE call lists all its valid object
candidates grouped by relation, and asks which real edges exist. Seeing the whole
local neighbourhood in one call gives chain coherence; partial and empty are valid
(an AutomaticThought with no Reaction, a Situation with no thought, etc.).

Three passes:
  A. Subject-anchored chain + hinge (ANCHOR_FAMILIES).
  B. reinforces — wide-window: all Reaction × all CoreBelief in one call.
  C. Deterministic structure: hasSession, hasProblem/Intervention/Homework
     (single-session = every such node belongs to the one session).

Belief candidates (CoreBelief / IntermediateBelief) are shown in FULL (not turn-
windowed) because laddering links thoughts to a belief reached many turns later.

LLM: llama3.1:8b, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import (Turn, Node, Edge, ANCHOR_FAMILIES, REINFORCES,
                                  leaf_label, turn_texts, render_turns)

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
MAX_EVIDENCE = 5
_JSON_REMINDER = ("\n\nOutput ONLY a JSON array starting with [ and ending with ]. "
                  "Empty array [] is a correct answer when no relationship holds.")


def _llm() -> ChatOllama:
    return ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)


def _node_line(i: int, n: Node) -> str:
    return f"  {i}. '{n.text}'  (turns {sorted(n.evidence)})"


# ---------------------------------------------------------------------------
# Pass A — subject-anchored
# ---------------------------------------------------------------------------

_ANCHOR_PROMPT = """You extract CBT relationships from a therapy transcript (Thai).

SUBJECT ({subj_label}): '{subj_text}'   (turns {subj_turns})

CONTEXT (subject turns ± nearby):
{context}

For the subject above, decide which of the following relationships hold. Each
relationship lists its candidate OBJECT nodes (numbered within that relationship).
Assert a relationship when the transcript supports the directional link from THIS
subject to THAT object. Do not invent links from pure theme; but do not under-
extract either — Beck's basic cognitive model expects these canonical edges:

  - a Situation usually `triggers` at least one AutomaticThought
  - an AutomaticThought usually `leadsTo` at least one Reaction
  - an AutomaticThought usually `stemsFrom` at least one CoreBelief (the ladder)
  - a CoreBelief usually `givesRiseTo` at least one IntermediateBelief
  - a Problem usually `manifestsAs` at least one specific Situation
  - an Intervention is `appliedTo` whatever target it surfaced or examined
  - a Homework `targets` whatever it works on

If a candidate plausibly fits one of these canonical patterns and the topic is
consistent across the subject's and object's evidence turns, fire the edge.
Always include `evidence_turns` (the turn indices grounding the assertion) so
the verifier can check; an empty list will get the edge dropped.

An empty array is correct only when the transcript clearly does not support any
candidate (e.g. the subject and all candidates are about entirely different topics).

{families}

Output one object per asserted edge:
  {{"relation": "<name>", "object": <number within that relation's list>,
    "evidence_turns": [..], "reason": "<short>"{intensity}}}
{reminder}"""


def _extract_subject(subj: Node, candidates: dict[str, list[Node]],
                     turns: list[Turn], llm: ChatOllama,
                     audit: list[dict] | None = None) -> list[Edge]:
    families = ANCHOR_FAMILIES.get(subj.label, [])
    # build per-relation candidate lists (dedupe object nodes per relation name,
    # but keep separate numbering per relation block)
    blocks: list[str] = []
    rel_objs: dict[str, list[Node]] = {}     # relation -> ordered object nodes
    for (pred, obj_label, hint) in families:
        objs = candidates.get(obj_label, [])
        objs = [o for o in objs if o.id != subj.id]
        if not objs:
            continue
        rel_objs.setdefault(pred, [])
        # for a predicate with multiple object classes (targets, appliedTo), append
        existing_ids = {o.id for o in rel_objs[pred]}
        for o in objs:
            if o.id not in existing_ids:
                rel_objs[pred].append(o)
    if not rel_objs:
        if audit is not None:
            audit.append({
                "subject_id": subj.id, "subject_label": subj.label,
                "subject_text": subj.text,
                "predicates": {}, "edges_proposed": 0, "note": "no candidates",
            })
        return []

    # render blocks with combined hints per predicate
    hint_by_pred: dict[str, str] = {}
    for (pred, obj_label, hint) in families:
        hint_by_pred.setdefault(pred, hint)
    for pred, objs in rel_objs.items():
        lines = "\n".join(_node_line(i, o) for i, o in enumerate(objs, 1))
        blocks.append(f"RELATION {pred} — {hint_by_pred[pred]}\n{lines}")

    intensity = (', "reportedIntensity": "<text or omit>"'
                 if "leadsTo" in rel_objs else "")
    ctx = render_turns(turn_texts(turns, subj.evidence, window=2))
    prompt = _ANCHOR_PROMPT.format(
        subj_label=leaf_label(subj.label, subj.group_key), subj_text=subj.text,
        subj_turns=sorted(subj.evidence), context=ctx,
        families="\n\n".join(blocks), intensity=intensity, reminder=_JSON_REMINDER)

    raw = llm.invoke("/no_think\n" + prompt).content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        print(f"[stage3] subject {subj.id} ({subj.label}): parse fail", file=sys.stderr)
        if audit is not None:
            audit.append({
                "subject_id": subj.id, "subject_label": subj.label,
                "subject_text": subj.text,
                "predicates": {p: len(os) for p, os in rel_objs.items()},
                "edges_proposed": 0, "note": "parse-fail",
            })
        return []

    edges: list[Edge] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        pred = str(it.get("relation", "")).strip()
        objs = rel_objs.get(pred)
        num = it.get("object")
        if not objs or not isinstance(num, int) or not (1 <= num <= len(objs)):
            continue
        obj = objs[num - 1]
        ev = {int(t) for t in it.get("evidence_turns", []) if isinstance(t, (int, float))}
        ev &= (subj.evidence | obj.evidence)         # clamp to endpoints' turns
        if not ev:
            # LLM omitted evidence_turns OR everything got clamped out.
            # Default to the subject's turns: the edge is asserted, and the
            # subject's evidence is the minimum reasonable grounding. Without
            # this, Stage 4b verifier drops the edge as "no evidence provided".
            ev = set(subj.evidence)
        if len(ev) > MAX_EVIDENCE:
            ev = set(sorted(ev)[:MAX_EVIDENCE])
        e = Edge(predicate=pred, subject_id=subj.id, object_id=obj.id, evidence=ev,
                 reason=str(it.get("reason", "")).strip())
        if pred == "leadsTo" and isinstance(it.get("reportedIntensity"), str):
            ri = it["reportedIntensity"].strip()
            if ri and ri.lower() != "omit":
                e.properties["reportedIntensity"] = ri
        edges.append(e)
    if audit is not None:
        proposed_by_pred: dict[str, int] = {}
        for e in edges:
            proposed_by_pred[e.predicate] = proposed_by_pred.get(e.predicate, 0) + 1
        audit.append({
            "subject_id": subj.id, "subject_label": subj.label,
            "subject_text": subj.text,
            "predicates": {p: {"candidates": len(os),
                               "proposed": proposed_by_pred.get(p, 0)}
                           for p, os in rel_objs.items()},
            "edges_proposed": len(edges),
            "note": "",
        })
    return edges


# ---------------------------------------------------------------------------
# Pass B — reinforces (wide-window)
# ---------------------------------------------------------------------------

_REINFORCES_PROMPT = """In this CBT session, which client REACTIONS are maintaining
or strengthening which CORE BELIEFS (a feedback loop that keeps the belief in
place)? Texts are in Thai. Assert a pair only when the transcript shows the
reaction feeding back to the belief. Many sessions have none.

REACTIONS:
{reactions}

CORE BELIEFS:
{beliefs}

Output one object per pair: [{{"reaction":<n>,"belief":<n>,"reason":"<short>"}}, ...].{reminder}"""


def _extract_reinforces(reactions: list[Node], beliefs: list[Node],
                        llm: ChatOllama) -> list[Edge]:
    if not reactions or not beliefs:
        return []
    prompt = _REINFORCES_PROMPT.format(
        reactions="\n".join(_node_line(i, n) for i, n in enumerate(reactions, 1)),
        beliefs="\n".join(_node_line(i, n) for i, n in enumerate(beliefs, 1)),
        reminder=_JSON_REMINDER)
    raw = llm.invoke("/no_think\n" + prompt).content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    edges: list[Edge] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        ri, bi = it.get("reaction"), it.get("belief")
        if (isinstance(ri, int) and 1 <= ri <= len(reactions)
                and isinstance(bi, int) and 1 <= bi <= len(beliefs)):
            r, b = reactions[ri - 1], beliefs[bi - 1]
            edges.append(Edge(predicate="reinforces", subject_id=r.id, object_id=b.id,
                              evidence=r.evidence | b.evidence,
                              reason=str(it.get("reason", "")).strip()))
    return edges


# ---------------------------------------------------------------------------
# Pass C — deterministic structure
# ---------------------------------------------------------------------------

def _structure_edges(survivors: dict[str, list[Node]], client_id: int,
                     session_id: int) -> list[Edge]:
    edges = [Edge("hasSession", client_id, session_id)]
    for label, pred in (("Problem", "hasProblem"), ("Intervention", "hasIntervention"),
                        ("Homework", "hasHomework")):
        for n in survivors.get(label, []):
            edges.append(Edge(pred, session_id, n.id))
    return edges


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_stage3(survivors: dict[str, list[Node]], turns: list[Turn],
               client_id: int, session_id: int,
               llm: ChatOllama | None = None) -> tuple[list[Edge], list[dict]]:
    """Return (edges, audit). audit is a per-subject record (subject id, label,
    candidates per predicate, edges proposed) — for the report."""
    llm = llm or _llm()
    edges: list[Edge] = []
    audit: list[dict] = []

    # Pass A — anchor on every subject node that has outgoing families
    subjects = [n for label in ANCHOR_FAMILIES for n in survivors.get(label, [])]
    for subj in tqdm(subjects, desc="stage3 anchored"):
        edges.extend(_extract_subject(subj, survivors, turns, llm, audit=audit))

    # Pass B — reinforces wide-window
    edges.extend(_extract_reinforces(survivors.get("Reaction", []),
                                     survivors.get("CoreBelief", []), llm))

    # Pass C — deterministic structure
    edges.extend(_structure_edges(survivors, client_id, session_id))

    print(f"[stage3] {len(edges)} candidate edges "
          f"({len(subjects)} anchored subjects)", file=sys.stderr)
    return edges, audit
