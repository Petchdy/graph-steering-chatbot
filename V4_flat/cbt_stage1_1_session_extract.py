"""
CBT KG — Stage 1.1 (V4_flat): session-level extraction (NEW).

Rationale: per-turn Stage 1 reads each turn with ±2 context. Concepts that only
emerge across multiple turns — laddered CoreBeliefs, session-level Interventions
(CACTUS technique label), AdaptiveResponses, Problem themes, Goals — get missed
or extracted as misleading surface fragments. This stage skims the WHOLE session
with each class's definition in mind and emits additional Nodes with the
most-related evidence turn(s) (1–3, not the whole span).

Additive design: Stage 1's per-turn output is preserved verbatim and passed in
as priors. Stage 1.1 adds new Nodes. Stage 1.2 then atomizes, Stage 1.5 validates,
Stage 2 merges and reconciles duplicates against Stage 1 outputs.

Classes targeted (session-spanning):
  CoreBelief, IntermediateBelief, Problem, Goal,
  Intervention, Homework, AdaptiveResponse

Per-turn-only classes (NOT extracted here): Situation, AutomaticThought, Reaction.

Adjacent-class priors: for each target class, the prompt also receives a small
context block of adjacent-class extractions (e.g., AutomaticThoughts when
extracting CoreBeliefs — the ladder rungs are useful context). This is opt-in
bias that may be revisited.

Chunking: when the rendered transcript exceeds CHAR_BUDGET (~3500 tokens worth
of chars across EN/TH), split into 2 halves with TURN_OVERLAP turns of overlap
and run each per-class extraction once per chunk. Outputs from both chunks are
deduplicated by (text, evidence_turns) and pass through Stage 1.5+Stage 2 as usual.

Fail mode: parse failure on any (class, chunk) call → skip that call's output
(per-class additive — silent skip can't poison; Stage 1 priors still flow through).

LLM: qwen3.5-nothink, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import count

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import Turn, Node, CLASS_DEFINITIONS

OLLAMA_TIMEOUT = 900

# Classes that benefit from session-level extraction.
SESSION_LEVEL_CLASSES = (
    "CoreBelief", "IntermediateBelief", "Problem", "Goal",
    "Intervention", "Homework", "AdaptiveResponse",
)

# For each target class, which other classes' Stage 1 outputs to include as
# adjacent-class priors (small, for grounding context).
ADJACENT_PRIORS: dict[str, tuple[str, ...]] = {
    "CoreBelief":         ("AutomaticThought",),
    "IntermediateBelief": ("CoreBelief", "AutomaticThought"),
    "Problem":            ("Situation", "Reaction"),
    "Goal":               ("Problem",),
    "Intervention":       ("AutomaticThought", "IntermediateBelief", "CoreBelief"),
    "Homework":           ("Problem", "AutomaticThought", "IntermediateBelief"),
    "AdaptiveResponse":   ("AutomaticThought", "Intervention"),
}

# Chunking constants (char-budget proxy for token budget).
# 1 token ≈ 4 chars (EN) or ≈ 1.5 chars (TH). We use a conservative bound that
# leaves ~30% headroom for the per-class prompt + priors + completion.
CHAR_BUDGET = 14000     # ≈ 3500 tokens English / ≈ 9000 tokens Thai
TURN_OVERLAP = 5

_JSON_REMINDER = ("\n\nOutput ONLY a JSON array. Start with [ and end with ]. "
                  "Empty [] if nothing new is supported.")


def _llm() -> ChatOllama:
    return ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)


def _strip_think(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _parse_array(raw: str) -> list | None:
    raw = _strip_think(raw)
    try:
        return json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _render_turns(turns: list[Turn]) -> str:
    """Render a (sub)list of turns as 'turn N | T/C: text' lines."""
    return "\n".join(
        f"turn {t.turn_index} | {'T' if t.speaker == 'therapist' else 'C'}: {t.text}"
        for t in turns
    )


def _chunks(turns: list[Turn]) -> list[list[Turn]]:
    """Split turns into chunks whose rendered char count fits CHAR_BUDGET.
    Adjacent chunks overlap by TURN_OVERLAP turns so a concept that straddles
    the boundary still appears whole in at least one chunk."""
    rendered = _render_turns(turns)
    if len(rendered) <= CHAR_BUDGET:
        return [turns]

    # Greedy bisect: keep splitting until each piece fits.
    pieces: list[list[Turn]] = []
    pending = [turns]
    while pending:
        cur = pending.pop(0)
        if len(_render_turns(cur)) <= CHAR_BUDGET or len(cur) <= TURN_OVERLAP * 2 + 1:
            pieces.append(cur)
            continue
        mid = len(cur) // 2
        left = cur[: mid + TURN_OVERLAP]                      # extend right edge
        right = cur[max(0, mid - TURN_OVERLAP):]              # extend left edge
        pending.insert(0, right)
        pending.insert(0, left)
    return pieces


def _format_priors(by_label: dict[str, list[Node]], classes: tuple[str, ...]) -> str:
    """Render priors as bullet lines per class. Empty classes are skipped."""
    blocks: list[str] = []
    for cls in classes:
        items = by_label.get(cls, [])
        if not items:
            continue
        lines = "\n".join(f"  - '{n.text}'  (turns {sorted(n.evidence)})" for n in items)
        blocks.append(f"{cls}:\n{lines}")
    return "\n\n".join(blocks) if blocks else "(none)"


_PROMPT = """You are reviewing a full CBT therapy session transcript to extract
{class_label} entities that span multiple turns or only become clear when you
see the whole session — concepts a per-turn extractor would miss. Texts are
in Thai (English glosses may appear).

DEFINITION of {class_label}: {class_definition}

ALREADY EXTRACTED per-turn — same class (these may be partial, duplicated, or
mis-classified; correct them if needed):
{same_class_priors}

ADJACENT CONTEXT — other classes for grounding only (do NOT re-extract these):
{adjacent_class_priors}

TRANSCRIPT:
{transcript}

EXTRACTION RULES:
1. Output the {class_label}s the session as a whole supports — including any
   the per-turn extractor missed.
2. For each output, pick 1–3 MOST-EVIDENTIARY turn indices where the concept is
   grounded. Do not list every mention; pick the strongest turns.
3. Therapist meta-commentary about CBT terminology ("the core belief is called
   X", "people will say Y", "this is what we call Z") is NOT a node — extract
   only what the client actually believes or experiences.
4. Stay faithful: do not invent concepts not grounded in real evidence turns.

Output JSON array, one object per {class_label}:
[{{"label":"{class_label}","text":"<short>","evidence_turns":[<int>,...]}}]
{reminder}"""


def _extract_class_for_chunk(
    class_label: str,
    chunk_turns: list[Turn],
    valid_turn_indices: set[int],
    by_label: dict[str, list[Node]],
    llm: ChatOllama,
    id_fn,
) -> list[Node]:
    """One LLM call for one class on one chunk. Returns new Nodes."""
    transcript = _render_turns(chunk_turns)
    same_priors = _format_priors(by_label, (class_label,))
    adjacent_priors = _format_priors(by_label, ADJACENT_PRIORS.get(class_label, ()))

    prompt = _PROMPT.format(
        class_label=class_label,
        class_definition=CLASS_DEFINITIONS.get(class_label, ""),
        same_class_priors=same_priors,
        adjacent_class_priors=adjacent_priors,
        transcript=transcript,
        reminder=_JSON_REMINDER,
    )
    raw = llm.invoke("/no_think\n" + prompt).content
    arr = _parse_array(raw)
    if arr is None:
        print(f"[stage1.1] parse fail on {class_label} chunk — skipping",
              file=sys.stderr)
        return []

    out: list[Node] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        lbl = str(it.get("label", "")).strip()
        if lbl != class_label:
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        # validate evidence turns
        raw_ev = it.get("evidence_turns") or []
        evidence: set[int] = set()
        for x in raw_ev:
            try:
                ti = int(x)
            except (TypeError, ValueError):
                continue
            if ti in valid_turn_indices:
                evidence.add(ti)
        if not evidence:
            # No grounded evidence — refuse (Stage 1.5 would drop it anyway,
            # and an evidenceless node fails the Stage 2 evidence-union assertion).
            print(f"[stage1.1] dropped {class_label} '{text[:40]}' — no valid evidence",
                  file=sys.stderr)
            continue
        out.append(Node(
            id=id_fn(),
            label=class_label,
            text=text,
            group_key=None,
            evidence=evidence,
            context={},                            # populated lazily downstream
            props={},
        ))
    return out


def _dedup_within_class(nodes: list[Node]) -> list[Node]:
    """Drop exact (text, evidence) duplicates from the same chunk-set. Stage 2
    will handle semantic dedup; this only collapses chunk-overlap repeats."""
    seen: set[tuple[str, frozenset[int]]] = set()
    out: list[Node] = []
    for n in nodes:
        key = (n.text.strip().lower(), frozenset(n.evidence))
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def run_stage1_1(by_label: dict[str, list[Node]], turns: list[Turn],
                 llm: ChatOllama | None = None) -> dict[str, list[Node]]:
    """Add session-level extractions to `by_label` (additive — Stage 1 outputs
    are preserved). Returns the augmented dict."""
    llm = llm or _llm()

    valid_turn_indices = {t.turn_index for t in turns}

    # Seed id counter above current max (Stage 1 may have produced up to N nodes).
    max_id = 0
    for ns in by_label.values():
        for n in ns:
            if n.id > max_id:
                max_id = n.id
    counter = count(max_id + 1)
    def _next_id() -> int:
        return next(counter)

    chunks = _chunks(turns)
    if len(chunks) > 1:
        print(f"[stage1.1] transcript split into {len(chunks)} chunks "
              f"(turns/chunk: {[len(c) for c in chunks]})", file=sys.stderr)

    out = dict(by_label)                                  # copy; we mutate per-class lists
    for class_label in tqdm(SESSION_LEVEL_CLASSES, desc="stage1.1 session-extract", unit="class"):
        added: list[Node] = []
        for ci, chunk in enumerate(chunks):
            new_nodes = _extract_class_for_chunk(
                class_label, chunk, valid_turn_indices, by_label, llm, _next_id,
            )
            if new_nodes and len(chunks) > 1:
                print(f"[stage1.1]   {class_label} chunk {ci+1}/{len(chunks)}: "
                      f"+{len(new_nodes)} nodes", file=sys.stderr)
            added.extend(new_nodes)
        added = _dedup_within_class(added)
        out.setdefault(class_label, [])
        out[class_label] = out[class_label] + added

    added_total = sum(len(out[c]) - len(by_label.get(c, [])) for c in SESSION_LEVEL_CLASSES)
    print(f"[stage1.1] session-level pass added {added_total} nodes "
          f"(across {len(SESSION_LEVEL_CLASSES)} target classes)", file=sys.stderr)
    return out
