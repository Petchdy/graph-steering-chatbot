"""
CBT KG — Stage 1.2 (V4_flat): atomize + normalize (NEW).

Rationale (see PIPELINE_UPDATE_v4_flat.md §Change 1): a Stage 1 node can carry
two concepts ("worthless + unlovable"); without splitting, Stage 2.5 picks one
`category` and loses the other, and merge can never collapse the kept one with
its real twin. This stage splits multi-concept AT/CoreBelief/IntermediateBelief
into atomic propositions, and normalizes Intervention text into a clean
one-sentence description that Stage 5 exports as `properties.description`.

Scope:
  - ATOMIZE_CLASSES   = {AutomaticThought, CoreBelief, IntermediateBelief}
                       — split into 1..MAX_SPLITS atomic Nodes (1 is the common case).
  - NORMALIZE_CLASSES = {Intervention, Homework}  — rephrase text 1:1 for clarity.
                       The prompt does NOT force a single sentence: clarity matters
                       more than brevity, since these often need to preserve
                       specifics (target, time bound, sequence).
  - Pass-through: Situation, Problem, Reaction, Goal, AdaptiveResponse.

Flat simplification: properties are not yet set (Stage 2.5 owns all of them),
so a split child inherits only `label`, `text` (its own clean proposition), and
COPIES of the parent's `evidence` and `context`. No property inheritance logic.
Each child stores `props["sourceText"]` = parent's raw text (audit field —
stripped from the JSON export at Stage 5).

Fail mode: parse failure → keep original (fail SAFE; spec §"Global invariants" item 6).

LLM: qwen3.5-nothink, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import count

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import Turn, Node, CLASS_DEFINITIONS, turn_texts, render_turns

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
BATCH = 6             # normalize_interventions batch size
MAX_SPLITS = 4        # hard cap on atomize output

ATOMIZE_CLASSES   = {"AutomaticThought", "CoreBelief", "IntermediateBelief"}
NORMALIZE_CLASSES = {"Intervention", "Homework"}

_UNIT = {
    "AutomaticThought":   "thought",
    "CoreBelief":         "belief",
    "IntermediateBelief": "belief",
}

_AT_SPECIFIC = (
    "Keep each thought situation-specific; keep emotions OUT "
    "(a feeling is a separate Reaction)."
)

_JSON_REMINDER_LIST = (
    "\n\nOutput ONLY a JSON array of strings. Start with [ and end with ]. "
    f"1 to {MAX_SPLITS} items. Most inputs are a single idea — a 1-item array is correct."
)

_JSON_REMINDER_OBJS = "\n\nOutput ONLY a JSON array starting with [ and ending with ]."


def _llm() -> ChatOllama:
    return ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)


def _evidence(n: Node, turns: list[Turn]) -> str:
    return render_turns(turn_texts(turns, n.evidence, window=1))


def _strip_think(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _parse_json_array(raw: str) -> list | None:
    """Return a parsed JSON array, or None on failure (caller decides fallback)."""
    raw = _strip_think(raw)
    try:
        return json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# (a) atomize — AT / CoreBelief / IntermediateBelief
# ---------------------------------------------------------------------------

_ATOMIZE_PROMPT = """You clean and split extracted CBT {class_label}s. Texts/evidence are in Thai.

DEFINITION of {class_label}: {class_definition}

Rewrite the text into one or more ATOMIC, self-contained, first-person propositions:
- SPLIT only when the text contains genuinely DISTINCT {unit}s (e.g. two separate
  beliefs). Do NOT split one idea into clauses.
- CONDENSE rambling into a short, clear statement.
- Stay FAITHFUL: add no meaning, infer nothing, do not generalize. {at_specific}
- Most inputs are a single idea — ONE cleaned proposition is the common, correct answer.

TEXT: '{node_text}'
EVIDENCE:
{evidence_turns}
{reminder}"""


def atomize_node(n: Node, turns: list[Turn], llm: ChatOllama, id_fn) -> list[Node]:
    """Return 1..MAX_SPLITS new Nodes that replace `n`. Fail SAFE."""
    class_label = n.label
    class_definition = CLASS_DEFINITIONS.get(class_label, "")
    prompt = _ATOMIZE_PROMPT.format(
        class_label=class_label,
        class_definition=class_definition,
        unit=_UNIT[class_label],
        at_specific=_AT_SPECIFIC if class_label == "AutomaticThought" else "",
        node_text=n.text,
        evidence_turns=_evidence(n, turns),
        reminder=_JSON_REMINDER_LIST,
    )
    raw = llm.invoke("/no_think\n" + prompt).content
    arr = _parse_json_array(raw)

    # Fail-safe: keep the original parent unchanged (same id, same text, sourceText audit).
    if arr is None:
        print(f"[stage1.2] atomize parse fail on {class_label} id={n.id} — keeping original",
              file=sys.stderr)
        n.props.setdefault("sourceText", n.text)
        return [n]

    # Clean strings, drop blanks, cap at MAX_SPLITS.
    cleaned: list[str] = []
    for item in arr:
        if isinstance(item, str):
            t = item.strip()
            if t:
                cleaned.append(t)
    if not cleaned:
        n.props.setdefault("sourceText", n.text)
        return [n]
    cleaned = cleaned[:MAX_SPLITS]

    # 1:1 case — rewrite text in place (still record sourceText audit).
    if len(cleaned) == 1:
        new_text = cleaned[0]
        if new_text != n.text:
            n.props.setdefault("sourceText", n.text)
            n.text = new_text
        return [n]

    # Genuine split — emit fresh ids; copy evidence + context.
    parent_text = n.text
    children: list[Node] = []
    for clean_text in cleaned:
        child = Node(
            id=id_fn(),
            label=class_label,
            text=clean_text,
            group_key=None,
            evidence=set(n.evidence),                 # copy
            context=dict(n.context),                  # copy
            props={"sourceText": parent_text},
        )
        children.append(child)
    return children


# ---------------------------------------------------------------------------
# (b) normalize — Intervention (1:1, never splits)
# ---------------------------------------------------------------------------

_NORMALIZE_PROMPT = """Rephrase each {class_label} description into a clearer, more concise
version that preserves the specifics (what the therapist did, who/what target it
applies to, any time-bound details). You may use one or a few sentences — clarity
matters more than brevity. DO NOT force a single sentence if it loses information.
Stay faithful; add no detail not present in the source. Texts/evidence are in Thai.

{candidates}

Return one object per item: [{{"item":1,"description":"<rephrased text>"}}, ...].{reminder}"""


def normalize_descriptions(nodes: list[Node], turns: list[Turn], llm: ChatOllama,
                           class_label: str) -> None:
    """Rewrite each node's text in place with a clearer rephrasing (1–few sentences).
    Stores original `n.text` on `n.props["sourceText"]`. Pass-through on parse fail."""
    if not nodes:
        return
    for start in tqdm(range(0, len(nodes), BATCH),
                      desc=f"stage1.2 normalize {class_label}", unit="batch"):
        batch = nodes[start:start + BATCH]
        blocks = [f"ITEM {i}: '{n.text}'\nEVIDENCE:\n{_evidence(n, turns)}"
                  for i, n in enumerate(batch, 1)]
        prompt = _NORMALIZE_PROMPT.format(class_label=class_label,
                                          candidates="\n\n".join(blocks),
                                          reminder=_JSON_REMINDER_OBJS)
        arr = _parse_json_array(llm.invoke("/no_think\n" + prompt).content)
        if arr is None:
            print(f"[stage1.2] normalize {class_label} parse fail — batch kept as-is",
                  file=sys.stderr)
            continue
        new_text_by_idx: dict[int, str] = {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("item")
            desc = str(it.get("description", "")).strip()
            if isinstance(idx, int) and 1 <= idx <= len(batch) and desc:
                new_text_by_idx[idx] = desc
        rewrites = 0
        for i, n in enumerate(batch, 1):
            new_text = new_text_by_idx.get(i)
            if new_text and new_text != n.text:
                n.props.setdefault("sourceText", n.text)
                n.text = new_text
                rewrites += 1
        if rewrites == 0:
            print(f"[stage1.2] normalize {class_label} batch returned no rewrites "
                  f"(check prompt vs source verbosity)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_stage1_2(by_label: dict[str, list[Node]], turns: list[Turn],
                 llm: ChatOllama | None = None) -> dict[str, list[Node]]:
    """Split ATOMIZE_CLASSES; normalize NORMALIZE_CLASSES; pass through others."""
    llm = llm or _llm()

    # Seed fresh-id counter above the current max so retired/raw ids stay disjoint.
    max_id = 0
    for ns in by_label.values():
        for n in ns:
            if n.id > max_id:
                max_id = n.id
    counter = count(max_id + 1)
    def _next_id() -> int:
        return next(counter)

    out: dict[str, list[Node]] = {}
    for label, nodes in by_label.items():
        if label in ATOMIZE_CLASSES and nodes:
            new_nodes: list[Node] = []
            for n in tqdm(nodes, desc=f"stage1.2 atomize {label}", unit="node"):
                new_nodes.extend(atomize_node(n, turns, llm, _next_id))
            out[label] = new_nodes
        elif label in NORMALIZE_CLASSES and nodes:
            normalize_descriptions(nodes, turns, llm, label)
            out[label] = nodes
        else:
            out[label] = nodes

    print(f"[stage1.2] atomize+normalize: "
          f"{sum(len(v) for v in by_label.values())} -> "
          f"{sum(len(v) for v in out.values())} nodes", file=sys.stderr)
    return out
