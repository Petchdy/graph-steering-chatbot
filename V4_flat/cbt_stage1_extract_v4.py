"""
CBT KG — Stage 1 (V4): all-classes entity extraction, one call per turn.

Design imports:
  - EDC "Define" step: each class's definition is in the prompt (CLASS_DEFINITIONS).
  - PJKG default-value spec: the model is told exactly what to output when a field
    is absent (null), reducing hallucinated fills.

Stage 1 emits {label, text, group_key_guess?} only. The reliable partition key
(subclass) is set in Stage 1.5; properties (distortion/technique/kind/...) in 2.5.

LLM: llama3.1:8b, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import count

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import (Turn, Node, EXTRACT_CLASSES, CLASS_DEFINITIONS,
                                  SPEAKER_PRIOR, load_transcript, turn_texts, render_turns)

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
CONTEXT_WINDOW = 2   # ± turns of context shown around the target turn

_DEFS_BLOCK = "\n".join(f"- {c}: {CLASS_DEFINITIONS[c]}" for c in EXTRACT_CLASSES)

_JSON_REMINDER = ("\n\nOutput ONLY a JSON array. Start with [ and end with ]. "
                  "No other text. Empty array [] if the target turn has nothing to extract.")

PROMPT = """You extract CBT entities from a therapy transcript. The text is in Thai
(translated glosses may appear). Keep entity text faithful to the speaker's words;
extract the core concept, not whole sentences.

CLASS DEFINITIONS:
{defs}

GLOBAL RULES:
1. Extract ONLY what is stated in the TARGET TURN. Do not infer or embellish.
2. If nothing fits, return []. Never invent a node to fill a gap.
3. One node per distinct item. Do not split one statement into many, or merge two.
4. Keep emotions OUT of AutomaticThought content (the feeling is a separate Reaction).
5. Do NOT extract a therapist question as an AutomaticThought.
6. Speaker prior — this turn's speaker is {speaker}. Typically extract: {prior}.
   (Goals and AdaptiveResponses may come from either speaker.)

The surrounding CONTEXT is for understanding only; entity text must be grounded in
the TARGET TURN.

CONTEXT:
{context}

TARGET TURN [{idx}] ({speaker}): {target}

For each entity output an object:
  {{"label": "<one class name>", "text": "<core concept, faithful wording>",
    "group_key": "<your best subclass guess or null>"}}
group_key is optional — null is acceptable (it is re-checked later).{reminder}"""


def _extract_turn(turn: Turn, turns: list[Turn], llm: ChatOllama,
                  id_fn) -> list[Node]:
    ctx_rows = [r for r in turn_texts(turns, [turn.turn_index], window=CONTEXT_WINDOW)
                if r[0] != turn.turn_index]
    prompt = PROMPT.format(
        defs=_DEFS_BLOCK, speaker=turn.speaker,
        prior=", ".join(SPEAKER_PRIOR.get(turn.speaker, EXTRACT_CLASSES)),
        context=render_turns(ctx_rows), idx=turn.turn_index, target=turn.text,
        reminder=_JSON_REMINDER,
    )
    raw = llm.invoke("/no_think\n" + prompt).content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        print(f"[stage1] turn {turn.turn_index}: parse fail — {raw[:120]!r}", file=sys.stderr)
        return []

    ctx_blob = render_turns(turn_texts(turns, [turn.turn_index], window=CONTEXT_WINDOW))
    nodes: list[Node] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        label = it.get("label")
        text = (it.get("text") or "").strip()
        if label not in EXTRACT_CLASSES or not text:
            if label is not None:
                print(f"[stage1] turn {turn.turn_index}: skip {label!r}/{text[:40]!r}",
                      file=sys.stderr)
            continue
        gk = it.get("group_key")
        gk = gk.strip() if isinstance(gk, str) and gk.strip().lower() != "null" else None
        nodes.append(Node(id=id_fn(), label=label, text=text, group_key=gk,
                          evidence={turn.turn_index},
                          context={turn.turn_index: ctx_blob}))
    return nodes


def run_stage1(turns: list[Turn], llm: ChatOllama | None = None,
               max_workers: int = 1) -> dict[str, list[Node]]:
    """Returns {abstract_label: [Node, ...]}."""
    llm = llm or ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)
    _count = count(1)
    _lock = threading.Lock()
    def _next_id() -> int:
        with _lock:
            return next(_count)

    by_label: dict[str, list[Node]] = {c: [] for c in EXTRACT_CLASSES}

    def _process(t: Turn) -> list[Node]:
        return _extract_turn(t, turns, llm, _next_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for nodes in tqdm(executor.map(_process, turns), total=len(turns), desc="stage1 extract"):
            for n in nodes:
                by_label[n.label].append(n)

    total = sum(len(v) for v in by_label.values())
    print(f"[stage1] extracted {total} raw nodes across {len(turns)} turns", file=sys.stderr)
    return by_label


if __name__ == "__main__":
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else "../transcripts/demo1_transcript_translated_gemini.json"
    if os.path.exists(path):
        out = run_stage1(load_transcript(path))
        for lbl, ns in out.items():
            if ns:
                print(f"{lbl}: {len(ns)}")
