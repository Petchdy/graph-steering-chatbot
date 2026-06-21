"""
CBT KG — Stage 1.5 (V4_flat): validate (keep/drop) ONLY.

Per PIPELINE_UPDATE_v4_flat.md: under the flat ontology, the discriminator
(domain / subtype / channel) is no longer set here — it is assigned by Stage 2.5
after merge. Stage 1.5's only job is keep/drop validation against the class
definition (OL-KGC "verbalize the rule, then judge" prompt style — the paper's
highest-impact ontology factor). Fails are DROPPED: bad nodes poison Stage 2/3.

Properties (kind / distortion / technique / valence / temporality / category /
taskType / modality / isOptional / domain / channel / subtype) are all Stage 2.5.

LLM: qwen3.5-nothink, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import (Turn, Node, CLASS_DEFINITIONS,
                                  turn_texts, render_turns)

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
BATCH = 6
_JSON_REMINDER = "\n\nOutput ONLY a JSON array starting with [ and ending with ]."


def _llm() -> ChatOllama:
    return ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)


def _evidence(node: Node, turns: list[Turn]) -> str:
    return render_turns(turn_texts(turns, node.evidence, window=1))


def _parse(raw: str) -> list:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# (a) node validation — keep/drop
# ---------------------------------------------------------------------------

_VALIDATE_PROMPT = """You verify whether extracted CBT entities truly belong to
their assigned class. Entity texts/evidence are in Thai.

For each candidate, the class DEFINITION is given. Judge whether the candidate text
genuinely fits THAT class (not a neighbouring class). When the text really belongs
to a different class, or is not a real instance, answer "drop". When in doubt about
a borderline-but-plausible case, answer "keep".

CLASS-SPECIFIC GUIDANCE (apply only to the relevant class):
- Situation: a trigger context. Acceptable forms include a single concrete moment
  ("the breakup yesterday"), a state of being ("being alone"), a recurring trigger
  pattern ("looking around at friends in relationships"), or a recalled past event.
  Be lenient: if it names something the client experiences that could plausibly
  trigger a thought, KEEP. Only drop when it is clearly NOT a trigger context.
- Problem: a session-agenda heading — an ongoing area of difficulty. A Problem
  may name an underlying fear or belief ("fear of being alone leading to people-
  pleasing") without that disqualifying it as a Problem. Drop only when the text
  is purely a momentary thought or a single Reaction, not a recurring theme.

{candidates}

Return one object per candidate:
[{{"item": 1, "verdict": "keep", "reason": "<short>"}}, ...]
verdict is exactly "keep" or "drop".{reminder}"""


def validate_nodes(by_label: dict[str, list[Node]], turns: list[Turn],
                   llm: ChatOllama | None = None) -> tuple[dict[str, list[Node]], list[dict]]:
    llm = llm or _llm()
    kept: dict[str, list[Node]] = {lbl: [] for lbl in by_label}
    dropped: list[dict] = []

    for label, nodes in by_label.items():
        if not nodes:
            continue
        definition = CLASS_DEFINITIONS.get(label, "")
        for start in tqdm(range(0, len(nodes), BATCH),
                          desc=f"stage1.5 validate {label}", unit="batch"):
            batch = nodes[start:start + BATCH]
            blocks = [
                f"CANDIDATE {i} — assigned class: {label}\n"
                f"DEFINITION of {label}: {definition}\n"
                f"TEXT: '{n.text}'\nEVIDENCE:\n{_evidence(n, turns)}"
                for i, n in enumerate(batch, 1)
            ]
            prompt = _VALIDATE_PROMPT.format(candidates="\n\n".join(blocks),
                                             reminder=_JSON_REMINDER)
            verdicts = {i: (True, "parse-fail kept") for i in range(1, len(batch) + 1)}
            for it in _parse(llm.invoke("/no_think\n" + prompt).content):
                if isinstance(it, dict) and isinstance(it.get("item"), int):
                    idx = it["item"]
                    if 1 <= idx <= len(batch):
                        verdicts[idx] = (str(it.get("verdict", "")).strip().lower() == "keep",
                                         str(it.get("reason", "")).strip())
            for i, n in enumerate(batch, 1):
                keep, reason = verdicts[i]
                if keep:
                    kept[label].append(n)
                else:
                    dropped.append({"id": n.id, "label": label, "text": n.text,
                                    "reason": reason or "(no reason)"})

    n_drop = len(dropped)
    print(f"[stage1.5] validation: dropped {n_drop} node(s)", file=sys.stderr)
    return kept, dropped


def run_stage1_5(by_label: dict[str, list[Node]], turns: list[Turn],
                 llm: ChatOllama | None = None) -> tuple[dict[str, list[Node]], list[dict]]:
    """Validate-only: keep/drop against class definition. No subclass classification."""
    llm = llm or _llm()
    return validate_nodes(by_label, turns, llm)
