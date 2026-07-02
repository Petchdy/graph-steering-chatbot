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

from cbt_ontology_v4_flat import (Turn, Node, CLASS_DEFINITIONS, SUBCLASSED,
                                  SUBCLASS_GLOSS, SUBCLASS_RULES, turn_texts, render_turns)

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
BATCH = 1
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

_VALIDATE_PROMPT = """You verify whether extracted CBT entities belong to their assigned
class, and fix the label when they don't. Entity texts/evidence are in Thai.

For each candidate, the class DEFINITION is given. Decide one verdict:
- "keep"       — the text genuinely fits its assigned class.
- "drop"       — the text is not a real instance of any class (noise, a fragment, a
                 therapist question, a repeated filler line).
- "reclassify" — the text is real but belongs to a DIFFERENT class. Give the correct
                 newLabel.

Use "reclassify" mainly inside the belief/thought family. The boundary:
- CoreBelief: an ABSOLUTE identity claim — "I am worthless", "people can't be trusted".
  No "if", no "must".
- IntermediateBelief: a rule ("I must/should..."), an assumption ("if...then..."), or
  an attitude ("it's terrible to..."). Conditional or instrumental.
- AutomaticThought: a spontaneous thought tied to ONE moment. Not a rule, not an
  absolute identity claim.
A statement containing "I must / I should / I have to / if ... then" that was labeled
CoreBelief or AutomaticThought is almost always an IntermediateBelief — reclassify it.
A bare identity claim ("I am unlovable") labeled IntermediateBelief or AutomaticThought
is a CoreBelief — reclassify it.

When genuinely unsure on a borderline-but-plausible case, "keep".

CLASS-SPECIFIC GUIDANCE (apply only to the relevant class):
- Situation: a trigger context. Acceptable forms include a single concrete moment
  ("the breakup yesterday"), a state of being ("being alone"), a recurring trigger
  pattern ("looking around at friends in relationships"), or a recalled past event.
  Be lenient: if it names something the client experiences that could plausibly trigger
  a thought, KEEP. Only drop when it is clearly NOT a trigger context.
- Problem: a session-agenda heading — an ongoing area of difficulty. A Problem may name
  an underlying fear or belief ("fear of being alone leading to people-pleasing")
  without that disqualifying it. Drop only when the text is purely a momentary thought
  or a single Reaction, not a recurring theme.

{candidates}

Return one object per candidate:
[{{"item": 1, "verdict": "keep", "newLabel": "", "reason": "<short>"}}, ...]
verdict is exactly "keep", "drop", or "reclassify". Set newLabel only when
reclassifying (one of CoreBelief, IntermediateBelief, AutomaticThought); otherwise "".{reminder}"""


_RECLASSIFY_TARGETS = frozenset({"CoreBelief", "IntermediateBelief", "AutomaticThought"})


def validate_nodes(by_label: dict[str, list[Node]], turns: list[Turn],
                   llm: ChatOllama | None = None) -> tuple[dict[str, list[Node]], list[dict]]:
    llm = llm or _llm()
    kept: dict[str, list[Node]] = {lbl: [] for lbl in by_label}
    # Ensure reclassify targets always have a bucket, even if absent from input.
    for tgt in _RECLASSIFY_TARGETS:
        kept.setdefault(tgt, [])
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
            # default: parse-fail -> keep as-is
            verdicts: dict[int, tuple[str, str, str]] = {
                i: ("keep", "", "parse-fail kept") for i in range(1, len(batch) + 1)
            }
            for it in _parse(llm.invoke("/no_think\n" + prompt).content):
                if isinstance(it, dict) and isinstance(it.get("item"), int):
                    idx = it["item"]
                    if 1 <= idx <= len(batch):
                        v = str(it.get("verdict", "")).strip().lower()
                        new_label = str(it.get("newLabel", "")).strip()
                        reason = str(it.get("reason", "")).strip()
                        verdicts[idx] = (v, new_label, reason)
            for i, n in enumerate(batch, 1):
                verdict, new_label, reason = verdicts[i]
                if verdict == "drop":
                    dropped.append({"id": n.id, "label": label, "text": n.text,
                                    "reason": reason or "(no reason)"})
                elif verdict == "reclassify" and new_label in _RECLASSIFY_TARGETS and new_label != label:
                    print(f"[stage1.5] reclassify id={n.id} {label} -> {new_label}: "
                          f"{n.text[:60]}", file=sys.stderr)
                    n.label = new_label
                    kept.setdefault(new_label, []).append(n)
                else:
                    # keep, or unknown/missing newLabel on reclassify -> fail safe
                    kept[label].append(n)

    n_drop = len(dropped)
    print(f"[stage1.5] validation: dropped {n_drop} node(s)", file=sys.stderr)
    return kept, dropped


# ---------------------------------------------------------------------------
# (b) subclass classification — sets group_key (partition key)
# ---------------------------------------------------------------------------

_SUBCLASS_PROMPT = """You assign each CBT entity to exactly one subclass of {family}.
Entity texts/evidence are in Thai.

SUBCLASSES:
{gloss}

{candidates}

Return one object per candidate:
[{{"item": 1, "subclass": "<one value>"}}, ...].{reminder}"""


def classify_subclasses(by_label: dict[str, list[Node]], turns: list[Turn],
                        llm: ChatOllama | None = None) -> None:
    """Mutates Node.group_key in place for the 4 subclassed families."""
    llm = llm or _llm()
    for family in SUBCLASSED:
        nodes = by_label.get(family, [])
        if not nodes:
            continue
        gloss = "\n".join(f"  - {k}: {v}" for k, v in SUBCLASS_GLOSS[family].items())
        valid = set(SUBCLASS_RULES[family])
        for start in tqdm(range(0, len(nodes), BATCH),
                          desc=f"stage1.5 subclass {family}"):
            batch = nodes[start:start + BATCH]
            blocks = [f"CANDIDATE {i}: '{n.text}'\nEVIDENCE:\n{_evidence(n, turns)}"
                      for i, n in enumerate(batch, 1)]
            prompt = _SUBCLASS_PROMPT.format(family=family, gloss=gloss,
                                             candidates="\n\n".join(blocks),
                                             reminder=_JSON_REMINDER)
            for it in _parse(llm.invoke("/no_think\n" + prompt).content):
                if not isinstance(it, dict):
                    continue
                idx, sub = it.get("item"), str(it.get("subclass", "")).strip()
                if isinstance(idx, int) and 1 <= idx <= len(batch) and sub in valid:
                    batch[idx - 1].group_key = sub
        # nodes that stayed unclassified: fall back to the catch-all where one exists
        for n in nodes:
            if n.group_key not in valid:
                fallback = "other" if "other" in valid else None
                if fallback:
                    n.group_key = fallback
                    print(f"[stage1.5] {family} '{n.text[:40]}' unclassified -> {fallback}",
                          file=sys.stderr)


def run_stage1_5(by_label: dict[str, list[Node]], turns: list[Turn],
                 llm: ChatOllama | None = None) -> tuple[dict[str, list[Node]], list[dict]]:
    llm = llm or _llm()
    kept, dropped = validate_nodes(by_label, turns, llm)
    classify_subclasses(kept, turns, llm)
    return kept, dropped
