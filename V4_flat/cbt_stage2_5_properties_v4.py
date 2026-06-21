"""
CBT KG — Stage 2.5 (V4_flat): post-merge property classification.

On merged canonical text (one call per concept, not per mention). Under flat,
Stage 2.5 also owns the four discriminators (run FIRST so category/valence can
gate on them).

LLM (batched, one property family per call):
  Discriminators (flat-specific):
  - Problem.domain                    (academic|work|social|family|financial|health|other)
  - CoreBelief.domain                 (self|world|others)
  - IntermediateBelief.subtype        (attitude|rule|assumption)
  - Reaction.channel                  (emotional|behavioral|physiological)

  Other properties:
  - AutomaticThought.distortionType   (PatternReframe 10 + none)
  - AutomaticThought.modality         (verbal | image)
  - Situation.kind                    (6 channels)
  - Intervention.technique            (CACTUS 12 + other, +techniqueLabel)
  - Homework.taskType                 (7 types)
  - Homework.isOptional               (bool; true only on explicit optional framing)
  - CoreBelief.category               (helpless|unlovable|worthless) — only when domain="self"

Deterministic (ontology rule — NOT LLM):
  - Reaction.valence                  (Thai/EN emotion lexicon) — only when channel="emotional"
  - Situation.temporality             (Thai/EN time-marker gate)

Writes onto Node.props in place.
LLM: qwen3.5-nothink, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import (Turn, Node, DISTORTION_TYPES, TECHNIQUES, SITUATION_KINDS,
                                  SELF_CB_CATEGORIES, HOMEWORK_TASKTYPES, SUBCLASS_GLOSS,
                                  emotion_valence_from_text, temporality_from_text,
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


def _classify(nodes: list[Node], turns: list[Turn], llm: ChatOllama, task: str,
              gloss: dict[str, str], field: str, free_label: bool = False,
              desc: str | None = None) -> None:
    """Generic batched single-property classifier; writes Node.props[field]."""
    if not nodes:
        return
    gloss_block = "\n".join(f"  - {k}: {v}" for k, v in gloss.items())
    bar_desc = desc or f"stage2.5 {field}"
    for start in tqdm(range(0, len(nodes), BATCH), desc=bar_desc, unit="batch"):
        batch = nodes[start:start + BATCH]
        blocks = [f"ITEM {i}: '{n.text}'\nEVIDENCE:\n{_evidence(n, turns)}"
                  for i, n in enumerate(batch, 1)]
        extra = (' If value is "other", also return "techniqueLabel" (short free text).'
                 if free_label else "")
        prompt = (f"{task}\nItem texts/evidence are in Thai.\n\n"
                  f"ALLOWED VALUES (choose exactly one):\n{gloss_block}\n\n"
                  + "\n\n".join(blocks) +
                  f'\n\nReturn one object per item: [{{"item":1,"{field}":"<value>"}}, ...].'
                  f"{extra}{_JSON_REMINDER}")
        for it in _parse(llm.invoke("/no_think\n" + prompt).content):
            if not isinstance(it, dict):
                continue
            idx, val = it.get("item"), str(it.get(field, "")).strip()
            if not (isinstance(idx, int) and 1 <= idx <= len(batch)) or val not in gloss:
                continue
            batch[idx - 1].props[field] = val
            if free_label and val == "other":
                tl = str(it.get("techniqueLabel", "")).strip()
                if tl:
                    batch[idx - 1].props["techniqueLabel"] = tl


_BOOL_PROMPT = """For each homework task, decide if the therapist framed it as
OPTIONAL (explicit "if you want to", "you don't have to") rather than expected.
Texts/evidence are in Thai.

{candidates}

Return one object per item: [{{"item":1,"isOptional":true|false}}, ...].{reminder}"""


def _classify_isoptional(nodes: list[Node], turns: list[Turn], llm: ChatOllama) -> None:
    if not nodes:
        return
    for start in tqdm(range(0, len(nodes), BATCH),
                      desc="stage2.5 Homework.isOptional", unit="batch"):
        batch = nodes[start:start + BATCH]
        blocks = [f"ITEM {i}: '{n.text}'\nEVIDENCE:\n{_evidence(n, turns)}"
                  for i, n in enumerate(batch, 1)]
        prompt = _BOOL_PROMPT.format(candidates="\n\n".join(blocks), reminder=_JSON_REMINDER)
        for it in _parse(llm.invoke("/no_think\n" + prompt).content):
            if isinstance(it, dict) and isinstance(it.get("item"), int):
                idx = it["item"]
                if 1 <= idx <= len(batch) and isinstance(it.get("isOptional"), bool):
                    batch[idx - 1].props["isOptional"] = it["isOptional"]
        for n in batch:                              # default: not optional
            n.props.setdefault("isOptional", False)


def run_stage2_5(survivors: dict[str, list[Node]], turns: list[Turn],
                 llm: ChatOllama | None = None) -> None:
    llm = llm or _llm()

    # ── discriminators ── must run FIRST so category/valence can gate on them.
    _classify(survivors.get("Problem", []), turns, llm,
              "Pick the domain that best fits each problem.",
              SUBCLASS_GLOSS["Problem"], "domain",
              desc="stage2.5 Problem.domain")
    _classify(survivors.get("CoreBelief", []), turns, llm,
              "Pick the domain that best fits each core belief.",
              SUBCLASS_GLOSS["CoreBelief"], "domain",
              desc="stage2.5 CoreBelief.domain")
    _classify(survivors.get("IntermediateBelief", []), turns, llm,
              "Pick the subtype that best fits each intermediate belief.",
              SUBCLASS_GLOSS["IntermediateBelief"], "subtype",
              desc="stage2.5 IntermediateBelief.subtype")
    _classify(survivors.get("Reaction", []), turns, llm,
              "Pick the channel that best fits each reaction.",
              SUBCLASS_GLOSS["Reaction"], "channel",
              desc="stage2.5 Reaction.channel")

    ats = survivors.get("AutomaticThought", [])
    _classify(ats, turns, llm,
              "Label the cognitive-distortion pattern of each automatic thought. "
              "Use 'none' if it is accurate or no pattern fits — do not force one.",
              DISTORTION_TYPES, "distortionType",
              desc="stage2.5 AutomaticThought.distortionType")
    _classify(ats, turns, llm,
              "Label whether each automatic thought is a worded thought or a mental image.",
              {"verbal": "a worded thought", "image": "a mental picture"}, "modality",
              desc="stage2.5 AutomaticThought.modality")
    for n in ats:
        n.props.setdefault("modality", "verbal")     # default

    _classify(survivors.get("Situation", []), turns, llm,
              "Identify the trigger CHANNEL of each situation (no time meaning).",
              SITUATION_KINDS, "kind",
              desc="stage2.5 Situation.kind")

    _classify(survivors.get("Intervention", []), turns, llm,
              "Identify which therapeutic technique each intervention uses. "
              "Beck techniques outside the list (psychoeducation, coping card) are 'other'.",
              TECHNIQUES, "technique", free_label=True,
              desc="stage2.5 Intervention.technique")

    hw = survivors.get("Homework", [])
    _classify(hw, turns, llm, "Classify each homework task by type.",
              HOMEWORK_TASKTYPES, "taskType",
              desc="stage2.5 Homework.taskType")
    _classify_isoptional(hw, turns, llm)

    self_cb = [n for n in survivors.get("CoreBelief", []) if n.props.get("domain") == "self"]
    _classify(self_cb, turns, llm,
              "Categorize each self-directed core belief into Beck's three categories.",
              SELF_CB_CATEGORIES, "category",
              desc="stage2.5 CoreBelief.category")

    # deterministic passes
    for n in survivors.get("Reaction", []):
        if n.props.get("channel") == "emotional":
            v = emotion_valence_from_text(n.text)
            if v:
                n.props["valence"] = v
    by_idx = {t.turn_index: t for t in turns}
    for n in survivors.get("Situation", []):
        v = temporality_from_text(n.text)
        if v is None:
            for ti in sorted(n.evidence):
                t = by_idx.get(ti)
                if t and (v := temporality_from_text(t.text)):
                    break
        if v:
            n.props["temporality"] = v

    print("[stage2.5] properties assigned", file=sys.stderr)
