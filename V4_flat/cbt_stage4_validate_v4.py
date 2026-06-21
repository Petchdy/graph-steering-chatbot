"""
CBT KG — Stage 4 (V4): validate -> repair(4a-only) -> verify.

4a  deterministic (code). Edges failing a check are DROPPED:
    D0 dangling endpoint
    D1 signature (predicate, subj_label, obj_label) not in ALLOWED_SIGNATURES
    D2 disjointness (DISJOINT_RULES, e.g. any AT->AT)
    D3 evidence outside endpoints  -> soft-clamp; drop only if nothing remains

4a-repair  (kept, scoped — flagged: revisit). ONLY repairs edges 4a dropped for a
    fixable endpoint: predicate is valid, but the object is the wrong class. If
    exactly ONE node of an allowed object-class for that predicate has evidence
    overlapping the dropped edge, re-point to it (recovers recall without fan-out).
    Nothing else is repaired; no new edges are invented.

4b  LLM keep/drop verifier on every surviving LLM edge (decision: verify all).
    Verbalizes the relation definition + endpoint texts + evidence turns. Fails
    OPEN (a parse failure keeps the batch — never silently deletes edges).
    Deterministic structure edges (hasSession/hasProblem/...) skip 4b.

LLM: llama3.1:8b, temperature 0.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from tqdm import tqdm

from cbt_ontology_v4_flat import (Turn, Node, Edge, ALLOWED_SIGNATURES, DISJOINT_RULES,
                                  DETERMINISTIC_PREDICATES, PREDICATE_OBJECTS, ANCHOR_FAMILIES,
                                  REINFORCES, leaf_label, turn_texts, render_turns)

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call
VERIFY_BATCH = 6
_JSON_REMINDER = "\n\nOutput ONLY a JSON array starting with [ and ending with ]."

# predicate -> hint (for 4b), from the registry
_HINT: dict[str, str] = {}
for _subj, _fams in ANCHOR_FAMILIES.items():
    for (_p, _o, _h) in _fams:
        _HINT.setdefault(_p, _h)
_HINT["reinforces"] = "this reaction maintains/strengthens that core belief (feedback loop)"


@dataclass
class ValidationReport:
    dropped_4a: list[dict] = field(default_factory=list)
    repaired: list[dict] = field(default_factory=list)
    dropped_4b: list[dict] = field(default_factory=list)
    kept: int = 0

    def summary(self) -> str:
        return (f"kept={self.kept}  dropped_4a={len(self.dropped_4a)}  "
                f"repaired={len(self.repaired)}  dropped_4b={len(self.dropped_4b)}")


def _index(survivors: dict[str, list[Node]]) -> dict[int, Node]:
    idx = {n.id: n for ns in survivors.values() for n in ns}
    return idx


def _ed(e: Edge, s: Node | None, o: Node | None) -> dict:
    return {"predicate": e.predicate,
            "subject": s.text if s else f"id={e.subject_id}",
            "object": o.text if o else f"id={e.object_id}",
            "evidence": sorted(e.evidence)}


# ---------------------------------------------------------------------------
# 4a + repair
# ---------------------------------------------------------------------------

def _signature(e: Edge, by_id: dict[int, Node]) -> tuple[str, str, str] | None:
    s, o = by_id.get(e.subject_id), by_id.get(e.object_id)
    if s is None or o is None:
        return None
    return (e.predicate, s.label, o.label)


def _try_repair(e: Edge, by_id: dict[int, Node],
                nodes_by_label: dict[str, list[Node]]) -> Edge | None:
    """Re-point an endpoint-broken edge to the single correct-class node whose
    evidence overlaps. Returns a repaired Edge or None."""
    allowed_obj_labels = PREDICATE_OBJECTS.get(e.predicate)
    if not allowed_obj_labels:
        return None
    subj = by_id.get(e.subject_id)
    if subj is None:
        return None
    # candidate correct-class nodes whose evidence overlaps the dropped edge's
    cands: list[Node] = []
    ev = e.evidence or subj.evidence
    for lbl in allowed_obj_labels:
        for n in nodes_by_label.get(lbl, []):
            if n.id != subj.id and (n.evidence & ev):
                cands.append(n)
    if len(cands) != 1:
        return None
    target = cands[0]
    # subject class must be valid for this predicate too
    if (e.predicate, subj.label, target.label) not in ALLOWED_SIGNATURES:
        return None
    new = Edge(predicate=e.predicate, subject_id=subj.id, object_id=target.id,
               evidence=(e.evidence & (subj.evidence | target.evidence)) or set(),
               properties=dict(e.properties), reason=e.reason, repaired=True)
    return new


def validate_4a(edges: list[Edge], survivors: dict[str, list[Node]],
                report: ValidationReport) -> list[Edge]:
    by_id = _index(survivors)
    # map abstract label -> nodes (object candidates are matched on abstract label)
    nodes_by_label: dict[str, list[Node]] = {}
    for ns in survivors.values():
        for n in ns:
            nodes_by_label.setdefault(n.label, []).append(n)

    out: list[Edge] = []
    for e in edges:
        if e.predicate in DETERMINISTIC_PREDICATES:
            out.append(e)                             # structural — trusted
            continue
        s, o = by_id.get(e.subject_id), by_id.get(e.object_id)
        if s is None or o is None:
            report.dropped_4a.append({"edge": _ed(e, s, o), "rule": "D0 dangling"})
            continue
        # D2 disjointness
        if any(s.label == ds and o.label == do and dp in ("*", e.predicate)
               for (ds, dp, do) in DISJOINT_RULES):
            report.dropped_4a.append({"edge": _ed(e, s, o), "rule": "D2 disjoint"})
            continue
        # D1 signature
        if (e.predicate, s.label, o.label) not in ALLOWED_SIGNATURES:
            repaired = _try_repair(e, by_id, nodes_by_label)
            if repaired is not None:
                rs, ro = by_id[repaired.subject_id], by_id[repaired.object_id]
                report.repaired.append({"from": _ed(e, s, o), "to": _ed(repaired, rs, ro)})
                out.append(repaired)
            else:
                report.dropped_4a.append(
                    {"edge": _ed(e, s, o),
                     "rule": f"D1 signature ({e.predicate},{s.label},{o.label})"})
            continue
        # D3 evidence clamp
        allowed = s.evidence | o.evidence
        if e.evidence and not e.evidence <= allowed:
            e.evidence &= allowed
            if not e.evidence:
                report.dropped_4a.append({"edge": _ed(e, s, o), "rule": "D3 evidence"})
                continue
        # property hygiene: reportedIntensity only on leadsTo
        if e.properties and (e.predicate != "leadsTo"
                             or set(e.properties) - {"reportedIntensity"}):
            e.properties = ({"reportedIntensity": e.properties["reportedIntensity"]}
                            if e.predicate == "leadsTo" and "reportedIntensity" in e.properties
                            else {})
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 4b verifier
# ---------------------------------------------------------------------------

_VERIFY_PROMPT = """You verify candidate CBT relationship edges from a therapy
transcript (Thai). For each candidate, judge whether the SPECIFIC relationship
between the SPECIFIC subject and object is supported by the evidence turns, given
the definition.

DEFAULT: "keep" when the evidence plausibly supports THIS pair; mere co-occurrence
or thematic similarity without a directional link is not enough. When in doubt,
"drop".

EXCEPTION for chain-essential predicates (triggers, leadsTo, stemsFrom,
givesRiseTo): these form the canonical Situation→AutomaticThought→Reaction chain
plus the laddering links. If the subject and object are in the same exchange (or
the subject's evidence turns directly precede the object's), and the topic is
consistent, KEEP — Beck's basic cognitive model expects these edges. Only drop
when the evidence actively contradicts the pairing.

{candidates}

Return one object per candidate:
[{{"candidate":1,"verdict":"keep","reason":"<short>"}}, ...]
verdict is exactly "keep" or "drop".{reminder}"""


def _render_candidate(i: int, e: Edge, by_id: dict[int, Node],
                      turns_by_idx: dict[int, Turn]) -> str:
    s, o = by_id[e.subject_id], by_id[e.object_id]
    ev = []
    for ti in sorted(e.evidence)[:5]:
        t = turns_by_idx.get(ti)
        if t:
            ev.append(f"  turn {ti} | {'T' if t.speaker=='therapist' else 'C'}: {t.text}")
    return (f"CANDIDATE {i}: ({leaf_label(s.label,s.group_key)}) '{s.text}' "
            f"--{e.predicate}--> ({leaf_label(o.label,o.group_key)}) '{o.text}'\n"
            f"DEFINITION: {e.predicate} = {_HINT.get(e.predicate,'(relation)')}\n"
            f"EVIDENCE:\n" + ("\n".join(ev) or "  (none)"))


def verify_4b(edges: list[Edge], survivors: dict[str, list[Node]], turns: list[Turn],
              report: ValidationReport, llm: ChatOllama | None = None) -> list[Edge]:
    llm = llm or ChatOllama(model="qwen3.5-nothink", temperature=0, request_timeout=OLLAMA_TIMEOUT)
    by_id = _index(survivors)
    turns_by_idx = {t.turn_index: t for t in turns}

    structural = [e for e in edges if e.predicate in DETERMINISTIC_PREDICATES]
    to_check = [e for e in edges if e.predicate not in DETERMINISTIC_PREDICATES]
    kept: list[Edge] = list(structural)              # structure passes through

    for start in tqdm(range(0, len(to_check), VERIFY_BATCH), desc="stage4b verify"):
        batch = to_check[start:start + VERIFY_BATCH]
        blocks = [_render_candidate(i + 1, e, by_id, turns_by_idx)
                  for i, e in enumerate(batch)]
        prompt = _VERIFY_PROMPT.format(candidates="\n\n".join(blocks), reminder=_JSON_REMINDER)
        raw = re.sub(r"<think>.*?</think>", "", llm.invoke("/no_think\n" + prompt).content, flags=re.DOTALL).strip()
        verdicts = [(True, "parse-fail kept")] * len(batch)   # fail OPEN
        try:
            arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            for it in arr:
                if isinstance(it, dict) and isinstance(it.get("candidate"), int):
                    idx = it["candidate"]
                    if 1 <= idx <= len(batch):
                        verdicts[idx - 1] = (
                            str(it.get("verdict", "")).strip().lower() == "keep",
                            str(it.get("reason", "")).strip())
        except (ValueError, json.JSONDecodeError):
            print("[stage4b] verifier parse fail — keeping batch", file=sys.stderr)
        for e, (keep, reason) in zip(batch, verdicts):
            if keep:
                kept.append(e)
            else:
                s, o = by_id[e.subject_id], by_id[e.object_id]
                report.dropped_4b.append({"edge": _ed(e, s, o), "reason": reason or "(none)"})
    return kept


def _dedup(edges: list[Edge]) -> list[Edge]:
    """Collapse duplicate (predicate, subject, object) edges (repair can recreate an
    edge that already exists). Merge evidence + properties; keep a non-repaired
    reason if any."""
    by_key: dict[tuple[str, int, int], Edge] = {}
    for e in edges:
        k = (e.predicate, e.subject_id, e.object_id)
        if k not in by_key:
            by_key[k] = e
        else:
            keep = by_key[k]
            keep.evidence |= e.evidence
            for pk, pv in e.properties.items():
                keep.properties.setdefault(pk, pv)
            if keep.repaired and not e.repaired:      # prefer the genuine edge
                keep.repaired = False
                keep.reason = e.reason or keep.reason
    return list(by_key.values())


def run_stage4(edges: list[Edge], survivors: dict[str, list[Node]], turns: list[Turn],
               llm: ChatOllama | None = None) -> tuple[list[Edge], ValidationReport]:
    report = ValidationReport()
    survived = _dedup(validate_4a(edges, survivors, report))
    final = verify_4b(survived, survivors, turns, report, llm=llm)
    report.kept = len(final)
    print(f"[stage4] {report.summary()}", file=sys.stderr)
    return final, report
