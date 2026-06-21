"""
CBT KG — Stage 2 (V4_flat): merge by class-only partition + candidate gate + LLM judge.

Per PIPELINE_UPDATE_v4_flat.md §Change 2: the prior pairwise-best HIGH/LOW two-band
logic almost never fires. New design:

  1. Partition by class label ONLY. The discriminator (domain/subtype/channel)
     is set in Stage 2.5, so it cannot be used as a partition key. Class-only is
     also CORRECT: a self-belief vs world-belief never crosses the cosine gate
     and never gets LLM-confirmed as "same", so wrong-domain merges are blocked
     by the gate, not by the partition.
  2. Within each partition: embed all texts, generate candidate pairs at cosine
     >= GATE, auto-confirm at >= AUTO, batched LLM judge on the rest.
  3. Build connected components over confirmed pairs (auto + llm-yes) and fold
     each component into its lowest-id member. Transitive merge is allowed ONLY
     over confirmed edges — never over raw embedding similarity.

Invariants preserved: lower id survives; survivor carries UNION of evidence,
context, and props; post-merge evidence-union assertion still hard.

Fail modes:
  - LLM pair-judge parse failure: FAIL CLOSED (none of that batch's pairs confirmed).
  - Embedding failure: bubbles up; pipeline does not silently skip merge.

Embedder: qwen3-embedding:8b.  Pair-judge LLM: qwen3.5-nothink, temperature 0.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass

from langchain_ollama import ChatOllama, OllamaEmbeddings
from tqdm import tqdm

from cbt_ontology_v4_flat import Node

OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call

GATE = 0.80           # candidate generation threshold
AUTO = 0.95           # skip LLM for near-identical pairs
JUDGE_BATCH = 8       # pair-judge batch size


@dataclass
class MergeResult:
    survivors: dict[str, list[Node]]      # {abstract_label: [surviving Node, ...]}
    retired_ids: set[int]
    audit: list[dict]                     # rows of actual folds (one per retire)
    pair_decisions: list[dict]            # every candidate pair above GATE, with verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _candidate_pairs(nodes: list[Node], vecs: list[list[float]],
                     gate: float) -> list[tuple[int, int, float]]:
    """All unordered (i, j) index pairs with cos >= gate, sorted by cos desc."""
    out: list[tuple[int, int, float]] = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            c = _cos(vecs[i], vecs[j])
            if c >= gate:
                out.append((i, j, c))
    out.sort(key=lambda t: -t[2])
    return out


_JUDGE_PROMPT = """For each pair of CBT entities of the same class from one session, decide if they are
the SAME underlying item (duplicate mentions) or DIFFERENT items. Texts are in Thai.
Same wording is not required; same meaning is what matters.

{pairs}

Return one object per pair: [{{"pair":1,"same":true|false}}, ...]."""

_JSON_REMINDER = "\n\nOutput ONLY a JSON array starting with [ and ending with ]."


def _judge_pairs(pairs: list[tuple[Node, Node]], llm: ChatOllama,
                 debug: bool = False) -> set[frozenset[int]]:
    """Batched LLM "same/different" judgment. FAIL CLOSED on parse error."""
    confirmed: set[frozenset[int]] = set()
    if not pairs:
        return confirmed

    iterator = range(0, len(pairs), JUDGE_BATCH)
    if not debug:
        iterator = tqdm(iterator, desc="stage2 pair-judge", unit="batch")

    for start in iterator:
        batch = pairs[start:start + JUDGE_BATCH]
        blocks = "\n\n".join(
            f"PAIR {i}\n  A: '{a.text}'\n  B: '{b.text}'"
            for i, (a, b) in enumerate(batch, 1)
        )
        prompt = _JUDGE_PROMPT.format(pairs=blocks) + _JSON_REMINDER
        raw = llm.invoke("/no_think\n" + prompt).content
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        except (ValueError, json.JSONDecodeError):
            print("[stage2] pair-judge parse fail — batch FAIL CLOSED (no merges)",
                  file=sys.stderr)
            continue
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("pair")
            same = bool(it.get("same"))
            if isinstance(idx, int) and 1 <= idx <= len(batch) and same:
                a, b = batch[idx - 1]
                confirmed.add(frozenset({a.id, b.id}))
    return confirmed


def _components(ids: list[int], confirmed: set[frozenset[int]]) -> list[set[int]]:
    """Union-find over confirmed pairs; return one component set per id."""
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # lower id is the canonical root
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    for pair in confirmed:
        a, b = tuple(pair)
        if a in parent and b in parent:
            union(a, b)

    comp_by_root: dict[int, set[int]] = {}
    for i in ids:
        r = find(i)
        comp_by_root.setdefault(r, set()).add(i)
    return list(comp_by_root.values())


def _fold(into: Node, other: Node) -> None:
    """Merge `other` into `into` (into has the lower id). Union everything."""
    into.evidence |= other.evidence
    into.context.update(other.context)
    for k, v in other.props.items():
        into.props.setdefault(k, v)


# ---------------------------------------------------------------------------
# Per-partition merge
# ---------------------------------------------------------------------------

def _merge_partition(nodes: list[Node], embedder: OllamaEmbeddings,
                     llm: ChatOllama, audit: list[dict],
                     pair_decisions: list[dict],
                     debug: bool = False, partition_name: str = "") -> tuple[list[Node], set[int]]:
    if len(nodes) < 2:
        return nodes, set()
    nodes = sorted(nodes, key=lambda n: n.id)
    vecs = embedder.embed_documents([n.text for n in nodes])

    cands = _candidate_pairs(nodes, vecs, GATE)

    if debug:
        tqdm.write(f"[merge:debug] partition={partition_name} size={len(nodes)} "
                   f"candidates={len(cands)}")

    # split into auto-confirmed and to-judge
    auto_confirmed: set[frozenset[int]] = set()
    to_judge_pairs: list[tuple[Node, Node]] = []
    cos_by_pair: dict[frozenset[int], float] = {}
    for i, j, c in cands:
        pair = frozenset({nodes[i].id, nodes[j].id})
        cos_by_pair[pair] = c
        if c >= AUTO:
            auto_confirmed.add(pair)
            if debug:
                tqdm.write(f"[merge:debug]   {nodes[i].id}-{nodes[j].id} cos={c:.3f} -> auto")
        else:
            to_judge_pairs.append((nodes[i], nodes[j]))
            if debug:
                tqdm.write(f"[merge:debug]   {nodes[i].id}-{nodes[j].id} cos={c:.3f} -> judge")

    llm_confirmed = _judge_pairs(to_judge_pairs, llm, debug=debug)

    # Record every candidate pair decision (auto/llm-yes/llm-no) for the report.
    by_id_local = {n.id: n for n in nodes}
    for pair in auto_confirmed:
        a_id, b_id = sorted(pair)
        pair_decisions.append({
            "partition": partition_name,
            "a_id": a_id, "b_id": b_id,
            "a_text": by_id_local[a_id].text,
            "b_text": by_id_local[b_id].text,
            "cos": round(cos_by_pair.get(pair, 0.0), 3),
            "verdict": "auto",
        })
    for (a, b) in to_judge_pairs:
        pair = frozenset({a.id, b.id})
        verdict = "llm-yes" if pair in llm_confirmed else "llm-no"
        if debug:
            tqdm.write(f"[merge:debug]   verdict {a.id}-{b.id}: {verdict}")
        pair_decisions.append({
            "partition": partition_name,
            "a_id": a.id, "b_id": b.id,
            "a_text": a.text, "b_text": b.text,
            "cos": round(cos_by_pair.get(pair, 0.0), 3),
            "verdict": verdict,
        })

    confirmed = auto_confirmed | llm_confirmed
    comps = _components([n.id for n in nodes], confirmed)

    by_id = {n.id: n for n in nodes}
    retired: set[int] = set()
    for comp in comps:
        if len(comp) < 2:
            continue
        survivor_id = min(comp)
        survivor = by_id[survivor_id]
        for other_id in sorted(comp - {survivor_id}):
            other = by_id[other_id]
            _fold(survivor, other)
            retired.add(other_id)
            pair = frozenset({survivor_id, other_id})
            audit.append({
                "survivor": survivor_id, "retired": other_id,
                "cos": round(cos_by_pair.get(pair, 0.0), 3),
                "via": "auto" if pair in auto_confirmed else "llm-yes",
                "survivor_text": survivor.text, "retired_text": other.text,
            })
        if debug:
            tqdm.write(f"[merge:debug] component {sorted(comp)} -> survivor={survivor_id}")

    survivors = [n for n in nodes if n.id not in retired]
    return survivors, retired


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stage2(by_label: dict[str, list[Node]],
               embedder: OllamaEmbeddings | None = None,
               llm: ChatOllama | None = None,
               debug: bool = False) -> MergeResult:
    embedder = embedder or OllamaEmbeddings(model="qwen3-embedding:8b")
    llm = llm or ChatOllama(model="qwen3.5-nothink", temperature=0,
                            request_timeout=OLLAMA_TIMEOUT)

    survivors: dict[str, list[Node]] = {}
    retired_all: set[int] = set()
    audit: list[dict] = []
    pair_decisions: list[dict] = []

    for label, nodes in tqdm(by_label.items(), desc="stage2 merge", unit="class"):
        # Partition by class label ONLY (flat: discriminator not yet set).
        if debug:
            tqdm.write(f"[merge:debug] === class {label} ({len(nodes)} nodes) ===")
        surv, retired = _merge_partition(nodes, embedder, llm, audit, pair_decisions,
                                         debug=debug, partition_name=label)
        retired_all |= retired
        survivors[label] = surv

    # Hard invariant: every survivor carries a non-empty evidence union.
    for label, nodes in survivors.items():
        for n in nodes:
            assert n.evidence, f"node {n.id} ({label}) lost its evidence union"

    print(f"[stage2] merged: retired {len(retired_all)} node(s), "
          f"{sum(len(v) for v in survivors.values())} survivors, "
          f"{len(audit)} merge actions, {len(pair_decisions)} candidate pairs",
          file=sys.stderr)
    return MergeResult(survivors=survivors, retired_ids=retired_all,
                       audit=audit, pair_decisions=pair_decisions)
