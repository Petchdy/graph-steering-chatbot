"""Extraction pipeline for Part 1 (V4_flat per-turn + consolidation).

StubExtractor   — offline deterministic, used by tests. Parses "Label: text"
                  lines into nodes.
TurnPipeline    — the real two-tier pipeline that mirrors V4_flat's batch
                  stages but runs incrementally.

Tier A (every client turn — runs in background asyncio Task):
  1. EXTRACT     — V4_flat Stage 1 prompt (per-turn, ±2 context).
  2. ATOMIZE     — V4_flat Stage 1.2 prompt for AT / CoreBelief / IB only.
  3. PROPERTIES  — V4_flat Stage 2.5 classifiers (discriminators FIRST).
  4. MERGE       — string-Jaccard against existing graph nodes.
  5. EDGES       — V4_flat Stage 3 Pass A anchor prompt (ANCHOR_FAMILIES), run against this
                   turn's new nodes as subject. Covers every ANCHOR_FAMILIES relation, e.g.
                   triggers/leadsTo/stemsFrom/manifestsAs/hasAdaptiveResponse/becomesSituation
                   — whatever fires depends on which compatible object nodes already exist.
  6. RETRY       — a node only ever gets to be a relation SUBJECT on the turn it's created;
                   an old node whose only relation needed an object that didn't exist yet at
                   its own turn is otherwise stuck forever. Targeted, cheap fix: for each new
                   node this turn, BFS outward through existing edges (bounded hops) to find
                   old "orphan" nodes (no outgoing relation yet) in the same structural
                   neighborhood whose class could plausibly point at the new node, and retry
                   just those — not a full-graph scan, same cheap window context as step 5.

Tier B (every CONSOLIDATE_EVERY turns; reset/session-end):
  1. SESSION-LEVEL — V4_flat Stage 1.1 over whole transcript (Problem, Goal, Intervention,
                     Homework, CoreBelief, IntermediateBelief, AdaptiveResponse).
  2. EDGES (wide)  — same Pass A anchor prompt as Tier A, but run against the nodes just
                     added by the session-level pass, over the FULL transcript as context.
                     This is what actually resolves appliedTo/produces/targets/targetsProblem
                     — their subjects (Intervention/Homework/Goal) are only ever created here.
  3. REINFORCES    — V4_flat Stage 3 Pass B wide-window Reaction × CoreBelief. The one relation
                     that is genuinely never per-turn-safe (needs the full Reaction×CoreBelief
                     cross-product; absence is clinically meaningful, so never guessed locally).
  4. REFRAME       — heuristic best-effort hasAdaptiveResponse fallback (Jaccard word-overlap)
                     for cases the anchor prompt in step 2 misses.
  5. STRUCTURE     — deterministic Client/Session/Goal-targets-Problem/etc.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

from .interfaces import GraphStore
from .ontology import (ANCHOR_FAMILIES, CLASS_DEFINITIONS, CORE_BELIEF_DOMAINS,
                       DISTORTION_TYPES, EXTRACT_CLASSES, HOMEWORK_TASKTYPES,
                       IB_SUBTYPES, OBJECT_EDGES, PROBLEM_DOMAINS, REACTION_CHANNELS,
                       SELF_CB_CATEGORIES, SITUATION_KINDS, SPEAKER_PRIOR,
                       SUBCLASS_GLOSS, TECHNIQUES, TEXT_PROP,
                       emotion_valence_from_text, temporality_from_text)
from .prompts import (ATOMIZE_PROMPT, AT_SPECIFIC_CLAUSE, EDGE_ANCHOR_PROMPT,
                      EXTRACT_PROMPT, PROPERTY_CLASSIFY_PROMPT,
                      PROPERTY_ISOPTIONAL_PROMPT, PROPERTY_TASKS,
                      REINFORCES_PROMPT, SELF_BELIEF_CLAUSE,
                      SESSION_LEVEL_PROMPT)

# Per cbt_kg_extraction_descriptions_V4_flat.md, every ANCHOR_FAMILIES relation is Pass A
# (local, subject-anchored) — V4_flat never distinguished a "per-turn-safe" subset. The one
# genuinely wide-window-only relation is `reinforces` (Reaction reinforces CoreBelief, resolved
# separately by _reinforces_pass), which is NOT part of ANCHOR_FAMILIES. So the safe set is
# simply everything ANCHOR_FAMILIES defines.
PER_TURN_SAFE_PREDICATES = frozenset(
    pred for fams in ANCHOR_FAMILIES.values() for (pred, _obj, _hint) in fams
)

ATOMIZE_CLASSES = frozenset({"AutomaticThought", "CoreBelief", "IntermediateBelief"})

# Per-class enum tables for the property pass.
_PROPERTY_BATCH: list[tuple[str, str, dict]] = [
    # (label, field, gloss_dict)  — discriminators first
    ("Problem", "domain", {k: SUBCLASS_GLOSS["Problem"][k] for k in PROBLEM_DOMAINS}),
    ("CoreBelief", "domain", {k: SUBCLASS_GLOSS["CoreBelief"][k] for k in CORE_BELIEF_DOMAINS}),
    ("IntermediateBelief", "subtype", {k: SUBCLASS_GLOSS["IntermediateBelief"][k] for k in IB_SUBTYPES}),
    ("Reaction", "channel", {k: SUBCLASS_GLOSS["Reaction"][k] for k in REACTION_CHANNELS}),
    # Then the others (gated by discriminators where applicable).
    ("AutomaticThought", "distortionType", DISTORTION_TYPES),
    ("AutomaticThought", "modality",
        {"verbal": "a worded thought", "image": "a mental picture"}),
    ("Situation", "kind", SITUATION_KINDS),
    ("Intervention", "technique", TECHNIQUES),
    ("Homework", "taskType", HOMEWORK_TASKTYPES),
]


def _text_of(label: str, props: dict) -> str:
    key = TEXT_PROP.get(label)
    if key:
        v = props.get(key)
        if isinstance(v, str) and v:
            return v
    for fb in ("description", "content", "statement", "taskDescription", "text"):
        v = props.get(fb)
        if isinstance(v, str) and v:
            return v
    return ""


# ─────────────────────────────────────────────────────────────────────────
# StubExtractor — offline / tests
# ─────────────────────────────────────────────────────────────────────────

_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z]+)\s*:\s*(.+?)\s*$")


class StubExtractor:
    """Deterministic, offline extractor for tests.

    Parses lines of the form `<NodeLabel>: <text>` (e.g. `Situation: exam tomorrow`)
    where the label must be a V4_flat content class. Edges are not asserted —
    only nodes are emitted. consolidate() is a no-op."""

    def process_turn(self, client_msg: str, window: list[tuple[str, str]],
                     graph: GraphStore, turn_index: int) -> dict:
        new_nodes = []
        for line in client_msg.splitlines():
            m = _KV_RE.match(line)
            if not m:
                continue
            label, text = m.group(1), m.group(2)
            if label not in EXTRACT_CLASSES:
                continue
            text_key = TEXT_PROP.get(label, "content")
            props = {text_key: text}
            node = graph.upsert_node(label, props, turn_index)
            new_nodes.append(node)
        return {"new_nodes": [n.node_id for n in new_nodes], "edges": []}

    def consolidate(self, transcript: list[tuple[int, str, str]],
                    graph: GraphStore) -> dict:
        return {"added_nodes": [], "added_edges": []}


# ─────────────────────────────────────────────────────────────────────────
# Ollama JSON helper (used by TurnPipeline)
# ─────────────────────────────────────────────────────────────────────────

def _strip_think(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _parse_json_array(raw: str) -> list | None:
    raw = _strip_think(raw)
    try:
        return json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _ollama_generate(host: str, model: str, prompt: str,
                     timeout: int = 120) -> str:
    import requests
    body = {
        "model": model, "prompt": prompt,
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0},
    }
    resp = requests.post(f"{host}/api/generate", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


# ─────────────────────────────────────────────────────────────────────────
# TurnPipeline — Tier A + Tier B
# ─────────────────────────────────────────────────────────────────────────

class TurnPipeline:
    """V4_flat per-turn (+ consolidation) pipeline driven by Ollama."""

    def __init__(self, model: str = "qwen3.5-nothink",
                 host: str = "http://localhost:11434",
                 language: str = "English"):
        self._model = model
        self._host = host.rstrip("/")
        self._language = language
        self._fast = os.environ.get("EXTRACT_FAST", "0") == "1"

    # ── Tier A ─────────────────────────────────────────────────────────

    def process_turn(self, client_msg: str, window: list[tuple[str, str]],
                     graph: GraphStore, turn_index: int) -> dict:
        try:
            candidates = self._extract(client_msg, window, turn_index)
        except Exception as exc:
            print(f"[extract] stage1 failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return {"new_nodes": [], "edges": [], "error": str(exc)}

        if not candidates:
            return {"new_nodes": [], "edges": []}

        # 2. Atomize AT/CB/IB candidates.
        if not self._fast:
            candidates = self._atomize(candidates)

        # 3. Properties (discriminators first).
        if not self._fast:
            self._classify_properties(candidates)
        self._deterministic_properties(candidates)

        # 4. Merge into graph (graph.upsert_node uses jaccard internally).
        new_nodes = []
        for cand in candidates:
            label = cand["label"]
            props = cand.get("props", {})
            text_key = TEXT_PROP.get(label, "content")
            props.setdefault(text_key, cand.get("text", ""))
            if not props.get(text_key):
                continue
            node = graph.upsert_node(label, props, turn_index)
            cand["_node_id"] = node.node_id
            new_nodes.append(node)

        # 5. Local edges (per-turn-safe predicates only).
        edges_added: list[tuple[str, str, str]] = []
        try:
            edges_added = self._resolve_local_edges(new_nodes, graph, window, turn_index)
        except Exception as exc:
            print(f"[extract] stage3 failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

        # 6. Targeted retry for old ("orphan") nodes a new node this turn might now
        # connect to. Traces the graph outward from the new node (bounded BFS) instead
        # of scanning the whole graph; falls back to a flat orphan scan only when the
        # new node has no edges yet to trace from. Same cheap window context as step 5.
        try:
            handled_ids = {n.node_id for n in new_nodes}
            unblocked = self._unblocked_subjects(graph, new_nodes, handled_ids)
            if unblocked:
                edges_added += self._resolve_local_edges(unblocked, graph, window, turn_index)
        except Exception as exc:
            print(f"[extract] targeted rescan failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

        return {
            "new_nodes": [n.node_id for n in new_nodes],
            "edges": edges_added,
        }

    # ── Tier B ─────────────────────────────────────────────────────────

    def consolidate(self, transcript: list[tuple[int, str, str]],
                    graph: GraphStore) -> dict:
        """Run wide-window passes. transcript: [(turn_index, speaker, text), ...]."""
        if not transcript:
            return {"added_nodes": [], "added_edges": []}
        added_nodes = []
        try:
            added_nodes = self._session_level_extract(transcript, graph)
        except Exception as exc:
            print(f"[consolidate] stage1.1 failed: {exc}", file=sys.stderr)

        added_edges = []
        # Local (ANCHOR_FAMILIES) edges for the nodes the session-level pass just added.
        # Intervention/Homework/Goal/etc. are only ever created here (never in Tier A), so
        # appliedTo/produces/targets/targetsProblem/manifestsAs can only fire from this call.
        if added_nodes:
            try:
                new_ids = set(added_nodes)
                new_node_objs = [n for n in graph.nodes() if n.node_id in new_ids]
                wide_window = [(sp, tx) for _, sp, tx in transcript]
                last_turn = transcript[-1][0]
                added_edges += self._resolve_local_edges(
                    new_node_objs, graph, wide_window, last_turn)
            except Exception as exc:
                print(f"[consolidate] session-level edges failed: {exc}", file=sys.stderr)

        try:
            added_edges += self._reinforces_pass(graph, transcript)
        except Exception as exc:
            print(f"[consolidate] reinforces failed: {exc}", file=sys.stderr)

        # Reframe sub-graph (best-effort).
        try:
            added_edges += self._reframe_subgraph(graph, transcript)
        except Exception as exc:
            print(f"[consolidate] reframe failed: {exc}", file=sys.stderr)

        # Deterministic structure.
        added_edges += self._structure_edges(graph)

        return {"added_nodes": added_nodes, "added_edges": added_edges}

    # ── Step 1: EXTRACT ────────────────────────────────────────────────

    def _extract(self, client_msg: str, window: list[tuple[str, str]],
                 turn_index: int) -> list[dict]:
        defs_block = "\n".join(f"- {c}: {CLASS_DEFINITIONS[c]}" for c in EXTRACT_CLASSES)
        prior = ", ".join(SPEAKER_PRIOR["client"])
        context = "\n".join(
            f"{'T' if role == 'therapist' else 'C'}: {text}" for role, text in window
        ) or "(none)"

        prompt = EXTRACT_PROMPT.format(
            language=self._language, defs=defs_block, speaker="client",
            prior=prior, context=context, idx=turn_index, target=client_msg,
        )
        raw = _ollama_generate(self._host, self._model, prompt)
        arr = _parse_json_array(raw)
        if arr is None:
            print(f"[extract] stage1 parse fail: {raw[:120]!r}", file=sys.stderr)
            return []

        out: list[dict] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            label = it.get("label")
            text = (it.get("text") or "").strip()
            if label not in EXTRACT_CLASSES or not text:
                continue
            gk = it.get("group_key")
            gk = gk.strip() if isinstance(gk, str) and gk.strip().lower() != "null" else None
            cand = {"label": label, "text": text, "props": {}}
            if gk:
                cand["group_key"] = gk
            out.append(cand)
        return out

    # ── Step 2: ATOMIZE ────────────────────────────────────────────────

    def _atomize(self, candidates: list[dict]) -> list[dict]:
        out: list[dict] = []
        for cand in candidates:
            if cand["label"] not in ATOMIZE_CLASSES:
                out.append(cand)
                continue
            split = self._atomize_one(cand)
            out.extend(split)
        return out

    def _atomize_one(self, cand: dict) -> list[dict]:
        label = cand["label"]
        prompt = ATOMIZE_PROMPT.format(
            class_label=label,
            class_definition=CLASS_DEFINITIONS.get(label, ""),
            unit={"AutomaticThought": "thought", "CoreBelief": "belief",
                  "IntermediateBelief": "belief"}.get(label, "item"),
            language=self._language,
            self_belief_clause=SELF_BELIEF_CLAUSE if label in ("CoreBelief", "AutomaticThought") else "",
            at_specific=AT_SPECIFIC_CLAUSE if label == "AutomaticThought" else "",
            node_text=cand["text"],
            max_splits=4,
        )
        raw = _ollama_generate(self._host, self._model, prompt)
        arr = _parse_json_array(raw)
        if not arr:
            return [cand]
        cleaned = [t.strip() for t in arr if isinstance(t, str) and t.strip()]
        cleaned = cleaned[:4]
        if not cleaned:
            return [cand]
        if len(cleaned) == 1:
            cand["text"] = cleaned[0]
            return [cand]
        return [{"label": label, "text": t, "props": {}, "group_key": cand.get("group_key")}
                for t in cleaned]

    # ── Step 3a: LLM PROPERTIES ────────────────────────────────────────

    def _classify_properties(self, candidates: list[dict]) -> None:
        for label, field, gloss in _PROPERTY_BATCH:
            items = [c for c in candidates if c["label"] == label]
            if not items:
                continue
            # Gate self-belief category on domain=self (done later, not here).
            self._classify_one(items, label, field, gloss)

        # CoreBelief.category (only when domain=self).
        self_cb = [c for c in candidates
                   if c["label"] == "CoreBelief" and c["props"].get("domain") == "self"]
        if self_cb:
            self._classify_one(self_cb, "CoreBelief", "category", SELF_CB_CATEGORIES)

        # Homework.isOptional.
        hw = [c for c in candidates if c["label"] == "Homework"]
        if hw:
            self._classify_isoptional(hw)

    def _classify_one(self, items: list[dict], label: str, field: str,
                      gloss: dict[str, str]) -> None:
        gloss_block = "\n".join(f"  - {k}: {v}" for k, v in gloss.items())
        blocks = [f"ITEM {i}: '{c['text']}'" for i, c in enumerate(items, 1)]
        task = self._task_for(label, field)
        extra = (' If value is "other", also return "techniqueLabel" (short free text).'
                 if (label == "Intervention" and field == "technique") else "")
        prompt = PROPERTY_CLASSIFY_PROMPT.format(
            task=task, language=self._language, gloss_block=gloss_block,
            candidates="\n\n".join(blocks), field=field, extra=extra,
        )
        try:
            raw = _ollama_generate(self._host, self._model, prompt, timeout=90)
        except Exception as exc:
            print(f"[extract] property {label}.{field} failed: {exc}",
                  file=sys.stderr)
            return
        arr = _parse_json_array(raw)
        if arr is None:
            return
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("item")
            val = str(it.get(field, "")).strip()
            if not (isinstance(idx, int) and 1 <= idx <= len(items)):
                continue
            if val not in gloss:
                continue
            items[idx - 1]["props"][field] = val
            if label == "Intervention" and field == "technique" and val == "other":
                tl = str(it.get("techniqueLabel", "")).strip()
                if tl:
                    items[idx - 1]["props"]["techniqueLabel"] = tl

    def _task_for(self, label: str, field: str) -> str:
        if label == "Problem" and field == "domain":
            return PROPERTY_TASKS["domain_problem"]
        if label == "CoreBelief" and field == "domain":
            return PROPERTY_TASKS["domain_corebelief"]
        if label == "IntermediateBelief" and field == "subtype":
            return PROPERTY_TASKS["subtype_ib"]
        if label == "Reaction" and field == "channel":
            return PROPERTY_TASKS["channel_reaction"]
        return PROPERTY_TASKS.get(field, f"Classify {label}.{field}")

    def _classify_isoptional(self, items: list[dict]) -> None:
        blocks = [f"ITEM {i}: '{c['text']}'" for i, c in enumerate(items, 1)]
        prompt = PROPERTY_ISOPTIONAL_PROMPT.format(
            language=self._language, candidates="\n\n".join(blocks),
        )
        try:
            raw = _ollama_generate(self._host, self._model, prompt, timeout=90)
        except Exception as exc:
            print(f"[extract] isOptional failed: {exc}", file=sys.stderr)
            for c in items:
                c["props"].setdefault("isOptional", False)
            return
        arr = _parse_json_array(raw)
        for c in items:
            c["props"].setdefault("isOptional", False)
        if not arr:
            return
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("item")
            v = it.get("isOptional")
            if isinstance(idx, int) and 1 <= idx <= len(items) and isinstance(v, bool):
                items[idx - 1]["props"]["isOptional"] = v

    # ── Step 3b: DETERMINISTIC PROPERTIES (lexicon) ────────────────────

    def _deterministic_properties(self, candidates: list[dict]) -> None:
        for c in candidates:
            if c["label"] == "Reaction" and c["props"].get("channel") == "emotional":
                v = emotion_valence_from_text(c["text"])
                if v:
                    c["props"]["valence"] = v
            if c["label"] == "Situation":
                v = temporality_from_text(c["text"])
                if v:
                    c["props"]["temporality"] = v
                c["props"].setdefault("kind", "externalSituation")
            if c["label"] == "AutomaticThought":
                c["props"].setdefault("modality", "verbal")

    # ── Step 6: TARGETED RETRY (orphans within a bounded graph-path neighborhood) ──

    def _is_orphan(self, graph: GraphStore, node) -> bool:
        """True if node's class could have an outgoing safe-predicate edge, but doesn't yet."""
        if node.label not in ANCHOR_FAMILIES:
            return False
        return not any(
            e.status == "found" and e.subject_id == node.node_id
            and e.predicate in PER_TURN_SAFE_PREDICATES
            for e in graph.edges()
        )

    def _bfs_reachable(self, graph: GraphStore, start_id: str, max_hops: int) -> set:
        """Node ids reachable from start_id within max_hops, walking found edges in
        either direction. Cheap in-memory BFS — these graphs run to tens of nodes."""
        found_edges = [e for e in graph.edges() if e.status == "found"]
        visited = {start_id}
        frontier = {start_id}
        for _ in range(max_hops):
            next_frontier = set()
            for e in found_edges:
                if e.subject_id in frontier and e.object_id not in visited:
                    next_frontier.add(e.object_id)
                if e.object_id in frontier and e.subject_id not in visited:
                    next_frontier.add(e.subject_id)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        visited.discard(start_id)
        return visited

    def _unblocked_subjects(self, graph: GraphStore, new_nodes, exclude_ids: set,
                            max_hops: int = 2) -> list:
        """Old nodes that could now form a new relation to one of this turn's new nodes.

        Traces the graph outward (BFS, up to max_hops) from each new node to find a
        structurally-relevant neighborhood, then filters to orphans within it whose
        class could plausibly point at the new node (per ontology.OBJECT_EDGES). Falls
        back to a flat whole-graph orphan scan when the new node has no edges yet —
        early in a session there's no structure to trace, so the conservative fallback
        is correct rather than silently finding nothing.
        """
        by_id = {n.node_id: n for n in graph.nodes() if n.status == "found"}
        candidates: list = []
        seen_ids: set = set()
        for new_node in new_nodes:
            subj_labels = {s for (_pred, s) in OBJECT_EDGES.get(new_node.label, [])}
            if not subj_labels:
                continue
            reachable = self._bfs_reachable(graph, new_node.node_id, max_hops)
            pool = [by_id[nid] for nid in reachable if nid in by_id] or list(by_id.values())
            for n in pool:
                if (n.node_id in exclude_ids or n.node_id in seen_ids
                        or n.label not in subj_labels):
                    continue
                if self._is_orphan(graph, n):
                    candidates.append(n)
                    seen_ids.add(n.node_id)
        return candidates

    # ── Step 5: LOCAL EDGES ────────────────────────────────────────────

    def _resolve_local_edges(self, new_nodes, graph: GraphStore,
                              window: list[tuple[str, str]],
                              turn_index: int) -> list[tuple[str, str, str]]:
        all_nodes = [n for n in graph.nodes() if n.status == "found"]
        added: list[tuple[str, str, str]] = []
        ctx = "\n".join(
            f"{'T' if role == 'therapist' else 'C'}: {text}"
            for role, text in window
        ) or "(none)"

        for subj in new_nodes:
            families = ANCHOR_FAMILIES.get(subj.label, [])
            # Filter to per-turn-safe predicates only.
            families = [f for f in families if f[0] in PER_TURN_SAFE_PREDICATES]
            if not families:
                continue

            # Build candidate object lists.
            rel_objs: dict[str, list] = {}
            hint_by_pred: dict[str, str] = {}
            for (pred, obj_label, hint) in families:
                hint_by_pred.setdefault(pred, hint)
                objs = [n for n in all_nodes
                        if n.label == obj_label and n.node_id != subj.node_id]
                if not objs:
                    continue
                seen = {o.node_id for o in rel_objs.get(pred, [])}
                rel_objs.setdefault(pred, [])
                for o in objs:
                    if o.node_id not in seen:
                        rel_objs[pred].append(o)
                        seen.add(o.node_id)
            if not rel_objs:
                continue

            blocks = []
            for pred, objs in rel_objs.items():
                lines = "\n".join(
                    f"  {i}. '{_text_of(o.label, o.props)}'"
                    for i, o in enumerate(objs, 1)
                )
                blocks.append(f"RELATION {pred} — {hint_by_pred[pred]}\n{lines}")
            intensity = (', "reportedIntensity": "<text or omit>"'
                         if "leadsTo" in rel_objs else "")
            prompt = EDGE_ANCHOR_PROMPT.format(
                language=self._language,
                subj_label=subj.label,
                subj_text=_text_of(subj.label, subj.props),
                context=ctx,
                families="\n\n".join(blocks),
                intensity=intensity,
            )
            try:
                raw = _ollama_generate(self._host, self._model, prompt, timeout=120)
            except Exception as exc:
                print(f"[extract] edge anchor failed: {exc}", file=sys.stderr)
                continue
            arr = _parse_json_array(raw)
            if not arr:
                continue
            for it in arr:
                if not isinstance(it, dict):
                    continue
                pred = str(it.get("relation", "")).strip()
                objs = rel_objs.get(pred)
                num = it.get("object")
                if not objs or not isinstance(num, int) or not (1 <= num <= len(objs)):
                    continue
                obj = objs[num - 1]
                props = {}
                if pred == "leadsTo":
                    ri = it.get("reportedIntensity")
                    if isinstance(ri, str) and ri.strip().lower() != "omit":
                        props["reportedIntensity"] = ri.strip()
                graph.add_edge(subj.node_id, pred, obj.node_id,
                               props=props, evidence=[turn_index])
                added.append((subj.node_id, pred, obj.node_id))
        return added

    # ── Tier B: session-level extract ───────────────────────────────────

    def _session_level_extract(self, transcript: list[tuple[int, str, str]],
                               graph: GraphStore) -> list[str]:
        transcript_rendered = "\n".join(
            f"turn {ti} | {'T' if sp == 'therapist' else 'C'}: {tx}"
            for ti, sp, tx in transcript
        )
        valid_turn_idx = {ti for ti, _, _ in transcript}
        targets = ("CoreBelief", "IntermediateBelief", "Problem", "Goal",
                   "Intervention", "Homework", "AdaptiveResponse")
        added = []
        for cls in targets:
            try:
                new = self._session_class_pass(cls, transcript_rendered,
                                                valid_turn_idx, graph)
            except Exception as exc:
                print(f"[consolidate] {cls} failed: {exc}", file=sys.stderr)
                new = []
            added.extend(new)
        return added

    def _session_class_pass(self, class_label: str, transcript_rendered: str,
                            valid_turn_idx: set, graph: GraphStore) -> list[str]:
        # Build priors block (same-class only — keeps prompt small).
        priors_block = self._priors_block(graph, class_label)
        adj_block = "(none)"
        prompt = SESSION_LEVEL_PROMPT.format(
            language=self._language, class_label=class_label,
            class_definition=CLASS_DEFINITIONS.get(class_label, ""),
            same_class_priors=priors_block,
            adjacent_class_priors=adj_block,
            transcript=transcript_rendered,
        )
        raw = _ollama_generate(self._host, self._model, prompt, timeout=180)
        arr = _parse_json_array(raw)
        if not arr:
            return []
        added: list[str] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            evidence = [int(t) for t in (it.get("evidence_turns") or [])
                        if isinstance(t, (int, float)) and int(t) in valid_turn_idx]
            if not evidence:
                continue
            text_key = TEXT_PROP.get(class_label, "content")
            node = graph.upsert_node(class_label, {text_key: text}, evidence[-1])
            for ev in evidence:
                if ev not in node.evidence:
                    node.evidence.append(ev)
            added.append(node.node_id)
        return added

    def _priors_block(self, graph: GraphStore, label: str) -> str:
        items = [n for n in graph.nodes()
                 if n.label == label and n.status == "found"]
        if not items:
            return "(none)"
        return "\n".join(
            f"  - '{_text_of(n.label, n.props)}' (turns {n.evidence})"
            for n in items
        )

    # ── Tier B: reinforces pass ─────────────────────────────────────────

    def _reinforces_pass(self, graph: GraphStore,
                         transcript: list[tuple[int, str, str]]) -> list[tuple[str, str, str]]:
        reactions = [n for n in graph.nodes()
                     if n.label == "Reaction" and n.status == "found"]
        beliefs = [n for n in graph.nodes()
                   if n.label == "CoreBelief" and n.status == "found"]
        if not reactions or not beliefs:
            return []

        prompt = REINFORCES_PROMPT.format(
            language=self._language,
            reactions="\n".join(
                f"  {i}. '{_text_of(n.label, n.props)}'"
                for i, n in enumerate(reactions, 1)
            ),
            beliefs="\n".join(
                f"  {i}. '{_text_of(n.label, n.props)}'"
                for i, n in enumerate(beliefs, 1)
            ),
        )
        try:
            raw = _ollama_generate(self._host, self._model, prompt, timeout=180)
        except Exception as exc:
            print(f"[consolidate] reinforces ollama failed: {exc}",
                  file=sys.stderr)
            return []
        arr = _parse_json_array(raw)
        if not arr:
            return []
        added = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            ri = it.get("reaction")
            bi = it.get("belief")
            if (isinstance(ri, int) and 1 <= ri <= len(reactions)
                    and isinstance(bi, int) and 1 <= bi <= len(beliefs)):
                r = reactions[ri - 1]
                b = beliefs[bi - 1]
                ev = sorted(set(r.evidence) | set(b.evidence))
                graph.add_edge(r.node_id, "reinforces", b.node_id, evidence=ev)
                added.append((r.node_id, "reinforces", b.node_id))
        return added

    # ── Tier B: reframe sub-graph ───────────────────────────────────────

    def _reframe_subgraph(self, graph: GraphStore,
                          transcript: list[tuple[int, str, str]]) -> list[tuple[str, str, str]]:
        # Best-effort: for each AdaptiveResponse, link it to all ATs whose
        # text shares jaccard > 0.2 with its own text via hasAdaptiveResponse.
        added: list[tuple[str, str, str]] = []
        ats = [n for n in graph.nodes()
               if n.label == "AutomaticThought" and n.status == "found"]
        ars = [n for n in graph.nodes()
               if n.label == "AdaptiveResponse" and n.status == "found"]
        for ar in ars:
            ar_text = _text_of(ar.label, ar.props).lower()
            sa = set(ar_text.split())
            best = None
            best_score = 0.2
            for at in ats:
                at_text = _text_of(at.label, at.props).lower()
                sb = set(at_text.split())
                if not sa or not sb:
                    continue
                score = len(sa & sb) / len(sa | sb)
                if score > best_score:
                    best_score = score
                    best = at
            if best is not None:
                graph.add_edge(best.node_id, "hasAdaptiveResponse", ar.node_id,
                               evidence=sorted(set(best.evidence) | set(ar.evidence)))
                added.append((best.node_id, "hasAdaptiveResponse", ar.node_id))
        return added

    # ── Tier B: deterministic structure ────────────────────────────────

    def _structure_edges(self, graph: GraphStore) -> list[tuple[str, str, str]]:
        added: list[tuple[str, str, str]] = []
        client = next((n for n in graph.nodes() if n.label == "Client"), None)
        session = next((n for n in graph.nodes() if n.label == "Session"), None)
        if not client or not session:
            return added
        # client → session (force found)
        graph.merge_into(client.node_id, {}, 0)
        graph.merge_into(session.node_id, {"sessionType": session.props.get("sessionType") or "therapy"}, 0)
        client.status = "found"
        session.status = "found"
        graph.add_edge(client.node_id, "hasSession", session.node_id)
        added.append((client.node_id, "hasSession", session.node_id))
        for pred, label in (("hasProblem", "Problem"),
                            ("hasIntervention", "Intervention"),
                            ("hasHomework", "Homework")):
            for n in graph.nodes():
                if n.label == label and n.status == "found":
                    graph.add_edge(session.node_id, pred, n.node_id)
                    added.append((session.node_id, pred, n.node_id))
        # Goal targetsProblem (every Goal → every Problem — best-effort,
        # tightened on real session graphs by Stage 4 in batch mode).
        goals = [n for n in graph.nodes() if n.label == "Goal" and n.status == "found"]
        problems = [n for n in graph.nodes() if n.label == "Problem" and n.status == "found"]
        for g in goals:
            for p in problems:
                graph.add_edge(g.node_id, "targetsProblem", p.node_id)
                added.append((g.node_id, "targetsProblem", p.node_id))
        return added
