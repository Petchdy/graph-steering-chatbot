"""Part 2 — natural-language → V4_flat graph query engine.

parse → execute → answer:
  1. PARSE      LLM translates the therapist's NL question into a structured
                query spec over V4_flat vocabulary.
  2. EXECUTE    Deterministic Python over the canonical (nodes, edges).
                Returns a structured result set (nodes + connecting edges +
                evidence).
  3. ANSWER     LLM writes the therapist-facing answer from the result set
                only — no invention.

Source-agnostic: works over any (nodes, edges) emitted by a GraphReader.
"""

from __future__ import annotations

import json
import re

from .interfaces import Generator, GraphEdge, GraphNode
from .ontology import EDGE_MAP, NODE_CLASSES, TEXT_PROP
from .prompts import QUERY_ANSWER_PROMPT, QUERY_PARSE_PROMPT

NODE_LABEL_LIST = sorted({c["label"] for c in NODE_CLASSES})
PREDICATE_LIST = sorted({pred for _, pred, _ in EDGE_MAP})

_PARSE_NODE_BLOCK = "\n".join(f"  - {l}" for l in NODE_LABEL_LIST)
_PARSE_PRED_BLOCK = "\n".join(f"  - {p}" for p in PREDICATE_LIST)


def _ollama_chat(host: str, model: str, system: str, user: str,
                  format_json: bool = True, timeout: int = 120) -> str:
    import requests
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False, "think": False, "keep_alive": "10m",
        "options": {"temperature": 0},
    }
    if format_json:
        body["format"] = "json"
    resp = requests.post(f"{host.rstrip('/')}/api/chat", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    return raw


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(_strip_fences(raw))
    except Exception:
        return {}


# ───────────────────────────────────────────────────────────────────────────
# Query engine
# ───────────────────────────────────────────────────────────────────────────

class QueryEngine:

    def __init__(self, generator: Generator | None = None,
                 model: str = "qwen3.5-nothink",
                 host: str = "http://localhost:11434"):
        self._generator = generator   # optional — used for ANSWER if provided
        self._model = model
        self._host = host.rstrip("/")

    def answer(self, question: str, nodes: list[GraphNode],
               edges: list[GraphEdge]) -> dict:
        """Run parse → execute → answer over (nodes, edges)."""
        spec = self._parse(question)
        result_set = execute(spec, nodes, edges)
        text = self._answer(question, result_set)
        return {"spec": spec, "result_set": result_set, "answer": text}

    # ── 1. PARSE ─────────────────────────────────────────────────────

    def _parse(self, question: str) -> dict:
        prompt = QUERY_PARSE_PROMPT.format(
            node_classes=_PARSE_NODE_BLOCK,
            predicates=_PARSE_PRED_BLOCK,
            question=question,
        )
        try:
            raw = _ollama_chat(self._host, self._model, prompt, question)
        except Exception as exc:
            print(f"[query] parse ollama failed: {exc}")
            return {"intent": "summarize", "node_labels": [], "predicates": [],
                    "property_filters": {}, "free_text": question}
        spec = _parse_json(raw)
        spec.setdefault("intent", "summarize")
        spec.setdefault("node_labels", [])
        spec.setdefault("predicates", [])
        spec.setdefault("property_filters", {})
        spec.setdefault("free_text", question)
        spec["property_filters"] = _normalize_filters(spec["property_filters"])
        # Strip off-ontology terms.
        spec["node_labels"] = [l for l in spec["node_labels"]
                                if l in NODE_LABEL_LIST]
        spec["predicates"] = [p for p in spec["predicates"]
                               if p in PREDICATE_LIST]
        return spec

    # ── 3. ANSWER ────────────────────────────────────────────────────

    def _answer(self, question: str, result_set: dict) -> str:
        compact = json.dumps(result_set, ensure_ascii=False, indent=2)
        prompt = QUERY_ANSWER_PROMPT.format(question=question, result_set=compact)
        if self._generator is not None:
            try:
                # Generators return {response, technique, phase}; we only need response.
                out = self._generator.generate(prompt, [(question, "")])
                if isinstance(out, dict) and out.get("response"):
                    return str(out["response"])
            except Exception as exc:
                print(f"[query] answer via generator failed: {exc}")
        try:
            raw = _ollama_chat(self._host, self._model,
                               "You are a CBT clinical assistant.",
                               prompt, format_json=False, timeout=180)
            return _strip_fences(raw)
        except Exception as exc:
            print(f"[query] answer ollama failed: {exc}")
            return _fallback_render(result_set)


# ───────────────────────────────────────────────────────────────────────────
# Deterministic executor
# ───────────────────────────────────────────────────────────────────────────

def _text_of(node: GraphNode) -> str:
    key = TEXT_PROP.get(node.label)
    if key and isinstance(node.props.get(key), str):
        return node.props[key]
    for k in ("description", "content", "statement", "taskDescription", "text"):
        v = node.props.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _normalize_filters(filters) -> dict:
    """Strip the `Class.` qualifier off property-filter keys.

    The parse prompt lists the enums in qualified form (`Problem.domain`), so the
    model naturally emits that form — but node props are stored under the bare
    name (`domain`). Unnormalized, the lookup misses on every node and the filter
    silently empties the result set, which reads as "not in the graph".
    """
    if not isinstance(filters, dict):
        return {}
    out = {}
    for k, v in filters.items():
        key = str(k).split(".")[-1]
        if key:
            out[key] = v
    return out


def _matches_filters(node: GraphNode, filters: dict) -> bool:
    for k, v in filters.items():
        if str(node.props.get(k, "")).lower() != str(v).lower():
            return False
    return True


def build_utterance_index(nodes: list[GraphNode]) -> dict[int, list[dict]]:
    """turn index -> the dialogue spoken on that turn, from Utterance nodes.

    Every source (live / JSON / Neo4j) carries utterances as ordinary nodes, so
    this works uniformly. Sessions recorded before utterances were materialized
    simply yield an empty index and degrade to the old citation-only behaviour.
    """
    idx: dict[int, list[dict]] = {}
    for n in nodes:
        if n.label != "Utterance":
            continue
        raw_turn = n.props.get("turnIndex")
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError):
            continue
        text = n.props.get("text")
        if not isinstance(text, str) or not text:
            continue
        idx.setdefault(turn, []).append({
            "turn": turn,
            "speaker": n.props.get("speaker") or "unknown",
            "text": text,
        })
    for turn in idx:
        # client before therapist, so a quoted exchange reads in spoken order
        idx[turn].sort(key=lambda q: q["speaker"] != "client")
    return idx


_TRANSCRIPT_MAX_TURNS = 60
_TRANSCRIPT_MAX_CHARS = 400


def _transcript_view(utt_idx: dict[int, list[dict]]) -> list[dict]:
    """Ordered dialogue for `summarize`, so "what did we talk about?" has words.

    Capped and truncated: the whole result set is json.dumps'd into the ANSWER
    prompt, and an uncapped transcript would crowd out the graph content it is
    supposed to be supporting.
    """
    out: list[dict] = []
    for turn in sorted(utt_idx)[-_TRANSCRIPT_MAX_TURNS:]:
        for q in utt_idx[turn]:
            text = q["text"]
            if len(text) > _TRANSCRIPT_MAX_CHARS:
                text = text[:_TRANSCRIPT_MAX_CHARS] + "…"
            out.append({**q, "text": text})
    return out


def _quotes_for(evidence: list, utt_idx: dict[int, list[dict]]) -> list[dict]:
    out: list[dict] = []
    for ev in evidence or []:
        try:
            turn = int(ev)
        except (TypeError, ValueError):
            continue
        out.extend(utt_idx.get(turn, []))
    return out


def _node_view(n: GraphNode, utt_idx: dict[int, list[dict]] | None = None) -> dict:
    view = {
        "id": n.node_id,
        "label": n.label,
        "text": _text_of(n),
        "props": n.props,
        "evidence": n.evidence,
    }
    quotes = _quotes_for(n.evidence, utt_idx or {})
    if quotes:
        view["evidence_quotes"] = quotes
    return view


def _edge_view(e: GraphEdge, utt_idx: dict[int, list[dict]] | None = None) -> dict:
    view = {
        "subject": e.subject_id,
        "predicate": e.predicate,
        "object": e.object_id,
        "props": e.props,
        "evidence": e.evidence,
    }
    quotes = _quotes_for(e.evidence, utt_idx or {})
    if quotes:
        view["evidence_quotes"] = quotes
    return view


def execute(spec: dict, nodes: list[GraphNode],
            edges: list[GraphEdge]) -> dict:
    intent = spec.get("intent") or "summarize"
    labels = set(spec.get("node_labels") or [])
    predicates = set(spec.get("predicates") or [])
    filters = spec.get("property_filters") or {}

    by_id = {n.node_id: n for n in nodes}
    utt_idx = build_utterance_index(nodes)

    if intent == "count":
        if labels:
            counts = {l: sum(1 for n in nodes if n.label == l) for l in labels}
        else:
            counts = {}
            for n in nodes:
                counts[n.label] = counts.get(n.label, 0) + 1
        return {"intent": "count", "counts": counts}

    if intent == "summarize":
        by_label_counts: dict[str, int] = {}
        for n in nodes:
            by_label_counts[n.label] = by_label_counts.get(n.label, 0) + 1
        # Include a 2-deep cognitive chain sample.
        sample_chain = []
        for e in edges:
            if e.predicate in ("triggers", "leadsTo", "stemsFrom"):
                subj = by_id.get(e.subject_id)
                obj = by_id.get(e.object_id)
                if subj and obj:
                    sample_chain.append({
                        "subject": _node_view(subj, utt_idx),
                        "predicate": e.predicate,
                        "object": _node_view(obj, utt_idx),
                    })
        return {"intent": "summarize", "counts": by_label_counts,
                "chain_sample": sample_chain[:8],
                "transcript": _transcript_view(utt_idx)}

    # `list` / `describe` — gather matching nodes. Utterances are provenance, so
    # they only appear when asked for by name; otherwise an unfiltered "list"
    # would be swamped by every line of dialogue instead of the CBT content.
    matching = [n for n in nodes
                if (labels or n.label != "Utterance")
                and (not labels or n.label in labels)
                and _matches_filters(n, filters)]

    if intent == "describe":
        out_nodes = [_node_view(n, utt_idx) for n in matching[:5]]
        # Neighborhood: every edge touching these.
        ids = {n.node_id for n in matching[:5]}
        out_edges = [_edge_view(e, utt_idx) for e in edges
                     if e.subject_id in ids or e.object_id in ids]
        return {"intent": "describe", "nodes": out_nodes, "edges": out_edges}

    if intent == "trace":
        # Walk along given predicates starting at matching nodes.
        ids = {n.node_id for n in matching}
        walked_nodes: dict[str, GraphNode] = {n.node_id: n for n in matching}
        walked_edges: list[GraphEdge] = []
        frontier = set(ids)
        for _ in range(4):
            new_frontier: set[str] = set()
            for e in edges:
                if predicates and e.predicate not in predicates:
                    continue
                if e.subject_id in frontier and e.object_id in by_id:
                    walked_edges.append(e)
                    if e.object_id not in walked_nodes:
                        walked_nodes[e.object_id] = by_id[e.object_id]
                        new_frontier.add(e.object_id)
            if not new_frontier:
                break
            frontier = new_frontier
        return {
            "intent": "trace",
            "nodes": [_node_view(n, utt_idx) for n in walked_nodes.values()],
            "edges": [_edge_view(e, utt_idx) for e in walked_edges],
        }

    # Default: list.
    return {"intent": "list",
            "nodes": [_node_view(n, utt_idx) for n in matching]}


def _fallback_render(rs: dict) -> str:
    """Plain-text rendering for use when no generator is available."""
    intent = rs.get("intent", "list")
    if intent == "count":
        counts = rs.get("counts", {})
        if not counts:
            return "Nothing matched in this session's graph."
        return "Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items())
    if intent == "summarize":
        counts = rs.get("counts", {})
        lines = ["Graph summary:"] + [
            f"  - {k}: {v}" for k, v in sorted(counts.items())
        ]
        chain = rs.get("chain_sample", [])
        if chain:
            lines.append("Sample cognitive chain edges:")
            for c in chain:
                subj = c['subject']
                obj = c['object']
                lines.append(f"  {subj['label']}('{subj['text'][:30]}') "
                             f"--[{c['predicate']}]--> "
                             f"{obj['label']}('{obj['text'][:30]}')")
        return "\n".join(lines)
    nodes = rs.get("nodes") or []
    if not nodes:
        return "This isn't in the session's graph."
    lines = []
    for n in nodes[:8]:
        lines.append(f"- {n['label']} ({n['id']}): {n['text']}")
        for q in (n.get("evidence_quotes") or [])[:2]:
            lines.append(f"    turn {q['turn']} {q['speaker']}: \"{q['text'][:120]}\"")
    return "\n".join(lines) if lines else "(empty)"
