"""In-memory V4_flat node/edge graph store.

Placeholder behavior: on reset() pre-creates one placeholder node per content
class and one placeholder edge per (subj, pred, obj) tuple in the schema's
edge map. upsert_node() flips the first instance to status='found' and fills
its props; further instances create new found nodes. add_edge() flips a
matching placeholder edge or creates a new one.
"""

from __future__ import annotations

import threading
from typing import Iterable

from .interfaces import GraphEdge, GraphNode, Schema
from .ontology import (CONTENT_LABELS, ID_PREFIX, NODE_CLASSES, REACTION_CHANNELS,
                       TEXT_PROP, apply_gating_constraints)


def _text_of(label: str, props: dict) -> str:
    key = TEXT_PROP.get(label)
    if key and isinstance(props.get(key), str):
        return props[key]
    for fallback in ("description", "content", "statement", "taskDescription", "text"):
        v = props.get(fallback)
        if isinstance(v, str) and v:
            return v
    return ""


def utterance_id(turn_index: int, speaker: str) -> str:
    """Deterministic Utterance id, shared by both graph stores.

    The client utterance keeps the bare `utt_<turn>` form because that is the id
    Neo4jGraphStore already lazily creates as the EVIDENCED_BY target, so
    recording the text MERGEs into the same node instead of forking a duplicate.
    """
    return f"utt_{turn_index}" if speaker == "client" else f"utt_{turn_index}_t"


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class InMemoryGraphStore:

    def __init__(self, schema: Schema):
        self._schema = schema
        self._lock = threading.Lock()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._label_counters: dict[str, int] = {}
        self._session_phase: str = "Rapport"
        self._active_technique: str = "Rapport Building"
        self.reset()

    # ─────────────────────────── lifecycle ─────────────────────────────

    def reset(self) -> None:
        with self._lock:
            self._nodes = {}
            self._edges = {}
            self._label_counters = {}
            self._session_phase = "Rapport"
            self._active_technique = "Rapport Building"
            # One placeholder node per class.
            for cls in NODE_CLASSES:
                nid = self._new_node_id(cls["label"])
                self._nodes[nid] = GraphNode(
                    node_id=nid, label=cls["label"], props={}, status="missing",
                )
            # One placeholder edge per (subj, pred, obj) tuple in the schema.
            for subj_label, pred, obj_label in self._schema.edge_map():
                subj = self._first_node(subj_label)
                obj = self._first_node(obj_label)
                if not subj or not obj:
                    continue
                key = f"{subj.node_id}__{pred}__{obj.node_id}"
                if key in self._edges:
                    continue
                self._edges[key] = GraphEdge(
                    subject_id=subj.node_id, predicate=pred, object_id=obj.node_id,
                    status="missing",
                )

    # ─────────────────────────── node ops ──────────────────────────────

    def upsert_node(self, label: str, props: dict, turn_index: int) -> GraphNode:
        """Add or refine a node of the given V4_flat label.

        Try to merge into an existing found node of the same label whose main
        text is jaccard-similar (>0.6). Otherwise flip the first matching
        placeholder, or create a new found node.
        """
        with self._lock:
            cand_text = _text_of(label, props)
            for node in self._found_nodes(label):
                existing_text = _text_of(label, node.props)
                if existing_text and cand_text and _jaccard(existing_text, cand_text) > 0.6:
                    # Never merge CoreBeliefs with different categories — different
                    # categories are different nodes by definition (v7.2 §2.1).
                    if label == "CoreBelief":
                        ec = node.props.get("category")
                        cc = props.get("category")
                        if ec and cc and ec != cc:
                            continue
                    return self._merge_into_locked(node, props, turn_index)
            # Use a missing placeholder if available.
            for node in self._nodes.values():
                if node.label == label and node.status == "missing":
                    node.status = "found"
                    node.props = apply_gating_constraints(label, dict(props))
                    node.turn_acquired = turn_index
                    if turn_index not in node.evidence:
                        node.evidence.append(turn_index)
                    return node
            return self._new_found_node_locked(label, props, turn_index)

    def record_utterance(self, turn_index: int, speaker: str,
                         text: str) -> GraphNode:
        """Materialize a dialogue turn as an Utterance node.

        Provenance, not extraction: keyed by (turn, speaker) rather than merged by
        text similarity, and deliberately not passed through
        apply_gating_constraints (those rules are for content classes). This is
        what lets Part 2 resolve a node's `evidence` turn indices back into the
        words the client actually said — without it the query engine can cite
        "turn 7" but never quote it.
        """
        uid = utterance_id(turn_index, speaker)
        with self._lock:
            node = self._nodes.get(uid)
            if node is None:
                node = GraphNode(node_id=uid, label="Utterance", props={},
                                 status="found", evidence=[],
                                 turn_acquired=turn_index)
                self._nodes[uid] = node
            node.status = "found"
            node.props = {"text": text, "speaker": speaker,
                          "turnIndex": turn_index}
            if turn_index not in node.evidence:
                node.evidence.append(turn_index)
            return node

    def merge_into(self, existing_id: str, props: dict, turn_index: int) -> GraphNode:
        with self._lock:
            node = self._nodes.get(existing_id)
            if node is None:
                raise KeyError(f"node {existing_id} not found")
            return self._merge_into_locked(node, props, turn_index)

    def _merge_into_locked(self, node: GraphNode, props: dict,
                           turn_index: int) -> GraphNode:
        for k, v in props.items():
            # Keep the richer existing value when the incoming one is empty.
            if v is None or v == "":
                continue
            node.props[k] = v
        apply_gating_constraints(node.label, node.props)
        if turn_index not in node.evidence:
            node.evidence.append(turn_index)
        node.turn_acquired = turn_index
        return node

    # ─────────────────────────── edge ops ──────────────────────────────

    def add_edge(self, subj_id: str, predicate: str, obj_id: str,
                 props: dict | None = None,
                 evidence: list[int] | None = None) -> GraphEdge:
        with self._lock:
            key = f"{subj_id}__{predicate}__{obj_id}"
            existing = self._edges.get(key)
            if existing is None:
                edge = GraphEdge(
                    subject_id=subj_id, predicate=predicate, object_id=obj_id,
                    props=dict(props or {}), status="found",
                    evidence=list(evidence or []),
                )
                self._edges[key] = edge
                return edge
            existing.status = "found"
            if props:
                existing.props.update(props)
            if evidence:
                for ev in evidence:
                    if ev not in existing.evidence:
                        existing.evidence.append(ev)
            return existing

    # ─────────────────────────── readers ──────────────────────────────

    def nodes(self) -> list[GraphNode]:
        with self._lock:
            return list(self._nodes.values())

    def edges(self) -> list[GraphEdge]:
        with self._lock:
            return list(self._edges.values())

    def count_found(self, label: str) -> int:
        with self._lock:
            return sum(1 for n in self._nodes.values()
                       if n.label == label and n.status == "found")

    # ─────────────────────────── session state ────────────────────────

    def apply_session_state(self, phase: str, technique: str) -> None:
        with self._lock:
            self._session_phase = phase
            self._active_technique = technique

    def cbt_context(self) -> str:
        """Compact found-node summary grouped by class, plus phase/technique."""
        with self._lock:
            phase = self._session_phase
            technique = self._active_technique
            by_label: dict[str, list[str]] = {}
            for n in self._nodes.values():
                if n.status != "found":
                    continue
                if n.label in ("Client", "Session", "Utterance"):
                    continue
                txt = _text_of(n.label, n.props)
                if not txt:
                    continue
                by_label.setdefault(n.label, []).append(txt[:80])
        lines = [f"Session phase: {phase}", f"Active CBT technique: {technique}",
                 "What's emerged in the graph so far:"]
        if not by_label:
            lines.append("  (nothing yet — still rapport)")
        else:
            for cls in CONTENT_LABELS:
                items = by_label.get(cls)
                if not items:
                    continue
                lines.append(f"  {cls}: " + " | ".join(items[:5])
                             + (f" (+{len(items) - 5} more)" if len(items) > 5 else ""))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "session_phase": self._session_phase,
                "active_technique": self._active_technique,
                "counts": {
                    cls["label"]: self._count_found_locked(cls["label"])
                    for cls in NODE_CLASSES
                },
            }

    def _count_found_locked(self, label: str) -> int:
        return sum(1 for n in self._nodes.values()
                   if n.label == label and n.status == "found")

    def cytoscape(self) -> dict:
        return _cytoscape_render(self.nodes(), self.edges())

    # ─────────────────────────── internals ──────────────────────────────

    def _new_node_id(self, label: str) -> str:
        n = self._label_counters.get(label, 0) + 1
        self._label_counters[label] = n
        prefix = ID_PREFIX.get(label, label.lower())
        return f"{prefix}_{n}"

    def _first_node(self, label: str) -> GraphNode | None:
        for n in self._nodes.values():
            if n.label == label:
                return n
        return None

    def _found_nodes(self, label: str) -> list[GraphNode]:
        return [n for n in self._nodes.values()
                if n.label == label and n.status == "found"]

    def _new_found_node_locked(self, label: str, props: dict,
                               turn_index: int) -> GraphNode:
        nid = self._new_node_id(label)
        node = GraphNode(
            node_id=nid, label=label,
            props=apply_gating_constraints(label, dict(props)),
            status="found", evidence=[turn_index], turn_acquired=turn_index,
        )
        self._nodes[nid] = node
        return node


# ───────────────────────────── Cytoscape rendering ────────────────────────

_TYPE_BY_LABEL = {
    "Session": "session", "Client": "session",
    "Problem": "session_structure", "Goal": "session_structure",
    "Intervention": "session_structure", "Homework": "session_structure",
    "CoreBelief": "cognitive", "IntermediateBelief": "cognitive",
    "Situation": "cognitive", "AutomaticThought": "cognitive",
    "Reaction": "cognitive", "AdaptiveResponse": "cognitive",
    "Utterance": "provenance",
}

_CONTENT_KEYS = ("content", "description", "statement", "taskDescription", "text")


def _node_type(label: str, status: str) -> str:
    if status == "missing":
        return "missing"
    return _TYPE_BY_LABEL.get(label, "field")


def _content_preview(props: dict) -> str:
    for key in _CONTENT_KEYS:
        val = props.get(key)
        if val:
            return str(val)[:25]
    return ""


def _cytoscape_render(nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]) -> dict:
    out_nodes = []
    for n in nodes:
        preview = _content_preview(n.props)
        label = (f"{n.label}\n{preview}"
                 if n.status == "found" and preview else n.label)
        out_nodes.append({"data": {
            "id": n.node_id,
            "label": label,
            "type": _node_type(n.label, n.status),
            "status": n.status,
            "raw_label": n.label,
        }})
    out_edges = []
    for e in edges:
        if e.status != "found":
            continue                  # suppress placeholder edges (§UI-1.1 §2.2)
        out_edges.append({"data": {
            "id": e.edge_id,
            "source": e.subject_id,
            "target": e.object_id,
            "label": e.predicate,
            "predicate": e.predicate,
            "status": e.status,
        }})
    return {"nodes": out_nodes, "edges": out_edges}


def cytoscape_render(nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]) -> dict:
    """Public helper — same rendering, but for nodes/edges loaded via GraphReader."""
    return _cytoscape_render(nodes, edges)
