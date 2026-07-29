"""Neo4j-backed graph store using the V4_flat :ABox model.

The model matches `cbt_stage5_persist_v4.write_neo4j` byte-for-byte so a graph
written by Part 1 is interoperable with the V4_flat batch pipeline export.

Cypher shape:
  (:Client:ABox {id})
  (:Session:ABox {id, sessionType})
  (:<ContentLabel>:ABox {id, primaryLabel, <props>})  e.g. :CoreBelief, :Problem
  (:Utterance:ABox {id, turnIndex, speaker, text})
  (content)-[:EVIDENCED_BY]->(:Utterance)
  (:Utterance)-[:IN_SESSION]->(:Session)
  typed edges via REL_TYPE: (a)-[:TRIGGERS]->(b), (a)-[:LEADS_TO {reportedIntensity}]->(b)
  (:TBox {name})-[:SUB_CLASS_OF]->(:TBox)  class hierarchy
  (content)-[:IS_A]->(:TBox)
"""

from __future__ import annotations

import threading
from typing import Iterable

from .interfaces import GraphEdge, GraphNode, Schema
from .graph_memory import cytoscape_render, utterance_id
from .ontology import (CLASS_HIERARCHY, CONTENT_LABELS, GROUP_KEY_PROP,
                       ID_PREFIX, NODE_CLASSES, REL_TYPE, TEXT_PROP)


class Neo4jGraphStore:

    def __init__(self, schema: Schema, uri: str, user: str, password: str,
                 session_id: str = "default"):
        from neo4j import GraphDatabase
        self._schema = schema
        self._session_id = session_id
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._lock = threading.Lock()
        self._label_counters: dict[str, int] = {}
        self._session_phase = "Rapport"
        self._active_technique = "Rapport Building"
        self.reset()

    def close(self) -> None:
        self._driver.close()

    # ─────────────────────── lifecycle ──────────────────────────────

    def reset(self) -> None:
        with self._lock, self._driver.session() as s:
            # Wipe everything tied to this session id, plus the V4_flat scaffold.
            s.run("MATCH (n) DETACH DELETE n")
            # TBox
            for cls in CLASS_HIERARCHY:
                s.run("MERGE (c:TBox {name: $n})", n=cls)
            for cls, parent in CLASS_HIERARCHY.items():
                if parent:
                    s.run("MATCH (a:TBox {name:$c}),(b:TBox {name:$p}) "
                          "MERGE (a)-[:SUB_CLASS_OF]->(b)", c=cls, p=parent)
            # Property indexes for the V4_flat discriminators.
            for label, prop in GROUP_KEY_PROP.items():
                s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})")
            # Scaffold nodes.
            s.run("MERGE (c:Client:ABox {id:'client_1'})")
            s.run("MERGE (se:Session:ABox {id:'session_1', sessionType:'therapy'})")
            self._label_counters = {"Client": 1, "Session": 1}
            self._session_phase = "Rapport"
            self._active_technique = "Rapport Building"

    # ─────────────────────── node ops ───────────────────────────────

    def upsert_node(self, label: str, props: dict, turn_index: int) -> GraphNode:
        with self._lock:
            existing = self._find_similar(label, props)
            if existing is not None:
                return self._merge_into_locked(existing.node_id, props, turn_index, label)
            nid = self._new_node_id(label)
            with self._driver.session() as s:
                s.run(
                    f"MERGE (x:`{label}`:ABox {{id:$id}}) "
                    f"SET x += $props, x.primaryLabel=$label, "
                    f"x.turnAcquired=$ti",
                    id=nid, props=self._safe_props(props), label=label, ti=turn_index,
                )
                s.run(
                    "MATCH (x:ABox {id:$id}),(c:TBox {name:$label}) MERGE (x)-[:IS_A]->(c)",
                    id=nid, label=label,
                )
                # Utterance for this turn (lazily).
                self._ensure_utterance_locked(s, turn_index)
                s.run(
                    "MATCH (x:ABox {id:$id}),(u:Utterance {id:$u}) "
                    "MERGE (x)-[:EVIDENCED_BY]->(u)",
                    id=nid, u=f"utt_{turn_index}",
                )
            return GraphNode(
                node_id=nid, label=label, props=dict(props), status="found",
                evidence=[turn_index], turn_acquired=turn_index,
            )

    def merge_into(self, existing_id: str, props: dict, turn_index: int) -> GraphNode:
        with self._lock:
            return self._merge_into_locked(existing_id, props, turn_index)

    def _merge_into_locked(self, existing_id: str, props: dict, turn_index: int,
                           label: str | None = None) -> GraphNode:
        clean_props = self._safe_props(props)
        with self._driver.session() as s:
            s.run(
                "MATCH (x:ABox {id:$id}) SET x += $props, x.turnAcquired=$ti",
                id=existing_id, props=clean_props, ti=turn_index,
            )
            self._ensure_utterance_locked(s, turn_index)
            s.run(
                "MATCH (x:ABox {id:$id}),(u:Utterance {id:$u}) "
                "MERGE (x)-[:EVIDENCED_BY]->(u)",
                id=existing_id, u=f"utt_{turn_index}",
            )
            row = s.run(
                "MATCH (x:ABox {id:$id}) "
                "RETURN x.primaryLabel AS lbl, properties(x) AS props",
                id=existing_id,
            ).single()
        if row is None:
            raise KeyError(existing_id)
        return GraphNode(
            node_id=existing_id, label=row["lbl"] or label or "Unknown",
            props={k: v for k, v in (row["props"] or {}).items()
                   if k not in ("id", "primaryLabel", "turnAcquired")},
            status="found",
            evidence=[turn_index], turn_acquired=turn_index,
        )

    # ─────────────────────── edge ops ───────────────────────────────

    def add_edge(self, subj_id: str, predicate: str, obj_id: str,
                 props: dict | None = None,
                 evidence: list[int] | None = None) -> GraphEdge:
        rel = REL_TYPE.get(predicate)
        if not rel:
            raise ValueError(f"unknown predicate {predicate!r}")
        ev = sorted(evidence or [])
        ri = (props or {}).get("reportedIntensity")
        with self._driver.session() as s:
            if ri:
                s.run(
                    f"MATCH (a:ABox {{id:$s}}),(b:ABox {{id:$o}}) "
                    f"MERGE (a)-[r:`{rel}`]->(b) "
                    f"SET r.evidence=$ev, r.reportedIntensity=$ri",
                    s=subj_id, o=obj_id, ev=ev, ri=ri,
                )
            else:
                s.run(
                    f"MATCH (a:ABox {{id:$s}}),(b:ABox {{id:$o}}) "
                    f"MERGE (a)-[r:`{rel}`]->(b) "
                    f"SET r.evidence=$ev",
                    s=subj_id, o=obj_id, ev=ev,
                )
        return GraphEdge(
            subject_id=subj_id, predicate=predicate, object_id=obj_id,
            props=dict(props or {}), status="found", evidence=ev,
        )

    # ─────────────────────── readers ────────────────────────────────

    def nodes(self) -> list[GraphNode]:
        out: list[GraphNode] = []
        with self._driver.session() as s:
            for row in s.run(
                "MATCH (n:ABox) "
                "OPTIONAL MATCH (n)-[:EVIDENCED_BY]->(u:Utterance) "
                "RETURN n.id AS id, n.primaryLabel AS lbl, labels(n) AS labels, "
                "properties(n) AS props, collect(DISTINCT u.turnIndex) AS evs"
            ):
                lbl = row["lbl"]
                if not lbl:
                    # First non-ABox label is the V4_flat class.
                    others = [l for l in (row["labels"] or []) if l != "ABox"]
                    lbl = others[0] if others else "Unknown"
                props = {k: v for k, v in (row["props"] or {}).items()
                         if k not in ("id", "primaryLabel", "turnAcquired")}
                evs = [e for e in (row["evs"] or []) if e is not None]
                out.append(GraphNode(
                    node_id=row["id"], label=lbl, props=props,
                    status="found", evidence=sorted(evs),
                ))
        return out

    def edges(self) -> list[GraphEdge]:
        out: list[GraphEdge] = []
        rel_to_pred = {v: k for k, v in REL_TYPE.items()}
        with self._driver.session() as s:
            for row in s.run(
                "MATCH (a:ABox)-[r]->(b:ABox) "
                "RETURN a.id AS sid, b.id AS oid, type(r) AS rel, "
                "properties(r) AS props"
            ):
                pred = rel_to_pred.get(row["rel"])
                if not pred:
                    continue
                rp = dict(row["props"] or {})
                ev = rp.pop("evidence", []) or []
                out.append(GraphEdge(
                    subject_id=row["sid"], predicate=pred, object_id=row["oid"],
                    props=rp, status="found", evidence=list(ev),
                ))
        return out

    def count_found(self, label: str) -> int:
        with self._driver.session() as s:
            row = s.run(
                f"MATCH (n:`{label}`:ABox) RETURN count(n) AS c"
            ).single()
            return int(row["c"] or 0)

    # ─────────────────────── session state ──────────────────────────

    def apply_session_state(self, phase: str, technique: str) -> None:
        with self._lock:
            self._session_phase = phase
            self._active_technique = technique
        with self._driver.session() as s:
            s.run(
                "MATCH (se:Session {id:'session_1'}) "
                "SET se.phase=$phase, se.activeTechnique=$tech",
                phase=phase, tech=technique,
            )

    def cbt_context(self) -> str:
        nodes = self.nodes()
        by_label: dict[str, list[str]] = {}
        for n in nodes:
            if n.label in ("Client", "Session", "Utterance"):
                continue
            key = TEXT_PROP.get(n.label, "content")
            txt = n.props.get(key) or n.props.get("description") or n.props.get("content")
            if isinstance(txt, str) and txt:
                by_label.setdefault(n.label, []).append(txt[:80])
        lines = [f"Session phase: {self._session_phase}",
                 f"Active CBT technique: {self._active_technique}",
                 "What's emerged in the graph so far:"]
        if not by_label:
            lines.append("  (nothing yet — still rapport)")
        else:
            for cls in CONTENT_LABELS:
                items = by_label.get(cls)
                if not items:
                    continue
                lines.append(f"  {cls}: " + " | ".join(items[:5])
                             + (f" (+{len(items)-5} more)" if len(items) > 5 else ""))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "session_phase": self._session_phase,
            "active_technique": self._active_technique,
            "counts": {cls["label"]: self.count_found(cls["label"])
                       for cls in NODE_CLASSES},
        }

    def cytoscape(self) -> dict:
        return cytoscape_render(self.nodes(), self.edges())

    # ─────────────────────── internals ──────────────────────────────

    def _new_node_id(self, label: str) -> str:
        n = self._label_counters.get(label, 0) + 1
        self._label_counters[label] = n
        prefix = ID_PREFIX.get(label, label.lower())
        return f"{prefix}_{n}"

    def _find_similar(self, label: str, props: dict) -> GraphNode | None:
        text_key = TEXT_PROP.get(label)
        cand_text = props.get(text_key) if text_key else None
        if not cand_text:
            return None
        sa = set(str(cand_text).lower().split())
        for n in self.nodes():
            if n.label != label:
                continue
            existing_text = n.props.get(text_key) if text_key else None
            if not existing_text:
                continue
            sb = set(str(existing_text).lower().split())
            if not sa or not sb:
                continue
            if len(sa & sb) / len(sa | sb) > 0.6:
                return n
        return None

    def _safe_props(self, props: dict) -> dict:
        """Neo4j won't accept dict / list of objects as a property value."""
        out = {}
        for k, v in props.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif v is None:
                continue
            else:
                out[k] = str(v)
        return out

    def record_utterance(self, turn_index: int, speaker: str,
                         text: str) -> GraphNode:
        """Store the dialogue text on the Utterance node (see graph_memory for why).

        For speaker='client' the id is the same `utt_<turn>` that
        _ensure_utterance_locked creates as the EVIDENCED_BY target, so this MERGE
        enriches that node in place rather than creating a parallel one.
        """
        uid = utterance_id(turn_index, speaker)
        with self._lock:
            with self._driver.session() as s:
                s.run(
                    "MERGE (u:Utterance:ABox {id:$id}) "
                    "SET u.turnIndex=$ti, u.speaker=$sp, u.text=$tx",
                    id=uid, ti=turn_index, sp=speaker, tx=text,
                )
                s.run(
                    "MATCH (u:Utterance {id:$id}),(se:Session {id:'session_1'}) "
                    "MERGE (u)-[:IN_SESSION]->(se)",
                    id=uid,
                )
        return GraphNode(
            node_id=uid, label="Utterance",
            props={"text": text, "speaker": speaker, "turnIndex": turn_index},
            status="found", evidence=[turn_index], turn_acquired=turn_index,
        )

    def _ensure_utterance_locked(self, s, turn_index: int) -> None:
        # Only the id/turnIndex scaffold — speaker/text arrive via record_utterance,
        # which MERGEs onto this same id.
        s.run(
            "MERGE (u:Utterance:ABox {id:$id}) "
            "SET u.turnIndex=$ti",
            id=f"utt_{turn_index}", ti=turn_index,
        )
        s.run(
            "MATCH (u:Utterance {id:$id}),(se:Session {id:'session_1'}) "
            "MERGE (u)-[:IN_SESSION]->(se)",
            id=f"utt_{turn_index}",
        )
