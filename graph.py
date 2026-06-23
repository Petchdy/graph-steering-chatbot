"""Swappable graph store implementations.

InMemoryGraphStore - no external dependency, default for demo and tests.
Neo4jGraphStore    - production backend. ALL Cypher lives here.

Both derive their slot set from the injected Schema; field keys are never hardcoded.
"""

import json

from interfaces import Schema, GraphNode, GraphEdge

_SESSION_KEYS = {"session_phase", "active_technique"}


class InMemoryGraphStore:

    def __init__(self, schema: Schema):
        self._schema = schema
        self._fields_by_priority = sorted(schema.fields(), key=lambda f: f.priority)
        self._state: dict[str, dict] = {}
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._label_counters: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self._state = {
            f.key: {"value": None, "acquired": False, "turns": []}
            for f in self._fields_by_priority
        }
        self._nodes = {}
        self._edges = {}
        self._label_counters = {}
        for cls in self._schema.node_classes():
            label = cls["label"]
            nid = self._new_node_id(label)
            self._nodes[nid] = GraphNode(
                node_id=nid, label=label, status="missing", props={}, turn_acquired=None
            )
        for subj_label, predicate, obj_label in self._schema.edge_map():
            subj = self._first_node(subj_label)
            obj = self._first_node(obj_label)
            if subj and obj:
                eid = f"{subj.node_id}__{predicate}__{obj.node_id}"
                if eid not in self._edges:
                    self._edges[eid] = GraphEdge(
                        edge_id=eid, predicate=predicate,
                        subject_id=subj.node_id, object_id=obj.node_id,
                        status="missing", turn_acquired=None
                    )

    def apply_deltas(self, deltas: dict[str, str], turn_id: int) -> None:
        for key, value in deltas.items():
            if key not in self._state:
                continue
            entry = self._state[key]
            entry["value"] = value
            entry["acquired"] = True
            entry["turns"].append(turn_id)

    def missing(self) -> list[str]:
        return [
            f.key for f in self._fields_by_priority
            if not self._state[f.key]["acquired"] and f.priority > 0
        ]

    def acquired_summary(self) -> str:
        acquired = [
            f"{key}={entry['value']}"
            for key, entry in self._state.items()
            if entry["acquired"] and key not in _SESSION_KEYS
        ]
        return ", ".join(acquired) if acquired else "(nothing acquired yet)"

    def snapshot(self) -> dict:
        return {
            key: {"value": entry["value"], "acquired": entry["acquired"]}
            for key, entry in self._state.items()
        }

    def cbt_context(self) -> str:
        phase = (self._state.get("session_phase") or {}).get("value") or "Rapport"
        technique = (self._state.get("active_technique") or {}).get("value") or "none yet"
        acquired_lines = [
            f'  {key}="{entry["value"]}"'
            for key, entry in self._state.items()
            if entry["acquired"] and key not in _SESSION_KEYS
        ]
        missing_keys = self.missing()
        return (
            f"Session phase: {phase}\n"
            f"Active CBT technique: {technique}\n"
            f"What we know so far:\n"
            + ("\n".join(acquired_lines) if acquired_lines else "  (nothing yet)") + "\n"
            + f"Still to explore (soft hints, not a checklist): "
            + (", ".join(missing_keys) if missing_keys else "none")
        )

    def apply_session_state(self, phase: str, technique: str) -> None:
        for key, value in (("session_phase", phase), ("active_technique", technique)):
            if key in self._state:
                self._state[key]["value"] = value
                self._state[key]["acquired"] = True

    # ── Rich graph methods ────────────────────────────────────────────────────

    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def edges(self) -> list[GraphEdge]:
        return list(self._edges.values())

    def upsert_node(self, label: str, props: dict, turn_id: int) -> GraphNode:
        for node in self._found_nodes(label):
            if self._similar(node.props, props):
                node.props.update(props)
                node.turn_acquired = turn_id
                return node
        for node in self._nodes.values():
            if node.label == label and node.status == "missing":
                node.status = "found"
                node.props = props
                node.turn_acquired = turn_id
                return node
        nid = self._new_node_id(label)
        node = GraphNode(node_id=nid, label=label, status="found",
                         props=props, turn_acquired=turn_id)
        self._nodes[nid] = node
        return node

    def resolve_edge(self, subject_id: str, predicate: str,
                     object_id: str, turn_id: int) -> GraphEdge:
        eid = f"{subject_id}__{predicate}__{object_id}"
        if eid in self._edges:
            self._edges[eid].status = "found"
            self._edges[eid].turn_acquired = turn_id
        else:
            self._edges[eid] = GraphEdge(
                edge_id=eid, predicate=predicate,
                subject_id=subject_id, object_id=object_id,
                status="found", turn_acquired=turn_id
            )
        return self._edges[eid]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _new_node_id(self, label: str) -> str:
        n = self._label_counters.get(label, 0) + 1
        self._label_counters[label] = n
        return f"{label}_{n}"

    def _first_node(self, label: str) -> GraphNode | None:
        for node in self._nodes.values():
            if node.label == label:
                return node
        return None

    def _found_nodes(self, label: str) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.label == label and n.status == "found"]

    def _similar(self, props_a: dict, props_b: dict) -> bool:
        def words(p: dict) -> set:
            text = p.get("content") or p.get("description") or p.get("statement") or ""
            return set(text.lower().split())
        a, b = words(props_a), words(props_b)
        if not a or not b:
            return props_a == props_b
        return len(a & b) / len(a | b) > 0.6


class Neo4jGraphStore:
    """Neo4j-backed graph store.

    Graph model:
      (:Session {id}) -[:HAS_FIELD]-> (:Field {key, value, acquired, priority})
      (:Field)-[:ACQUIRED_FROM]->(:Turn {id}) evidence edges.
      (:Session)-[:HAS_CLASS_NODE]->(:ClassNode {label, node_id, status, props_json})
      (:ClassNode)-[:PLACEHOLDER_EDGE {predicate, status}]->(:ClassNode)
    """

    def __init__(self, schema: Schema, uri: str, user: str, password: str, session_id: str = "default"):
        from neo4j import GraphDatabase
        self._schema = schema
        self._fields_by_priority = sorted(schema.fields(), key=lambda f: f.priority)
        self._session_id = session_id
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.reset()

    def close(self) -> None:
        self._driver.close()

    def reset(self) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (s:Session {id: $sid}) "
                "OPTIONAL MATCH (s)-[:HAS_FIELD]->(f:Field) "
                "OPTIONAL MATCH (f)-[:ACQUIRED_FROM]->(t:Turn) "
                "OPTIONAL MATCH (s)-[:HAS_CLASS_NODE]->(n:ClassNode) "
                "DETACH DELETE s, f, t, n",
                sid=self._session_id,
            )
            session.run("MERGE (s:Session {id: $sid})", sid=self._session_id)
            for f in self._fields_by_priority:
                session.run(
                    """
                    MATCH (s:Session {id: $sid})
                    MERGE (s)-[:HAS_FIELD]->(field:Field {key: $key})
                    SET field.value = null, field.acquired = false, field.priority = $priority
                    """,
                    sid=self._session_id, key=f.key, priority=f.priority,
                )
            for cls in self._schema.node_classes():
                label = cls["label"]
                nid = f"{label}_1"
                session.run(
                    """
                    MATCH (s:Session {id: $sid})
                    MERGE (s)-[:HAS_CLASS_NODE]->(n:ClassNode {node_id: $nid})
                    SET n.label = $label, n.status = 'missing', n.props_json = '{}'
                    """,
                    sid=self._session_id, label=label, nid=nid,
                )
            for subj_label, pred, obj_label in self._schema.edge_map():
                session.run(
                    """
                    MATCH (s:Session {id: $sid})
                    MATCH (s)-[:HAS_CLASS_NODE]->(a:ClassNode {label: $sl})
                    MATCH (s)-[:HAS_CLASS_NODE]->(b:ClassNode {label: $ol})
                    MERGE (a)-[e:PLACEHOLDER_EDGE {predicate: $pred}]->(b)
                    SET e.status = 'missing'
                    """,
                    sid=self._session_id, sl=subj_label, pred=pred, ol=obj_label,
                )

    def apply_deltas(self, deltas: dict[str, str], turn_id: int) -> None:
        if not deltas:
            return
        with self._driver.session() as session:
            session.run("MERGE (t:Turn {id: $turn_id})", turn_id=turn_id)
            for key, value in deltas.items():
                session.run(
                    """
                    MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(field:Field {key: $key})
                    MATCH (t:Turn {id: $turn_id})
                    SET field.value = $value, field.acquired = true
                    MERGE (field)-[:ACQUIRED_FROM]->(t)
                    """,
                    sid=self._session_id, key=key, value=value, turn_id=turn_id,
                )

    def missing(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(field:Field)
                WHERE field.acquired = false AND field.priority > 0
                RETURN field.key AS key ORDER BY field.priority ASC
                """,
                sid=self._session_id,
            )
            return [r["key"] for r in result]

    def acquired_summary(self) -> str:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(field:Field)
                WHERE field.acquired = true AND field.priority > 0
                RETURN field.key AS key, field.value AS value ORDER BY field.priority ASC
                """,
                sid=self._session_id,
            )
            acquired = [f"{r['key']}={r['value']}" for r in result]
            return ", ".join(acquired) if acquired else "(nothing acquired yet)"

    def snapshot(self) -> dict:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(field:Field)
                RETURN field.key AS key, field.value AS value,
                       field.acquired AS acquired, field.priority AS priority
                ORDER BY field.priority ASC
                """,
                sid=self._session_id,
            )
            return {
                r["key"]: {"value": r["value"], "acquired": r["acquired"]}
                for r in result
            }

    def cbt_context(self) -> str:
        with self._driver.session() as session:
            state_result = session.run(
                """
                MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(f:Field)
                WHERE f.key IN ['session_phase', 'active_technique']
                RETURN f.key AS key, f.value AS value
                """,
                sid=self._session_id,
            )
            state = {r["key"]: r["value"] for r in state_result}

            acquired_result = session.run(
                """
                MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(f:Field)
                WHERE f.acquired = true AND f.priority > 0
                RETURN f.key AS key, f.value AS value ORDER BY f.priority ASC
                """,
                sid=self._session_id,
            )
            acquired_lines = [f'  {r["key"]}="{r["value"]}"' for r in acquired_result]

        phase = state.get("session_phase") or "Rapport"
        technique = state.get("active_technique") or "none yet"
        missing_keys = self.missing()
        return (
            f"Session phase: {phase}\n"
            f"Active CBT technique: {technique}\n"
            f"What we know so far:\n"
            + ("\n".join(acquired_lines) if acquired_lines else "  (nothing yet)") + "\n"
            + f"Still to explore (soft hints, not a checklist): "
            + (", ".join(missing_keys) if missing_keys else "none")
        )

    def apply_session_state(self, phase: str, technique: str) -> None:
        with self._driver.session() as session:
            for key, value in (("session_phase", phase), ("active_technique", technique)):
                session.run(
                    """
                    MATCH (s:Session {id: $sid})-[:HAS_FIELD]->(f:Field {key: $key})
                    SET f.value = $value, f.acquired = true
                    """,
                    sid=self._session_id, key=key, value=value,
                )

    def nodes(self) -> list[GraphNode]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(n:ClassNode) "
                "RETURN n.node_id AS nid, n.label AS label, n.status AS status, "
                "n.props_json AS props_json, n.turn_acquired AS turn_acquired",
                sid=self._session_id,
            )
            nodes = []
            for r in result:
                try:
                    props = json.loads(r["props_json"] or "{}")
                except Exception:
                    props = {}
                nodes.append(GraphNode(
                    node_id=r["nid"], label=r["label"], status=r["status"] or "missing",
                    props=props, turn_acquired=r["turn_acquired"],
                ))
            return nodes

    def edges(self) -> list[GraphEdge]:
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(a:ClassNode)
                MATCH (a)-[e:PLACEHOLDER_EDGE]->(b:ClassNode)
                MATCH (sess)-[:HAS_CLASS_NODE]->(b)
                RETURN a.node_id AS subject_id, e.predicate AS predicate,
                       b.node_id AS object_id, e.status AS status,
                       e.turn_acquired AS turn_acquired
                """,
                sid=self._session_id,
            )
            edges = []
            for r in result:
                eid = f"{r['subject_id']}__{r['predicate']}__{r['object_id']}"
                edges.append(GraphEdge(
                    edge_id=eid, predicate=r["predicate"],
                    subject_id=r["subject_id"], object_id=r["object_id"],
                    status=r["status"] or "missing", turn_acquired=r["turn_acquired"],
                ))
            return edges

    def upsert_node(self, label: str, props: dict, turn_id: int) -> GraphNode:
        props_json = json.dumps(props)
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(n:ClassNode {label: $label, status: 'missing'})
                SET n.status = 'found', n.props_json = $props_json, n.turn_acquired = $turn_id
                RETURN n.node_id AS nid LIMIT 1
                """,
                sid=self._session_id, label=label, props_json=props_json, turn_id=turn_id,
            )
            record = result.single()
            if record:
                return GraphNode(node_id=record["nid"], label=label, status="found",
                                 props=props, turn_acquired=turn_id)
            cnt_result = s.run(
                "MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(n:ClassNode {label: $label}) "
                "RETURN count(n) AS cnt",
                sid=self._session_id, label=label,
            )
            cnt = cnt_result.single()["cnt"]
            nid = f"{label}_{cnt + 1}"
            s.run(
                """
                MATCH (sess:Session {id: $sid})
                CREATE (sess)-[:HAS_CLASS_NODE]->(n:ClassNode {
                    label: $label, node_id: $nid, status: 'found',
                    props_json: $props_json, turn_acquired: $turn_id
                })
                """,
                sid=self._session_id, label=label, nid=nid,
                props_json=props_json, turn_id=turn_id,
            )
            return GraphNode(node_id=nid, label=label, status="found",
                             props=props, turn_acquired=turn_id)

    def resolve_edge(self, subject_id: str, predicate: str,
                     object_id: str, turn_id: int) -> GraphEdge:
        eid = f"{subject_id}__{predicate}__{object_id}"
        with self._driver.session() as s:
            s.run(
                """
                MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(a:ClassNode {node_id: $sid_n})
                MATCH (sess)-[:HAS_CLASS_NODE]->(b:ClassNode {node_id: $oid})
                MERGE (a)-[e:PLACEHOLDER_EDGE {predicate: $pred}]->(b)
                SET e.status = 'found', e.turn_acquired = $turn_id
                """,
                sid=self._session_id, sid_n=subject_id, oid=object_id,
                pred=predicate, turn_id=turn_id,
            )
        return GraphEdge(
            edge_id=eid, predicate=predicate,
            subject_id=subject_id, object_id=object_id,
            status="found", turn_acquired=turn_id,
        )
