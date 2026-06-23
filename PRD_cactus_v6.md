# PRD: CACTUS CBT Chatbot — v6 (Rich Graph + Async Extraction)

**Version:** 6.0  
**Status:** Ready for implementation by Claude Code  
**Based on:** Current codebase state as read from project knowledge (post-v5 files already committed)  
**Model:** `qwen3.5-nothink` via Ollama native API

---

## 0. What Already Exists (Do Not Re-implement)

The following are **already in the current codebase** and must not be touched unless a section below explicitly says to edit them:

| File | Current state |
|---|---|
| `interfaces.py` | `OntologyField`, `Schema`, `GraphStore` (with `cbt_context`, `apply_session_state`), `Extractor`, `Generator` (returns `dict`) — **complete** |
| `schema.py` | `CBTSchema` with 9 clinical fields (priority 1–9) + 2 session-state fields (priority 0) — **complete** |
| `graph.py` | `InMemoryGraphStore` and `Neo4jGraphStore` with `cbt_context()`, `apply_session_state()`, priority-0 exclusion in `missing()` — **complete** |
| `generate.py` | `LocalLLMGenerator` using native `/api/chat`, returns `dict`, `EchoGenerator` returns dict — **complete** |
| `orchestrator.py` | `Session` dataclass, `validate_phase()`, `turn()` loop — **complete** |
| `prompts.py` | `CBT_SYSTEM_PROMPT`, `CBT_EXTRACTION_PROMPT` — **complete** |
| `api.py` | `POST /chat`, `POST /reset`, `GET /graph/{session_id}`, Gradio mount — **complete** |
| `ui.py` | Two-column Gradio layout, Cytoscape iframe panel, phase/technique display — **complete** |
| `factory.py` | `CBTSchema`, `qwen3.5-nothink` defaults — **complete** |
| `tests/test_orchestrator.py` | All existing tests passing — **do not break** |

---

## 1. What This PRD Adds

Three distinct upgrades, each self-contained:

1. **Rich graph with placeholder nodes + status + edges** — replaces the flat slot-filler with a proper node/edge graph matching `cbt_kg_ontology_v4_flat.txt`. Nodes pre-exist as placeholders at session start. Multiple instances per class are supported. Edges have status too.

2. **Per-turn mini extraction pipeline** — replaces the single `extractor.extract()` call with a 4-step pipeline: extract+atomize → normalize → apply nodes → resolve edges. Uses ±2 exchange window.

3. **Async-first / sync-opportunistic turn loop** — extraction runs as an async task; generate starts immediately using pre-turn graph state; if extraction finishes first (within `EXTRACTION_TIMEOUT` seconds) the graph is already updated before generate completes (Option 1). Otherwise generate returns and extraction updates the graph in the background (Option 2). Per-session lock prevents concurrent write races.

---

## 2. The Dependency Rule (Unchanged — Load-Bearing)

**Only `factory.py` may import concrete classes.** Every other module imports only from `interfaces.py`. This rule applies to all new code. Do not break it.

---

## 3. File-by-File Changes

### 3.1 `interfaces.py` — Add graph node/edge types + new protocols

Add the following **after** the existing `OntologyField` dataclass. Do not remove or modify anything already there.

```python
# ---------- Graph node / edge types (used by GraphStore internals + /graph endpoint) ----------

@dataclass
class GraphNode:
    """A node in the CBT knowledge graph."""
    node_id: str          # unique within session e.g. "Situation_1", "Situation_2"
    label: str            # ontology class: "Situation", "AutomaticThought", etc.
    status: str           # "missing" | "found"
    props: dict           # {"content": "...", "domain": "self", ...} — empty when missing
    turn_acquired: int | None = None   # turn number when first populated


@dataclass
class GraphEdge:
    """A directed edge in the CBT knowledge graph."""
    edge_id: str          # unique e.g. "Situation_1__triggers__AutomaticThought_1"
    predicate: str        # e.g. "triggers", "leadsTo", "stemsFrom"
    subject_id: str       # node_id of subject
    object_id: str        # node_id of object
    status: str           # "missing" | "found"
    turn_acquired: int | None = None
```

Add these methods to the `GraphStore` Protocol (append after `apply_session_state`):

```python
def nodes(self) -> list[GraphNode]: ...
    """Return all nodes (placeholder + found) in the graph."""

def edges(self) -> list[GraphEdge]: ...
    """Return all edges (missing + found) in the graph."""

def upsert_node(self, label: str, props: dict, turn_id: int) -> GraphNode: ...
    """
    Find or create a node of the given label.
    - If a placeholder node for this label exists and no found node with
      similar content exists → upgrade placeholder to found, set props.
    - If a sufficiently similar found node exists → merge (overwrite props).
    - Otherwise → create a new found node (multi-instance support).
    Returns the resulting GraphNode.
    """

def resolve_edge(self, subject_id: str, predicate: str, object_id: str,
                 turn_id: int) -> GraphEdge: ...
    """
    Mark an edge as found. Creates the edge if it doesn't exist yet.
    Both subject and object must already exist as nodes (found or placeholder).
    Returns the resulting GraphEdge.
    """
```

Keep all existing Protocol methods. Do not change their signatures.

---

### 3.2 `schema.py` — Add edge map and node classes

Add after the existing `CBTSchema` class (do not modify `CBTSchema.fields()` or `CBTSchema.render()`):

```python
# Ontology node classes — used by GraphStore to pre-create placeholder nodes.
# "multi" = True means multiple instances of this class can exist in one session.
CBT_NODE_CLASSES: list[dict] = [
    {"label": "Problem",             "multi": True},
    {"label": "Goal",                "multi": True},
    {"label": "Intervention",        "multi": True},
    {"label": "Homework",            "multi": True},
    {"label": "CoreBelief",          "multi": True},
    {"label": "IntermediateBelief",  "multi": True},
    {"label": "Situation",           "multi": True},
    {"label": "AutomaticThought",    "multi": True},
    {"label": "Reaction",            "multi": True},
    {"label": "AdaptiveResponse",    "multi": True},
    # Session-level scaffold (single-instance)
    {"label": "Session",             "multi": False},
    {"label": "Client",              "multi": False},
]

# Edge map — (subject_label, predicate, object_label).
# Used by GraphStore.reset() to pre-create placeholder edges,
# and by the extraction pipeline to know which edges to probe after a node is found.
CBT_EDGE_MAP: list[tuple[str, str, str]] = [
    # Cognitive chain
    ("CoreBelief",         "givesRiseTo",            "IntermediateBelief"),
    ("IntermediateBelief", "influencesPerceptionOf",  "Situation"),
    ("Situation",          "triggers",               "AutomaticThought"),
    ("AutomaticThought",   "leadsTo",                "Reaction"),
    ("AutomaticThought",   "stemsFrom",              "CoreBelief"),
    ("AutomaticThought",   "hasAdaptiveResponse",    "AdaptiveResponse"),
    ("Reaction",           "reinforces",             "CoreBelief"),
    ("Reaction",           "becomesSituation",       "Situation"),
    # Structure
    ("Client",             "hasSession",             "Session"),
    ("Session",            "hasProblem",             "Problem"),
    ("Session",            "hasIntervention",        "Intervention"),
    ("Session",            "hasHomework",            "Homework"),
    ("Problem",            "manifestsAs",            "Situation"),
    ("Goal",               "targetsProblem",         "Problem"),
    # Homework targets (multi-object — create one entry per valid object)
    ("Homework",           "targets",                "Problem"),
    ("Homework",           "targets",                "AutomaticThought"),
    ("Homework",           "targets",                "IntermediateBelief"),
    ("Homework",           "targets",                "CoreBelief"),
    # Cross-layer hinge
    ("AutomaticThought",   "associatedWith",         "Problem"),
    ("Intervention",       "appliedTo",              "AutomaticThought"),
    ("Intervention",       "appliedTo",              "IntermediateBelief"),
    ("Intervention",       "appliedTo",              "CoreBelief"),
    ("Intervention",       "appliedTo",              "Problem"),
    ("Intervention",       "produces",               "AdaptiveResponse"),
]

# Per-class: which predicates this class can emit (subject side).
# Used by extraction pipeline step D (edge resolution).
CBT_SUBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in CBT_EDGE_MAP:
    CBT_SUBJECT_EDGES.setdefault(_subj, []).append((_pred, _obj))

# Per-class: which predicates can point AT this class (object side).
CBT_OBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in CBT_EDGE_MAP:
    CBT_OBJECT_EDGES.setdefault(_obj, []).append((_pred, _subj))
```

**Important**: `CBT_NODE_CLASSES`, `CBT_EDGE_MAP`, `CBT_SUBJECT_EDGES`, `CBT_OBJECT_EDGES` are imported only by `factory.py` and `graph.py` (via factory). Do not import them directly in `orchestrator.py`, `extract.py`, or `api.py` — that would break the dependency rule. Pass them through `Schema` protocol methods instead:

Add to `Schema` Protocol in `interfaces.py`:

```python
def node_classes(self) -> list[dict]: ...
    """Return CBT_NODE_CLASSES."""

def edge_map(self) -> list[tuple[str, str, str]]: ...
    """Return CBT_EDGE_MAP."""

def subject_edges(self) -> dict[str, list[tuple[str, str]]]: ...
    """Return CBT_SUBJECT_EDGES."""
```

Add implementations to `CBTSchema`:

```python
def node_classes(self) -> list[dict]:
    from schema import CBT_NODE_CLASSES
    return CBT_NODE_CLASSES

def edge_map(self) -> list[tuple[str, str, str]]:
    from schema import CBT_EDGE_MAP
    return CBT_EDGE_MAP

def subject_edges(self) -> dict[str, list[tuple[str, str]]]:
    from schema import CBT_SUBJECT_EDGES
    return CBT_SUBJECT_EDGES
```

Wait — `CBTSchema` is in `schema.py`, so it imports from itself. Use:
```python
def node_classes(self): return CBT_NODE_CLASSES   # same file
def edge_map(self):     return CBT_EDGE_MAP
def subject_edges(self): return CBT_SUBJECT_EDGES
```

---

### 3.3 `graph.py` — Implement rich node/edge graph

#### 3.3.1 `InMemoryGraphStore` — additions

The existing flat `_state` dict (the slot-filler) is **kept as-is** — it still drives `missing()`, `acquired_summary()`, `cbt_context()`, `apply_session_state()`, and `snapshot()`. These are unchanged and the existing tests must continue to pass.

Add a second data structure alongside it for the rich graph:

```python
def __init__(self, schema: Schema):
    # --- existing ---
    self._schema = schema
    self._fields_by_priority = sorted(schema.fields(), key=lambda f: f.priority)
    self._state: dict[str, dict] = {}
    # --- new ---
    self._nodes: dict[str, GraphNode] = {}   # node_id -> GraphNode
    self._edges: dict[str, GraphEdge] = {}   # edge_id -> GraphEdge
    self._label_counters: dict[str, int] = {}  # "Situation" -> 2 (next free index)
    self.reset()
```

Update `reset()` to also initialize the rich graph:

```python
def reset(self) -> None:
    # existing flat state reset
    self._state = {
        f.key: {"value": None, "acquired": False, "turns": []}
        for f in self._fields_by_priority
    }
    # rich graph reset
    self._nodes = {}
    self._edges = {}
    self._label_counters = {}
    # pre-create one placeholder node per class
    for cls in self._schema.node_classes():
        label = cls["label"]
        nid = self._new_node_id(label)
        self._nodes[nid] = GraphNode(
            node_id=nid, label=label, status="missing", props={}, turn_acquired=None
        )
    # pre-create placeholder edges
    for subj_label, predicate, obj_label in self._schema.edge_map():
        # find the placeholder nodes (first one of each label)
        subj = self._first_node(subj_label)
        obj  = self._first_node(obj_label)
        if subj and obj:
            eid = f"{subj.node_id}__{predicate}__{obj.node_id}"
            self._edges[eid] = GraphEdge(
                edge_id=eid, predicate=predicate,
                subject_id=subj.node_id, object_id=obj.node_id,
                status="missing", turn_acquired=None
            )
```

Helper methods (private, not part of Protocol):

```python
def _new_node_id(self, label: str) -> str:
    n = self._label_counters.get(label, 0) + 1
    self._label_counters[label] = n
    return f"{label}_{n}"

def _first_node(self, label: str) -> GraphNode | None:
    """Return the first node of the given label (placeholder or found)."""
    for node in self._nodes.values():
        if node.label == label:
            return node
    return None

def _found_nodes(self, label: str) -> list[GraphNode]:
    return [n for n in self._nodes.values() if n.label == label and n.status == "found"]

def _similar(self, props_a: dict, props_b: dict) -> bool:
    """
    Simple content similarity check.
    Two nodes are 'similar' if their primary text field overlaps > 60%
    using word-set overlap (Jaccard). Falls back to exact match if no text.
    """
    def words(p: dict) -> set:
        text = p.get("content") or p.get("description") or p.get("statement") or ""
        return set(text.lower().split())
    a, b = words(props_a), words(props_b)
    if not a or not b:
        return props_a == props_b
    return len(a & b) / len(a | b) > 0.6
```

Implement the new Protocol methods:

```python
def nodes(self) -> list[GraphNode]:
    return list(self._nodes.values())

def edges(self) -> list[GraphEdge]:
    return list(self._edges.values())

def upsert_node(self, label: str, props: dict, turn_id: int) -> GraphNode:
    # Check if any existing found node is similar (merge candidate)
    for node in self._found_nodes(label):
        if self._similar(node.props, props):
            node.props.update(props)   # update in place (write on top)
            node.turn_acquired = turn_id
            return node

    # Check if there's a placeholder to upgrade
    for node in self._nodes.values():
        if node.label == label and node.status == "missing":
            node.status = "found"
            node.props = props
            node.turn_acquired = turn_id
            return node

    # No placeholder left — create a new found node (multi-instance)
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
```

#### 3.3.2 `Neo4jGraphStore` — additions

Add the same four Protocol methods with Cypher implementations.

`reset()` — add after existing node creation:
```python
# Pre-create placeholder class nodes (one per class)
for cls in self._schema.node_classes():
    label = cls["label"]
    session.run(
        "MATCH (s:Session {id: $sid}) "
        "MERGE (s)-[:HAS_CLASS_NODE]->(n:ClassNode {label: $label, node_id: $nid, status: 'missing'})",
        sid=self._session_id, label=label, nid=f"{label}_1"
    )
# Pre-create placeholder edges
for subj_label, pred, obj_label in self._schema.edge_map():
    session.run(
        """
        MATCH (s:Session {id: $sid})
        MATCH (s)-[:HAS_CLASS_NODE]->(a:ClassNode {label: $sl})
        MATCH (s)-[:HAS_CLASS_NODE]->(b:ClassNode {label: $ol})
        MERGE (a)-[e:PLACEHOLDER_EDGE {predicate: $pred}]->(b)
        SET e.status = 'missing'
        """,
        sid=self._session_id, sl=subj_label, pred=pred, ol=obj_label
    )
```

`nodes()`:
```python
def nodes(self) -> list[GraphNode]:
    with self._driver.session() as s:
        result = s.run(
            "MATCH (sess:Session {id: $sid})-[:HAS_CLASS_NODE]->(n:ClassNode) "
            "RETURN n.node_id AS nid, n.label AS label, n.status AS status, "
            "n.props AS props, n.turn_acquired AS turn_acquired",
            sid=self._session_id
        )
        nodes = []
        for r in result:
            nodes.append(GraphNode(
                node_id=r["nid"], label=r["label"], status=r["status"],
                props=r["props"] or {}, turn_acquired=r["turn_acquired"]
            ))
        return nodes
```

`edges()`: similar pattern, query `PLACEHOLDER_EDGE` relationships.

`upsert_node()`: Cypher MERGE on `node_id`, SET properties, SET status='found'.

`resolve_edge()`: MATCH existing PLACEHOLDER_EDGE, SET status='found'; or MERGE new.

Note: Neo4j stores `props` as a flat map. Store as `apoc.convert.toJson(props)` string if APOC is available, or flatten props into individual properties with a `prop_` prefix. **Simpler**: store `content`, `domain`, `subtype`, `channel` etc. as direct node properties — the set is small and known.

---

### 3.4 `extract.py` — Replace `LocalLLMExtractor` with `CBTExtractor`

Keep `StubExtractor` unchanged — tests depend on it.

Replace `LocalLLMExtractor` with a new class `CBTExtractor` that implements the 4-step per-turn pipeline. The `Extractor` Protocol signature stays: `extract(message, schema_text) -> dict[str, str]`. But internally this class does much more. The dict it returns is still the flat `{field_key: value}` deltas for `apply_deltas()` — the rich graph updates happen via the new graph methods called from `orchestrator.py`.

```python
class CBTExtractor:
    """
    4-step per-turn extraction:
      A. Extract + atomize (one Ollama call, returns list of node candidates)
      B. Normalize (string similarity check; optional LLM confirm if ambiguous)
      C. Classify to flat schema fields (maps node candidates to OntologyField keys)
      D. Edge resolution (called externally from orchestrator after nodes are applied)
    
    extract() returns flat deltas for backward compatibility with apply_deltas().
    extract_nodes() returns structured node candidates for the rich graph.
    resolve_edges() is called separately after nodes are written.
    """

    def __init__(self, model: str, host: str):
        self._model = model
        self._host = host.rstrip("/")

    def _ollama_generate(self, prompt: str) -> str:
        import requests, json
        resp = requests.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt,
                  "stream": False, "format": "json",
                  "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_json(self, raw: str) -> dict | list:
        import json, re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # ── Step A+B: Extract and atomize ────────────────────────────────────────

    def extract_nodes(self, message: str, window: list[tuple[str, str]],
                      schema_text: str) -> list[dict]:
        """
        Returns a list of extracted node candidates:
        [{"label": "Situation", "props": {"description": "...", "kind": "externalSituation"}}, ...]
        
        Uses ±2 exchange window for context. Therapist turns = context only.
        Client turns = extraction source.
        """
        from prompts import CBT_EXTRACTION_PROMPT
        window_text = "\n".join(
            f"{'Therapist' if role == 'therapist' else 'Client'}: {text}"
            for role, text in window
        )
        prompt = CBT_EXTRACTION_PROMPT.format(
            ontology_schema=schema_text,
            window=window_text,
            message=message,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "nodes" in result:
            return result["nodes"]
        return []

    # ── Step C: Map to flat schema fields ────────────────────────────────────

    def extract(self, message: str, schema_text: str) -> dict[str, str]:
        """
        Backward-compatible flat extraction for apply_deltas().
        Uses the existing CBT_EXTRACTION_PROMPT (single message, no window).
        Returns {field_key: value} dict.
        """
        from prompts import CBT_EXTRACTION_PROMPT
        prompt = CBT_EXTRACTION_PROMPT.format(
            ontology_schema=schema_text,
            window="",
            message=message,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        if not isinstance(result, dict):
            return {}
        # Drop unknown keys
        known = set(schema_text.split())  # crude; schema_text contains field keys
        return {k: str(v) for k, v in result.items() if isinstance(v, str) and v}

    # ── Step D: Edge resolution ───────────────────────────────────────────────

    def resolve_edges(
        self,
        new_node: "GraphNode",  # from interfaces
        existing_nodes: "list[GraphNode]",
        window_text: str,
        subject_edges: dict,   # schema.subject_edges()
    ) -> list[tuple[str, str, str]]:
        """
        For a newly found node, check which edges it could emit or receive.
        Returns list of (subject_node_id, predicate, object_node_id) tuples
        that the LLM confirms as present in the conversation window.
        
        Only checks edges where the OTHER endpoint is already status='found'.
        """
        from prompts import CBT_EDGE_RESOLUTION_PROMPT

        found_by_label: dict[str, list] = {}
        for n in existing_nodes:
            if n.status == "found" and n.node_id != new_node.node_id:
                found_by_label.setdefault(n.label, []).append(n)

        candidates = []
        # Edges where new_node is subject
        for pred, obj_label in subject_edges.get(new_node.label, []):
            for obj_node in found_by_label.get(obj_label, []):
                candidates.append((new_node.node_id, pred, obj_node.node_id,
                                    new_node, obj_node))
        # Edges where new_node is object — check reverse map
        for obj_label, pairs in found_by_label.items():
            # Check if any found node has new_node.label as a target
            pass  # simplified: object-side check omitted for v6, add in v7

        if not candidates:
            return []

        # Batch all candidates into one LLM call
        candidate_lines = "\n".join(
            f"{i+1}. {c[3].label}({c[3].props.get('content','')[:40]!r}) "
            f"--[{c[1]}]--> {c[4].label}({c[4].props.get('content','')[:40]!r})"
            for i, c in enumerate(candidates)
        )
        prompt = CBT_EDGE_RESOLUTION_PROMPT.format(
            window=window_text,
            candidates=candidate_lines,
        )
        raw = self._ollama_generate(prompt)
        result = self._parse_json(raw)
        confirmed_indices = result.get("confirmed", []) if isinstance(result, dict) else []
        return [
            (candidates[i-1][0], candidates[i-1][1], candidates[i-1][2])
            for i in confirmed_indices
            if isinstance(i, int) and 1 <= i <= len(candidates)
        ]
```

Update `factory.py` to import `CBTExtractor` instead of `LocalLLMExtractor`:

```python
# in factory.py make_extractor():
from extract import StubExtractor, CBTExtractor

def make_extractor() -> Extractor:
    kind = os.environ.get("EXTRACTOR", "local")
    if kind == "local":
        return CBTExtractor(
            model=os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    return StubExtractor()
```

---

### 3.5 `prompts.py` — Add two new prompts

Keep `CBT_SYSTEM_PROMPT` and `CBT_EXTRACTION_PROMPT` exactly as they are.

Add `window` as a new format field to `CBT_EXTRACTION_PROMPT` — insert `{window}` before the `{message}` block:

```
...
Recent conversation (±2 exchanges, therapist = context only):
{window}

Current client message to extract from:
{message}
...
```

If `window` is empty string, the section reads cleanly as blank — no conditional logic needed.

Add new `CBT_EDGE_RESOLUTION_PROMPT`:

```python
CBT_EDGE_RESOLUTION_PROMPT = """You are verifying which relationships exist between CBT concepts
in a therapy conversation.

CONVERSATION WINDOW:
{window}

CANDIDATE RELATIONSHIPS (numbered):
{candidates}

For each candidate, decide if the conversation window supports this directional relationship.
Only confirm if the window clearly shows the connection — do not infer beyond what is stated.

Respond ONLY with a JSON object:
{{"confirmed": [1, 3, ...]}}   ← list of confirmed candidate numbers, empty list if none.
"""
```

---

### 3.6 `orchestrator.py` — Async turn loop with optimistic extraction

This is the most significant change. The file becomes async. All existing logic is preserved; the new code wraps around it.

```python
"""The turn loop. Depends only on interfaces.py — never on concrete implementations."""

import asyncio
import os
from dataclasses import dataclass, field

from interfaces import Extractor, GraphStore, Generator, Schema, GraphNode
from prompts import CBT_SYSTEM_PROMPT

PHASE_ORDER = ["Rapport", "Exploration", "Technique", "Consolidation"]

PHASE_MINIMUMS: dict[str, dict] = {
    "Exploration":   {"fields": ["presenting_problem"],                     "min_turns": 2},
    "Technique":     {"fields": ["presenting_problem", "negative_thought"],  "min_turns": 5},
    "Consolidation": {"fields": ["negative_thought", "cognitive_pattern"],   "min_turns": 12},
}

# How long (seconds) to wait for extraction before falling back to async mode.
# Set via EXTRACTION_TIMEOUT env var. Default 8s.
EXTRACTION_TIMEOUT = float(os.environ.get("EXTRACTION_TIMEOUT", "8"))


def validate_phase(proposed: str, current: str, snapshot: dict, turn_count: int) -> str:
    """Unchanged from current codebase."""
    try:
        if PHASE_ORDER.index(proposed) <= PHASE_ORDER.index(current):
            return proposed
    except ValueError:
        return current
    mins = PHASE_MINIMUMS.get(proposed, {})
    fields_met = all(snapshot.get(f, {}).get("acquired") for f in mins.get("fields", []))
    turns_met = turn_count >= mins.get("min_turns", 0)
    return proposed if (fields_met and turns_met) else current


@dataclass
class Session:
    schema: Schema
    graph: GraphStore
    extractor: Extractor
    generator: Generator
    history: list[tuple[str, str]] = field(default_factory=list)
    turn_count: int = 0
    extraction_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # NOTE: extraction_lock is not picklable. Sessions are process-local only.


# ── Sync wrapper (keeps existing tests passing) ───────────────────────────────

def turn(session: Session, user_message: str) -> dict:
    """
    Sync entry point. Used by existing tests and by api.py's sync route handler.
    Runs the async turn in a new event loop if none is running, otherwise
    schedules it. In practice FastAPI's async handler calls async_turn directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context (FastAPI) — should not block.
        # This branch is a safety fallback; prefer calling async_turn directly.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, async_turn(session, user_message))
            return future.result()
    else:
        return asyncio.run(async_turn(session, user_message))


# ── Async core ────────────────────────────────────────────────────────────────

async def async_turn(session: Session, user_message: str) -> dict:
    """
    Main turn logic. Called directly by async FastAPI route.
    
    Dual-mode extraction:
      Option 1 (sync-opportunistic): extraction finishes within EXTRACTION_TIMEOUT →
        graph is updated before generate completes → snapshot reflects new state.
      Option 2 (async fallback): extraction takes longer → generate uses pre-turn
        graph → extraction continues in background → graph updates eventually.
    """
    session.turn_count += 1

    # Build ±2 exchange window for extractor
    window = _build_window(session.history, n=2)

    # Snapshot graph state BEFORE this turn (used by generate regardless of mode)
    pre_turn_context = session.graph.cbt_context()
    pre_turn_snapshot = session.graph.snapshot()
    current_phase = (pre_turn_snapshot.get("session_phase") or {}).get("value") or "Rapport"

    # History summary for prompt (last 10 complete turns)
    completed = [(u, a) for u, a in session.history if a]
    history_summary = "\n".join(
        f"Client: {u}\nTherapist: {a}" for u, a in completed[-10:]
    ) or "(session just started)"

    system_prompt = CBT_SYSTEM_PROMPT.format(
        cbt_context=pre_turn_context,
        history_summary=history_summary,
    )

    # Append user message to history (assistant slot empty until generate fills it)
    session.history.append((user_message, ""))

    # ── Launch both tasks concurrently ────────────────────────────────────────
    schema_text = session.schema.render()

    extraction_task = asyncio.create_task(
        _run_extraction(session, user_message, window, schema_text)
    )
    generate_task = asyncio.create_task(
        _run_generate(session.generator, system_prompt, session.history)
    )

    # ── Wait for generate (must complete for reply) ────────────────────────────
    result = await generate_task

    reply     = result.get("response", "")
    technique = result.get("technique", "Rapport Building")
    proposed  = result.get("phase", "Rapport")

    session.history[-1] = (user_message, reply)

    # ── Check if extraction finished in time (Option 1) ────────────────────────
    extraction_mode = "async"
    if extraction_task.done():
        extraction_mode = "sync"
        # Extraction already wrote to graph; re-read snapshot for response
        deltas = extraction_task.result() or {}
    else:
        # Option 2: extraction still running — don't cancel it
        deltas = {}
        # extraction_task continues in background; no further await needed

    # ── Validate and persist session state ─────────────────────────────────────
    validated_phase = validate_phase(proposed, current_phase,
                                     session.graph.snapshot(), session.turn_count)
    session.graph.apply_session_state(validated_phase, technique)

    return {
        "reply": reply,
        "technique": technique,
        "phase": validated_phase,
        "deltas": deltas,
        "slots": session.graph.snapshot(),
        "extraction_mode": extraction_mode,   # "sync" | "async" — useful for debugging
    }


async def _run_generate(generator: Generator, system: str,
                        history: list[tuple[str, str]]) -> dict:
    """Wrap the synchronous generator.generate() call for asyncio."""
    return await asyncio.to_thread(generator.generate, system, history)


async def _run_extraction(session: Session, message: str,
                          window: list[tuple[str, str]], schema_text: str) -> dict:
    """
    The full 4-step extraction pipeline. Runs under the per-session lock
    so concurrent turns don't write to the graph simultaneously.
    Returns the flat deltas dict (for the response payload).
    """
    async with session.extraction_lock:
        # Step A+B: extract + atomize (sync Ollama call, run in thread)
        node_candidates = await asyncio.to_thread(
            session.extractor.extract_nodes, message, window, schema_text
        )

        # Step C: flat extraction for apply_deltas (backward compat)
        flat_deltas = await asyncio.to_thread(
            session.extractor.extract, message, schema_text
        )
        session.graph.apply_deltas(flat_deltas, session.turn_count)

        # Step D: upsert rich nodes + resolve edges
        window_text = "\n".join(
            f"{'Therapist' if r == 'therapist' else 'Client'}: {t}"
            for r, t in window
        )
        subject_edges = session.schema.subject_edges()

        for candidate in node_candidates:
            label = candidate.get("label", "")
            props = candidate.get("props", {})
            if not label or not props:
                continue

            # Upsert into rich graph
            new_node = session.graph.upsert_node(label, props, session.turn_count)

            # Resolve edges for this node
            resolved = await asyncio.to_thread(
                session.extractor.resolve_edges,
                new_node,
                session.graph.nodes(),
                window_text,
                subject_edges,
            )
            for subj_id, pred, obj_id in resolved:
                session.graph.resolve_edge(subj_id, pred, obj_id, session.turn_count)

        return flat_deltas


def _build_window(history: list[tuple[str, str]], n: int = 2) -> list[tuple[str, str]]:
    """Return last n complete exchanges as [(role, text), ...] pairs."""
    completed = [(u, a) for u, a in history if a]
    recent = completed[-n:]
    window = []
    for user, assistant in recent:
        window.append(("client", user))
        window.append(("therapist", assistant))
    return window
```

---

### 3.7 `api.py` — Convert `/chat` to async, enrich `/graph` endpoint

Change the `/chat` route to async and call `async_turn` directly:

```python
# Replace:
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session = _get_or_create(request.session_id)
    result = turn(session, request.message)
    return ChatResponse(**result)

# With:
from orchestrator import Session, async_turn   # add async_turn to import

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session = _get_or_create(request.session_id)
    result = await async_turn(session, request.message)
    return ChatResponse(**result)
```

Update `ChatResponse` to include `extraction_mode`:

```python
class ChatResponse(BaseModel):
    reply: str
    technique: str = ""
    phase: str = ""
    deltas: dict[str, str]
    slots: dict
    extraction_mode: str = "async"   # new field — "sync" | "async"
```

Update `/graph/{session_id}` to return rich node/edge data:

```python
@app.get("/graph/{session_id}")
def get_graph(session_id: str) -> dict:
    if session_id not in _sessions:
        return {"nodes": [], "edges": []}

    graph = _sessions[session_id].graph
    
    nodes = []
    edges = []

    for node in graph.nodes():
        ntype = _node_type(node.label)
        label = (
            f"{node.label}\n{str(node.props.get('content') or node.props.get('description') or '')[:25]}"
            if node.status == "found" else node.label
        )
        nodes.append({"data": {
            "id": node.node_id,
            "label": label,
            "type": ntype if node.status == "found" else "missing",
            "status": node.status,
        }})

    for edge in graph.edges():
        edges.append({"data": {
            "source": edge.subject_id,
            "target": edge.object_id,
            "label": edge.predicate if edge.status == "found" else "",
            "status": edge.status,
        }})

    return {"nodes": nodes, "edges": edges}


_LABEL_TYPE_MAP = {
    "Session": "session", "Client": "session",
    "Problem": "session_structure", "Goal": "session_structure",
    "Intervention": "session_structure", "Homework": "session_structure",
    "CoreBelief": "cognitive", "IntermediateBelief": "cognitive",
    "Situation": "cognitive", "AutomaticThought": "cognitive",
    "Reaction": "cognitive", "AdaptiveResponse": "cognitive",
}

def _node_type(label: str) -> str:
    return _LABEL_TYPE_MAP.get(label, "field")
```

---

### 3.8 `ui.py` — Update Cytoscape styles for rich node types

Update `NODE_STYLES` in `ui.py` to add colors for the new node types returned by `/graph`:

```python
NODE_STYLES = json.dumps([
    # existing
    {"selector": 'node[type="session"]',
     "style": {"background-color": "#2d6a4f", "color": "#fff",
                "label": "data(label)", "text-wrap": "wrap",
                "text-valign": "center", "font-size": "11px",
                "width": 80, "height": 80, "shape": "ellipse"}},
    # new: session structure (Problem, Goal, Intervention, Homework)
    {"selector": 'node[type="session_structure"]',
     "style": {"background-color": "#74c69d", "label": "data(label)",
                "text-wrap": "wrap", "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "round-rectangle"}},
    # new: cognitive model nodes
    {"selector": 'node[type="cognitive"]',
     "style": {"background-color": "#9b72cf", "color": "#fff",
                "label": "data(label)", "text-wrap": "wrap",
                "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "ellipse"}},
    # existing session_state (phase/technique — keep)
    {"selector": 'node[type="session_state"]',
     "style": {"background-color": "#f4a261", "label": "data(label)",
                "text-wrap": "wrap", "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "diamond"}},
    # missing nodes — grey dashed
    {"selector": 'node[type="missing"]',
     "style": {"background-color": "#dee2e6", "label": "data(label)",
                "text-valign": "center", "font-size": "9px",
                "width": 50, "height": 50,
                "border-style": "dashed", "border-color": "#adb5bd", "border-width": 2}},
    # edges
    {"selector": 'edge[status="found"]',
     "style": {"label": "data(label)", "font-size": "8px", "curve-style": "bezier",
                "target-arrow-shape": "triangle", "line-color": "#74c69d",
                "target-arrow-color": "#74c69d", "arrow-scale": 0.7}},
    {"selector": 'edge[status="missing"]',
     "style": {"curve-style": "bezier", "target-arrow-shape": "triangle",
                "line-color": "#dee2e6", "target-arrow-color": "#dee2e6",
                "line-style": "dashed", "arrow-scale": 0.5}},
])
```

No other changes to `ui.py`.

---

### 3.9 `.env.example` — Add `EXTRACTION_TIMEOUT`

```bash
# Extraction pipeline timeout in seconds.
# If extraction finishes within this window, graph is updated before reply (Option 1).
# If not, extraction continues in background after reply is sent (Option 2).
EXTRACTION_TIMEOUT=8
```

Also fix the model name inconsistency:
```bash
OLLAMA_MODEL=qwen3.5-nothink
LOCAL_LLM_MODEL=qwen3.5-nothink
```

---

### 3.10 `tests/test_orchestrator.py` — Add tests for new behaviour

Keep all existing tests unchanged. Add:

```python
def test_async_turn_returns_expected_keys():
    """async_turn returns all expected keys including extraction_mode."""
    from orchestrator import async_turn
    session = _make_session()
    result = asyncio.run(async_turn(session, "presenting_problem: work stress"))
    assert "reply" in result
    assert "phase" in result
    assert "technique" in result
    assert "extraction_mode" in result
    assert result["extraction_mode"] in ("sync", "async")


def test_rich_graph_has_placeholder_nodes():
    """At session start, all CBT node classes exist as placeholder nodes."""
    from schema import CBT_NODE_CLASSES
    session = _make_session()
    nodes = session.graph.nodes()
    node_labels = {n.label for n in nodes}
    for cls in CBT_NODE_CLASSES:
        assert cls["label"] in node_labels, f"Missing placeholder for {cls['label']}"


def test_rich_graph_upsert_creates_found_node():
    """upsert_node upgrades a placeholder to found."""
    session = _make_session()
    node = session.graph.upsert_node(
        "Situation",
        {"description": "exam tomorrow", "kind": "externalSituation"},
        turn_id=1
    )
    assert node.status == "found"
    assert node.label == "Situation"
    # Placeholder should be gone or upgraded
    placeholders = [n for n in session.graph.nodes()
                    if n.label == "Situation" and n.status == "missing"]
    # After first upsert the placeholder was upgraded — no new placeholder
    assert len(placeholders) == 0


def test_rich_graph_multi_instance():
    """Second upsert of different content creates a new node."""
    session = _make_session()
    n1 = session.graph.upsert_node("Situation", {"description": "exam tomorrow"}, turn_id=1)
    n2 = session.graph.upsert_node("Situation", {"description": "fight with friend"}, turn_id=2)
    assert n1.node_id != n2.node_id
    found = [n for n in session.graph.nodes()
             if n.label == "Situation" and n.status == "found"]
    assert len(found) == 2


def test_resolve_edge_marks_found():
    """resolve_edge updates edge status to found."""
    session = _make_session()
    sit  = session.graph.upsert_node("Situation", {"description": "exam"}, 1)
    at   = session.graph.upsert_node("AutomaticThought", {"content": "I will fail"}, 1)
    edge = session.graph.resolve_edge(sit.node_id, "triggers", at.node_id, 1)
    assert edge.status == "found"
    found_edges = [e for e in session.graph.edges() if e.status == "found"]
    assert any(e.predicate == "triggers" for e in found_edges)


def test_extraction_lock_exists():
    """Session has an extraction lock."""
    session = _make_session()
    assert isinstance(session.extraction_lock, asyncio.Lock)
```

---

## 4. What Does NOT Change

- `interfaces.py` existing Protocol methods (only additions)
- `schema.py` `CBTSchema.fields()` and `CBTSchema.render()` (existing flat field logic)
- `graph.py` existing methods: `apply_deltas()`, `missing()`, `acquired_summary()`, `snapshot()`, `cbt_context()`, `apply_session_state()` — all unchanged
- `generate.py` — no changes at all
- `orchestrator.py` `validate_phase()` logic — unchanged
- `api.py` `/reset` endpoint — unchanged
- `ui.py` chat layout, phase/technique display, iframe rendering — unchanged
- The dependency rule: only `factory.py` imports concrete classes
- Ollama native `/api/chat` for generator, `/api/generate` for extractor
- `StubExtractor` — unchanged, tests depend on it

---

## 5. Implementation Order for Claude Code

Do these in order — each step is independently testable:

1. `interfaces.py` — add `GraphNode`, `GraphEdge` dataclasses + new Protocol methods + new `Schema` methods
2. `schema.py` — add `CBT_NODE_CLASSES`, `CBT_EDGE_MAP`, `CBT_SUBJECT_EDGES`, `CBT_OBJECT_EDGES`, implement new `CBTSchema` methods
3. `graph.py` — implement new Protocol methods on `InMemoryGraphStore` first, `Neo4jGraphStore` second
4. `prompts.py` — add `window` field to `CBT_EXTRACTION_PROMPT`, add `CBT_EDGE_RESOLUTION_PROMPT`
5. `extract.py` — add `CBTExtractor` alongside existing `StubExtractor` (keep `LocalLLMExtractor` if it exists, or remove if replaced)
6. `orchestrator.py` — replace with async version, keep `turn()` sync wrapper
7. `factory.py` — update `make_extractor()` to use `CBTExtractor`
8. `api.py` — make `/chat` async, enrich `/graph` response, add `extraction_mode` to response model
9. `ui.py` — update `NODE_STYLES` only
10. `.env.example` — add `EXTRACTION_TIMEOUT`, fix model names
11. `tests/` — add new tests, verify all existing tests still pass

After step 3: run `pytest` — all existing tests must pass before continuing.  
After step 6: run `pytest` — all tests including new async ones must pass.  
After step 11: run `pytest` — full suite green.

---

## 6. Deferred (Do Not Implement Now)

- Object-side edge resolution in `CBTExtractor.resolve_edges()` (the comment in the code marks where it goes)
- `Neo4jGraphStore` implementation of `nodes()`, `edges()`, `upsert_node()`, `resolve_edge()` — implement `InMemoryGraphStore` versions first, Neo4j second
- Option B typed node graph
- `"think": false` performance tuning
- WebSocket push for graph updates
- Session export to JSON

---

## 7. Running

```bash
# Prerequisites
ollama pull qwen3.5-nothink
ollama serve

# Optional Neo4j
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/changeme neo4j:5

# App
cp .env.example .env
pip install -r requirements.txt
uvicorn api:app --reload

# Tests
pytest
```

Open `http://localhost:8000/` — chat on the left, live knowledge graph on the right.  
Graph starts as all-grey placeholder nodes. Nodes turn coloured as the session fills them.  
Edges appear as solid lines once both endpoints are found.
