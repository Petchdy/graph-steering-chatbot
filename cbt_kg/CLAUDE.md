# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The runtime code lives in the `cbt_kg/` package at the repo root; this file
sits inside it (`cbt_kg/CLAUDE.md`). All commands below are run from the repo
root unless stated otherwise.

## Setup

```bash
# From repo root — one-time venv creation
python -m venv cbt_kg/chat-bot-env
source cbt_kg/chat-bot-env/bin/activate
pip install -r cbt_kg/requirements.txt
cp cbt_kg/.env.example cbt_kg/.env
```

Activate the venv (`source cbt_kg/chat-bot-env/bin/activate`) before running
any of the commands below.

## Commands

```bash
uvicorn cbt_kg.api:app --reload       # Gradio UI at / · FastAPI routes below
pytest                                # all tests; uses stub + echo (no Ollama / Neo4j needed)
pytest cbt_kg/tests/test_therapy.py::test_async_turn_returns_expected_keys   # single test
```

Offline (no Ollama): tests already default to `EXTRACTOR=stub GENERATOR=echo`
(see `cbt_kg/conftest.py`). For manual runs, set those in `cbt_kg/.env` to
bypass Ollama.

### Ollama model

The default model tag is `qwen3.5-nothink`. If it is not installed, either
pull it (`ollama pull qwen3.5-nothink`) or point `.env` at any nothink variant
you already have:

```
OLLAMA_MODEL=qwen3.5-27b-nothink:latest
LOCAL_LLM_MODEL=qwen3.5-27b-nothink:latest
```

`OLLAMA_MODEL` drives the extractor (`TurnPipeline`); `LOCAL_LLM_MODEL` drives
the generator (`LocalLLMGenerator`). Both default to `qwen3.5-nothink` and can
be set independently.

## Environment variables (`cbt_kg/.env`)

| Var | Values | Default |
|-----|--------|---------|
| `GRAPH_BACKEND` | `memory` \| `neo4j` | `memory` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | — | — |
| `EXTRACTOR` | `local` \| `stub` | `local` |
| `GENERATOR` | `local` \| `openrouter` \| `echo` \| `steered` | `local` |
| `STEER_URL` / `STEER_DEFAULT_STRATEGY` / `STEER_NO_OLLAMA` | steering overlay (`GENERATOR=steered`) — see root `CLAUDE.md` + `steering/NOTES.md` | `http://localhost:8100` / `none` / `0` |
| `OLLAMA_MODEL` | any Ollama tag | `qwen3.5-nothink` |
| `OLLAMA_HOST` | URL | `http://localhost:11434` |
| `LOCAL_LLM_MODEL` | any Ollama tag | falls back to `OLLAMA_MODEL` |
| `LOCAL_LLM_BASE_URL` | URL (`/v1` suffix stripped internally) | `http://localhost:11434/v1` |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | — | `anthropic/claude-sonnet-4-6` |
| `EXTRACT_LANGUAGE` | `English` \| `Thai` | `English` |
| `EXTRACT_FAST` | `0` \| `1` | `0` — set `1` to collapse Stage 2.5 into Stage 1 for fewer LLM calls (lower fidelity) |
| `CONSOLIDATE_EVERY` | integer | `6` — Tier B fires every Nth client turn as a detached background task |
| `EXTRACTION_TIMEOUT` | seconds | `8` — defined for reference; extraction is always awaited (no enforced cutoff in the current code) |

## Layout (PRD_cbt_v7.md §2)

```
graph-chatbot/             repo root
├── cbt_kg/                ← the package (all runtime code)
│   ├── ontology.py · interfaces.py · factory.py
│   ├── graph_memory.py · graph_neo4j.py · graph_reader.py
│   ├── extract.py · generate.py · prompts.py
│   ├── therapy.py · query.py
│   ├── api.py · ui.py
│   ├── tests/             (test_ontology / test_graph_memory / test_therapy / test_query)
│   ├── conftest.py · .env.example · requirements.txt · README.md · CLAUDE.md
├── V4_flat/               reference batch pipeline (ontology was ported from here)
├── PRD_cbt_v7.md          spec
└── pyproject.toml         pytest config — adds repo root to pythonpath
```

Inside `cbt_kg/`, every module uses **relative imports** (`from .ontology import …`).
Tests import via the package path (`from cbt_kg.ontology import …`). The repo
root is on `sys.path` via the `pyproject.toml` pytest config; uvicorn loads
the app as `cbt_kg.api:app`.

## Architecture (v7 — full V4_flat restructure)

Two parts, two Gradio tabs, one V4_flat ontology, one graph backend.

- **Part 1 — Therapy chatbot** (`therapy.py` + Tab 1). LLM plays therapist
  (CACTUS principles); user is the client. Every client turn runs the V4_flat
  per-turn extraction pipeline (`extract.TurnPipeline`, Tier A) against the
  client message — extract → atomize → property-classify → merge → local edges.
  Every `CONSOLIDATE_EVERY` turns, Tier B fires in a detached background task
  (session-level extract, reinforces wide-window, reframe sub-graph,
  deterministic structure).

- **Part 2 — Query chatbot** (`query.py` + Tab 2). Therapist user asks NL
  questions of any V4_flat-shaped graph: a live Part 1 session, a Stage 5 JSON
  export, or a Neo4j database. Three `GraphReader`s normalize the source into
  a canonical `list[GraphNode]` + `list[GraphEdge]`. The query engine does
  parse (LLM) → execute (deterministic Python) → answer (LLM), so retrieval is
  honest and the LLM only translates.

### The single source of truth

`ontology.py` (ported verbatim from `V4_flat/cbt_ontology_v4_flat.py`) defines
every CBT concept used anywhere: 13 node classes, property enums, glosses,
`CLASS_DEFINITIONS`, `ANCHOR_FAMILIES`, `EDGE_MAP`, `REL_TYPE`, `ID_PREFIX`,
`TEXT_PROP`, `CLASS_HIERARCHY`. **Nothing else.** The `CBTSchema` adapter at
the bottom of the file implements the `Schema` Protocol.

### The dependency rule (load-bearing)

`interfaces.py` defines five Protocols — `Schema`, `GraphStore`, `Extractor`,
`Generator`, `GraphReader` — plus `GraphNode` and `GraphEdge` dataclasses.
**Every module except `factory.py` imports only from `interfaces.py` and
`ontology.py`.** `factory.py` is the sole place that knows which class backs
each Protocol; env vars select at construction time.

### V4_flat extraction pipeline (extract.TurnPipeline)

Tier A (every client turn, background asyncio Task):

1. **EXTRACT** — V4_flat Stage 1 prompt (per-turn, ±2 context, speaker prior).
2. **ATOMIZE** — V4_flat Stage 1.2 for AutomaticThought / CoreBelief / IB only.
3. **PROPERTIES** — Stage 2.5 classifiers; discriminators first
   (`Problem.domain`, `CoreBelief.domain`, `IB.subtype`, `Reaction.channel`),
   then `distortionType`, `modality`, `Situation.kind`, `Intervention.technique`,
   `Homework.taskType`, `CoreBelief.category` (self-only), plus deterministic
   `Reaction.valence` + `Situation.temporality` from lexicons.
4. **MERGE** — string-Jaccard against existing nodes (handled inside
   `GraphStore.upsert_node`).
5. **EDGES (local)** — Stage 3 anchor prompt restricted to per-turn-safe
   predicates: `triggers`, `leadsTo`, `stemsFrom`, `manifestsAs`,
   `givesRiseTo`, `influencesPerceptionOf`, `associatedWith`. Skipped here
   (deferred to Tier B): `reinforces`, `hasAdaptiveResponse`, `produces`,
   `becomesSituation`.

Tier B (every `CONSOLIDATE_EVERY` turns, detached background task):

1. **SESSION-LEVEL** extract over the whole transcript so far
   (`CoreBelief`, `IntermediateBelief`, `Problem`, `Goal`, `Intervention`,
   `Homework`, `AdaptiveResponse`).
2. **REINFORCES** — wide-window `Reaction × CoreBelief`.
3. **REFRAME sub-graph** — `hasAdaptiveResponse` / `produces` / `appliedTo`.
4. **STRUCTURE** — deterministic `Client hasSession Session`,
   `Session hasProblem/hasIntervention/hasHomework`, `Goal targetsProblem`.

### Async turn loop (therapy.async_turn)

1. Snapshot pre-turn `cbt_context()` and `snapshot()`. Generator uses these.
2. Launch `_run_extraction` (Tier A) and `_run_generate` concurrently.
3. Await generate → `{response, technique, phase}`. Reply uses pre-turn state.
4. Await extraction. Per-session `asyncio.Lock` guards graph writes.
5. `validate_phase(...)` enforces node-class minimums from V4_flat:
   - Exploration requires `Problem` + 2 turns.
   - Technique requires `AutomaticThought` + `Situation` + 5 turns.
   - Consolidation requires `AdaptiveResponse` + 12 turns.
6. `apply_session_state(phase, technique)`.
7. If `turn_count % CONSOLIDATE_EVERY == 0`, fire Tier B as a detached task.
8. Return `{reply, technique, phase, extraction_mode, new_nodes, new_edges,
   graph_snapshot}`.

### Graph stores (graph_memory.py, graph_neo4j.py)

`InMemoryGraphStore.reset()` pre-creates one placeholder node per class plus
one placeholder edge per `(subj, pred, obj)` in the edge map. `upsert_node`
flips the first placeholder to `status='found'` on first match; further
instances create new found nodes. `add_edge` flips the matching placeholder
edge or creates a new one. Jaccard-based merging happens inside `upsert_node`.

`Neo4jGraphStore` uses the V4_flat `:ABox` model — one labeled node per
content class (label = `Problem`/`CoreBelief`/...), `primaryLabel` property,
direct property storage of `domain`/`subtype`/`channel`, `:TBox`
class-hierarchy, `EVIDENCED_BY → :Utterance`, typed edges via `REL_TYPE`.
This matches `cbt_stage5_persist_v4.write_neo4j` byte-for-byte.

### Part 2 — universal query

**Dialogue provenance.** `GraphStore.record_utterance(turn_index, speaker, text)`
writes each turn as an `Utterance` node (`text`/`speaker`/`turnIndex`), called from
`therapy.async_turn` for both speakers. Ids are deterministic via
`graph_memory.utterance_id`: the client's is the bare `utt_<turn>` that
`Neo4jGraphStore._ensure_utterance_locked` already creates as the `EVIDENCED_BY`
target (so recording MERGEs into it rather than forking a duplicate); the
therapist's is `utt_<turn>_t`.

This is what makes dialogue queryable. `node.evidence` / `edge.evidence` are bare
turn *indices*; `query.build_utterance_index` turns them back into words, and
`_node_view` / `_edge_view` attach an `evidence_quotes` list. Without utterances
in the graph the query engine can cite "turn 7" but never quote it —
`Session.transcript` is process-local and never reaches Part 2. Graphs recorded
before this existed yield an empty index and degrade to citation-only.

Two related traps: `execute` hides `Utterance` from *unlabelled* `list`/`describe`
(otherwise dialogue swamps the CBT content — ask for it by label, which
`QUERY_PARSE_PROMPT` now instructs), and `_normalize_filters` strips the `Class.`
qualifier off property-filter keys, because the parse prompt lists enums as
`Problem.domain` while props are stored bare — unnormalized, one hallucinated
filter silently empties the result set and reads as "not in the graph".

`GraphReader` Protocol has one method: `load() → (nodes, edges)`. Three
implementations all emit the same canonical shape — `LiveGraphReader`
(wraps a Part 1 `GraphStore`), `JsonGraphReader` (Stage 5 export),
`Neo4jGraphReader` (`:ABox` model, compatible with both Part 1 and the
V4_flat batch export). `QueryEngine.answer(question, nodes, edges)` does
parse → execute → answer; execute is deterministic Python so the LLM cannot
invent facts.

### The graph canvas (`ui.py` — the largest module, ~1.2k lines)

The visible graph is **not** Cytoscape. `graph_memory.cytoscape_render` /
`GraphStore.cytoscape()` exist only for the `/graph/*` API consumers; the UI
renders its own `<canvas>` + inspector panel and never calls them.

- `_build_canvas_data(nodes, edges)` → canvas dicts. It drops `Utterance`
  nodes and any edge whose status isn't `found` (placeholder edges are noise),
  and keeps nodes of both statuses so `missing` classes stay visible as the
  greyed-out "not yet discovered" scaffold.
- `_CANVAS_TEMPLATE` is a plain string, **not an f-string** — it contains raw
  CSS/JS braces. `_render_canvas` injects data via `__PLACEHOLDER__`
  substitution (`__NODES__`, `__EDGES__`, `__EDIT_MODE__`, colour maps,
  `__NODE_CLASSES__`, `__PREDICATES__`), then html-escapes the whole document
  into an `<iframe srcdoc="…">`. Adding a value to the canvas means adding
  both a placeholder and a `.replace()` — never an f-string interpolation.
- One template serves both tabs; `edit_mode` toggles the edit affordances.
- **Tab 2 edits never reach Python.** `saveNode` / `saveEdge` / `deleteNode` /
  the create-modal mutate the iframe's local JS arrays only — there is no
  postMessage or fetch back to the server, so any Gradio re-render discards
  them. The sanctioned round-trip is the in-canvas `saveJSON()` (exports the
  V4_flat Stage 5 shape, `found` items only) → re-upload via **Load JSON**.
- `_NODE_CLASSES` / `_PREDICATES` / `NODE_COLORS` are hand-maintained lists,
  not derived from `ontology.py`; extending the ontology means editing them too.
- `_render_canvas(hidden_nodes=…)` (`__HIDDEN_NODES__`) carries nodes that are
  excluded from the drawing but still written by `saveJSON` — currently the
  `Utterance` nodes, so a saved export can still quote dialogue when re-loaded.
  Anything filtered out of the canvas but needed on disk belongs here.

### FastAPI routes (`api.py`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Send a client turn; returns `reply`, `technique`, `phase`, `new_nodes`, `new_edges`, `graph_snapshot` |
| `GET` | `/strategies` | Steering strategies, proxied from the steering service; returns just `["none"]` when it is unreachable. It does **not** check `GENERATOR`, so it lists strategies whenever the service is up — even under `GENERATOR=local`, where picking one has no effect |
| `POST` | `/reset` | Wipe and reinitialise the session graph |
| `GET` | `/graph/{session_id}` | Cytoscape.js JSON of the live graph |
| `POST` | `/load_graph/live` | Load a live Part 1 session as a query source |
| `POST` | `/load_graph/json` | Load a Stage 5 JSON export as a query source |
| `POST` | `/load_graph/neo4j` | Connect a Neo4j database as a query source |
| `POST` | `/query` | NL question against a loaded query graph |
| `GET` | `/graph_preview/{handle}` | Cytoscape.js JSON of a loaded query graph |

The Gradio UI is mounted at `/` **after** all routes are defined; `/docs`
gives the full OpenAPI spec.

### Implementation notes worth knowing

- `LocalLLMGenerator` uses Ollama's **native `/api/chat`**, not `/v1`; always
  passes `"think": false`. `LOCAL_LLM_BASE_URL` accepts `/v1` for convenience
  but the suffix is stripped.
- `TurnPipeline` uses `/api/generate` with `format: "json"` and
  `temperature: 0`. Parse failures return `[]`; the pipeline soft-fails
  per-step (one bad prompt won't poison the whole turn).
- `validate_phase` can only *veto* a phase, so `async_turn` always proposes at
  least the next `PHASE_ORDER` entry — a generator that never advances the
  phase itself still progresses, and a model jumping ahead is clamped to +1.
- Under `GENERATOR=steered` the reply dict carries an extra `steer_status`
  (`steered` | `fallback` | `none`) so the session bar can show that a picked
  strategy silently failed instead of leaving it to be guessed from tone.
- Sessions and loaded query-graphs are process-local dicts in `api.py`;
  restart wipes them. No persistence layer.
- `api.py` mounts Gradio **after** all FastAPI routes are defined.
- `EXTRACTION_TIMEOUT` is defined and configurable but is not currently
  enforced as an `asyncio.wait_for` cutoff — extraction is always awaited.
  Tier B is what actually decouples from the response.
