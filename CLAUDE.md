# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env              # defaults: local extractor + local generator (qwen3.5-nothink via Ollama)
ollama pull qwen3.5-nothink       # required for default extractor + generator
uvicorn api:app --reload          # Gradio UI at / · FastAPI at /chat, /reset, /graph/{session_id}
pytest                            # all tests; no external services needed (uses stub + echo)
pytest tests/test_orchestrator.py::test_async_turn_returns_expected_keys  # single test
```

Offline (no Ollama): set `EXTRACTOR=stub GENERATOR=echo` in `.env` before running.
Backend selection: `GRAPH_BACKEND` (memory|neo4j), `EXTRACTOR` (stub|local), `GENERATOR` (echo|local|openrouter).

## Architecture

CACTUS CBT therapy chatbot: a local LLM (`qwen3.5-nothink` via Ollama) plays therapist,
following CACTUS paper principles (guided discovery, questioner not advisor). A
rich knowledge graph (nodes + edges, CBT ontology) tracks what has been revealed
in the session. A Cytoscape.js panel in the Gradio UI shows the live graph state.

The point of the codebase is the **swap architecture** — a real clinical ontology
drops into `schema.py` and real prompts into `prompts.py` without touching anything
else. See `PRD_cactus_v6.md` for the full spec.

### The dependency rule (load-bearing)

`interfaces.py` defines four `Protocol`s — `Schema`, `GraphStore`, `Extractor`,
`Generator` — plus `OntologyField`, `GraphNode`, `GraphEdge`. **Every module
except `factory.py` may import only from `interfaces.py`**, never from the
concretes. `factory.py` is the single place that knows which implementation backs
each protocol; env vars select at construction time.

### The async turn loop

`orchestrator.async_turn(session, user_message)` (FastAPI calls this directly;
sync `turn()` wrapper exists for tests and Gradio):

1. Snapshot pre-turn `cbt_context()` and `snapshot()` — generator uses these.
2. **Launch extraction + generation concurrently** as `asyncio.Task`s.
3. Await generator → get `{response, technique, phase}`.
4. Await extraction (whether finished or still running). The 4-step pipeline:
   a. `extractor.extract_nodes(message, window, schema)` → list of node candidates.
   b. `extractor.extract(message, schema)` → flat deltas for `apply_deltas()`.
   c. For each candidate: `graph.upsert_node(label, props, turn_id)`.
   d. `extractor.resolve_edges(new_node, all_nodes, window, subject_edges)` →
      `graph.resolve_edge(...)` for each confirmed predicate.
5. `validate_phase()` — enforces minimum-turns/fields before accepting phase advance.
6. `graph.apply_session_state(phase, technique)` → persists session-state fields.
7. Return `{reply, technique, phase, deltas, slots, extraction_mode}`.

`extraction_mode` is `"sync"` if extraction beat generate to completion, `"async"`
otherwise — informational only. Per-session `extraction_lock` (asyncio.Lock)
prevents concurrent writes to the same session's graph.

### Schema fields + rich graph

`CBTSchema` has 9 clinical fields (priority 1–9, filled by `CBTExtractor` from
client speech) and 2 session-state fields (priority 0, written by the LLM's JSON
output): `session_phase` and `active_technique`. `missing()` never surfaces
priority-0 fields.

Beyond the flat slot-fill state, `CBTSchema` also publishes the rich graph
ontology via `node_classes()`, `edge_map()`, and `subject_edges()`:

- `CBT_NODE_CLASSES` — 12 ontology classes (Problem, Goal, Intervention, Homework,
  CoreBelief, IntermediateBelief, Situation, AutomaticThought, Reaction,
  AdaptiveResponse, Session, Client).
- `CBT_EDGE_MAP` — 25 typed predicates (triggers, leadsTo, stemsFrom, …).
- `CBT_SUBJECT_EDGES` / `CBT_OBJECT_EDGES` — derived index used by edge resolution.

`InMemoryGraphStore.reset()` pre-creates one placeholder node per class and one
placeholder edge per `(subj, pred, obj)` tuple. `upsert_node()` upgrades the
placeholder to `status="found"` on first match, then creates new instances (multi).
`resolve_edge()` flips an edge to `status="found"`.

### Implementation notes worth knowing

- `LocalLLMGenerator` uses Ollama's **native `/api/chat`**, not `/v1`. Always
  passes `"think": false`. The `LOCAL_LLM_BASE_URL` env var accepts `/v1` for
  convenience but the suffix is stripped internally.
- `CBTExtractor` uses `/api/generate` with `format: "json"` and
  `options.temperature = 0`. Unknown keys are silently dropped; JSON parse
  failures return `{}` or `[]`.
- `Generator.generate()` returns `dict` (`{"response", "technique", "phase"}`),
  not `str`. `EchoGenerator` returns the same shape for offline testing.
- The Cytoscape.js graph panel in `ui.py` updates synchronously after each turn
  via the Gradio callback — no HTTP polling. `GET /graph/{session_id}` exists for
  external clients and returns Cytoscape-compatible node/edge JSON derived from
  `graph.nodes()` + `graph.edges()`.
- Sessions are process-local dicts in `api.py`. Restart loses chat history.
- `api.py` mounts Gradio **after** FastAPI routes are defined.
- `EXTRACTION_TIMEOUT` env var (default 8s) is informational right now —
  extraction is always awaited before the response returns, since `asyncio.run()`
  in the sync wrapper would cancel a fire-and-forget task. Hooking a true async
  fallback into a persistent event loop is a future change.
