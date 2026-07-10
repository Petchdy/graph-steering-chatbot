# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

Three independent pieces, run from the **repo root**:

- **`cbt_kg/`** — the base chatbot (FastAPI + Gradio). See `cbt_kg/CLAUDE.md` for its full
  architecture (V4_flat extraction pipeline, dependency rule, graph stores, query engine, API
  routes). Read that file before editing anything under `cbt_kg/`.
- **`steering/`** — optional activation-steering add-on that lets a therapist reply be steered
  toward an ESConv emotional-support strategy (Qwen3.5-9B, DiffMean vectors at layer 19). Isolated
  from `cbt_kg/` on purpose. See `steering/HANDOFF.md` (status/handoff) and `steering/NOTES.md`
  (detailed run/restart/troubleshoot + method).
- **`V4_flat/`** — the original batch pipeline `cbt_kg/ontology.py` was ported from verbatim. It is
  a frozen reference (git-ignored, not a tracked package) — don't modify it to fix `cbt_kg/` bugs;
  port fixes forward into `cbt_kg/ontology.py` instead.

`pyproject.toml` at the root only configures pytest (`testpaths = cbt_kg/tests`, repo root on
`pythonpath`) — there is no root-level package/build config.

## Commands

Base chatbot (no GPU needed):
```bash
pip install -r cbt_kg/requirements.txt
cp cbt_kg/.env.example cbt_kg/.env        # set EXTRACTOR=stub GENERATOR=echo for zero external services
uvicorn cbt_kg.api:app --reload           # UI at http://localhost:8000/
```

Tests (fully offline — `cbt_kg/conftest.py` forces `EXTRACTOR=stub GENERATOR=echo GRAPH_BACKEND=memory`):
```bash
pytest                                              # from repo root
pytest cbt_kg/tests/test_therapy.py                 # single file
pytest cbt_kg/tests/test_therapy.py::test_name      # single test
```

Steering add-on (needs a GPU, ~8GB, separate venv from `steering/requirements-steer.txt` — install
torch from the `cu12x` index matching the local driver *before* the rest of that file, or you'll get
a "driver too old" CUDA init error):
```bash
# terminal 1 — steering microservice, must be launched from repo root (imports `steering.*`)
uvicorn steering.serve_steer:app --host 127.0.0.1 --port 8100

# terminal 2 — chatbot wired to the steering service
export GENERATOR=steered STEER_URL=http://localhost:8100 EXTRACTOR=stub   # drop EXTRACTOR=stub once Ollama is up
uvicorn cbt_kg.api:app --port 8000
```
A/B compare: `python steering/compare.py -n 3 "your message"`. In the UI, `strategy="none"` requires
Ollama (or falls back to an unhooked call to the steering service); pick a real strategy to demo
steering without Ollama.

**GPU-lean single-model mode (no Ollama, ~8 GB, graph still works):** run everything on the one HF
model behind the steering service. `STEER_NO_OLLAMA=1` makes the generator produce both `none` and
steered replies from the HF model; `OLLAMA_HOST=http://localhost:8100` routes the extractor/narration
to the service's Ollama-compatible `/api/generate` (no-hook) route, so the graph keeps populating.
```bash
uvicorn steering.serve_steer:app --host 127.0.0.1 --port 8100   # the only model
export GENERATOR=steered STEER_NO_OLLAMA=1 EXTRACTOR=local OLLAMA_HOST=http://localhost:8100
uvicorn cbt_kg.api:app --port 8000
```
Cost is latency (~8+ HF generations/turn, 4-bit HF slower than Ollama GGUF); a lock in
`serve_steer.py` serializes model calls so steering can't contaminate extraction. Details in
`steering/HANDOFF.md`.

## Cross-cutting architecture notes

- **`cbt_kg/` never imports `steering/`, except through `factory.py`.** `make_generator()` in
  `cbt_kg/factory.py` does `from steering.steered_generator import SteeredRemoteGenerator` only when
  `GENERATOR=steered` — this is the one sanctioned coupling point between the two packages, mirroring
  `factory.py`'s existing rule that it's the only file allowed to import concretes.
- **`SteeredRemoteGenerator`** (`steering/steered_generator.py`) implements `cbt_kg.interfaces.Generator`
  and *composes* `LocalLLMGenerator` rather than replacing it: Ollama still supplies CBT
  `technique`/`phase` (drives the knowledge graph), the steering microservice supplies only the reply
  text. Any steering-service error falls back to the Ollama reply — the chat path must never hard-fail
  on a steering hiccup.
- **The steering service is a separate process on purpose** (`steering/serve_steer.py`, FastAPI on
  port 8100): Ollama exposes no residual stream, and activation steering needs a transformers
  forward-hook, so it can't happen inside Ollama. It lazy-loads Qwen3.5-9B (4-bit) on first
  `/generate` call.
- **Vectors are condition-specific** (`steering/artifacts/`): chat-template + `MINIMAL_SYSTEM` prompt,
  layer 19, nothink, Qwen3.5-**9B** specifically. If the steered-path prompt, model, or format
  changes, the vectors must be re-validated/re-extracted via `ES_Steering_SP/scripts/chat/` (an
  external research repo referenced in `steering/HANDOFF.md`, not part of this repo).
