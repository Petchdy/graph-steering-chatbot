# steering/ — activation-steering add-on for the CBT chatbot

Manual, per-turn steering of the therapist reply toward an ESConv emotional-support strategy, using
DiffMean activation vectors on **Qwen3.5-9B**. Isolated here so it doesn't mix with `cbt_kg/` or the
graph data. Research + vector building lives in `ES_Steering_SP/` (see its `RESULTS.md`); this folder
holds only the runtime service, the client generator, and the finalized vectors.

## Why a separate service (not Ollama)
The chatbot generates via **Ollama**, which exposes no residual stream — activation steering needs a
transformers forward-hook. So steering runs as a **local FastAPI service** (`serve_steer.py`) that
loads Qwen3.5-9B (4-bit, ~8 GB GPU) and adds the strategy vector at **layer 19** during generation.
Free — local GPU only, no API/token cost. Ollama is untouched (still supplies CBT technique/phase).

## How a turn works (GENERATOR=steered)
`SteeredRemoteGenerator` (a `Generator`) composes the existing Ollama generator:
- strategy **none** → delegates entirely to Ollama (identical to today).
- else → Ollama gives CBT `technique`/`phase` (drives the graph); the steering service gives the
  steered **reply text**. On any service error it falls back to the Ollama reply (chat never breaks).

The steered path uses a **neutral, non-guiding system prompt** (`steer_runtime.MINIMAL_SYSTEM`: warm/
safe counselor tone, but it does NOT tell the model to ask/advise/affirm) so the strategy that appears
is attributable to the **vector**, not the prompt. `SYSTEM_BY_STRATEGY` allows per-strategy overrides.

## Files
- `serve_steer.py` — FastAPI: `GET /strategies`, `POST /generate {messages, strategy}`.
- `steered_generator.py` — `SteeredRemoteGenerator` (client; imports `cbt_kg.interfaces.Generator`).
- `steer_runtime.py` — vendored minimal loader + hook + chat-context (trimmed from ES_Steering_SP
  `common_model` / `stage1_steer` / `chat_context`), so this repo runs self-contained.
- `artifacts/steering_config.json` — `{layer:19, typical_norm, alphas{...}, strategies[...]}`.
- `artifacts/vectors/<strategy>/vector_depth.npz` — the 5 chat-format depth vectors.
- `requirements-steer.txt` — service-only deps (torch/transformers/bitsandbytes/fastapi). The base
  chatbot's `cbt_kg/requirements.txt` stays ML-free.

## Layer
**All strategies steer at a single global layer 19** (of Qwen3.5-9B's 32-layer text backbone), stored
in `artifacts/steering_config.json` (`layer:19`) and in each `vector_depth.npz`. L19–20 is the
best-steering band: separability is flat across L18–22 (±0.02 AUC), and a steering-effect test showed
deeper layers (e.g. L24) have similar/higher AUC but steer *worse*. L20 is essentially equivalent.

## Offered strategies + α̂ (chat-format, JUDGE-picked, rep-control on)
LLM-judge-selected α̂ (0–4 `strategy` score at that α̂, with `fluency` intact):
| strategy | α̂ | judge strategy | note |
|---|---|---|---|
| Affirmation and Reassurance | 0.75 | **4.0** | excellent |
| Reflection of feelings | 0.5 | **3.9** | strong |
| Info+Suggest | 0.5 | **3.3** | strong; was 0.75 (looped/over-steered) |
| Question | 0.5 | 2.2 (vector) | + **question prompt nudge** (`SYSTEM_BY_STRATEGY["Question"]`) → reliably asks |
| Self-disclosure | 0.5 | 0.5 | genuinely weak — model resists self-disclosing |

Restatement is **not offered** (relational strategy a single additive vector can't express).

**Generation guards (in `serve_steer.py`, tune via env):** `repetition_penalty=1.3` (`STEER_REP_PENALTY`)
+ `no_repeat_ngram_size=3` (`STEER_NO_REPEAT`) — without these, steering + low temperature falls into
repetition loops (Info+Suggest broke this way). `max_new_tokens=96` (`STEER_MAX_NEW`).
**Question** uses a question-eliciting system prompt (the only per-strategy prompt override); all
others use the neutral `MINIMAL_SYSTEM`.

## Run (this VM — one conda env `./cbt-conda` runs both processes)
Two terminals, from the repo root:
```bash
# Terminal 1 — steering service (loads Qwen3.5-9B, ~8 GB GPU)
conda activate ./cbt-conda
uvicorn steering.serve_steer:app --host 127.0.0.1 --port 8100

# Terminal 2 — chatbot UI
conda activate ./cbt-conda
export GENERATOR=steered STEER_URL=http://localhost:8100
export EXTRACTOR=stub          # <- no Ollama needed (graph tab won't populate); drop this if Ollama is up
uvicorn cbt_kg.api:app --port 8000
```
- UI at **http://localhost:8000/** — pick a **Steering strategy** by the chat box and send a message.
- **Steering-only demo (no Ollama):** with `EXTRACTOR=stub`, keep a strategy **selected** (not "none")
  — the reply is fully produced by the steering service. "none" and the knowledge-graph both need Ollama.
- API: `POST /chat {session_id, message, strategy}` · `GET /strategies`.

## Viewing the UI from VS Code (Remote-SSH)
When uvicorn starts on the VM, VS Code (connected via Remote-SSH) **auto-forwards** the port. Open the
**Ports** panel (View → Ports, or the "Ports" tab next to the terminal), find **8000**, and click the
globe to open it in your local browser — or run **"Simple Browser: Show"** (Cmd/Ctrl-Shift-P) and paste
`http://localhost:8000`. If 8000 isn't listed, click **Forward a Port** and add `8000`.

## Full experience later (with Ollama, optional)
Ollama gives you the CBT knowledge graph + the "none" (normal) reply. To add it:
```bash
curl -fsSL https://ollama.com/install.sh | sh      # install
ollama pull qwen3.5-nothink                         # or any qwen3.5 nothink tag; set OLLAMA_MODEL
```
Then drop `EXTRACTOR=stub` and the graph tab populates; "none" returns the normal CBT reply.

## Caveats
- Steering is **Qwen3.5-9B-specific** — the Ollama `qwen3.5-nothink` tag must be the 9B model.
- Vectors were validated in this exact **chat-template + minimal-prompt** condition; changing the
  steered-path system prompt would require re-validating in `ES_Steering_SP/scripts/chat/`.
- Vulnerable-user domain: α̂ is server-fixed to validated values (no client override); the minimal
  prompt still carries a "never give unsafe or harmful advice" floor.

## Restart / stop / "address already in use"
Ports: chatbot UI = **8000**, steering service = **8100**.

**Start both** (two terminals, repo root):
```bash
conda activate ./cbt-conda && uvicorn steering.serve_steer:app --host 127.0.0.1 --port 8100
```
```bash
conda activate ./cbt-conda
export GENERATOR=steered STEER_URL=http://localhost:8100 EXTRACTOR=stub   # drop EXTRACTOR=stub once Ollama is installed
uvicorn cbt_kg.api:app --port 8000
```

**"[Errno 98] address already in use"** = a server is already running on that port. Either just use it
(open http://localhost:8000), or free the port and restart:
```bash
# see what's on the port (8000 or 8100)


# stop it (pattern-kill is easiest)
pkill -f "uvicorn cbt_kg.api"        # chatbot
pkill -f "serve_steer"               # steering service
# ...or run on a different port:  uvicorn cbt_kg.api:app --port 8001
```

**Stop everything / free the GPU:**
```bash
pkill -f "serve_steer" ; pkill -f "uvicorn cbt_kg.api"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # should drop to ~0 MiB
```

**View:** VS Code (Remote-SSH) → Ports panel → open forwarded **8000**; or Cmd/Ctrl-Shift-P →
"Simple Browser: Show" → http://localhost:8000. In the Therapy tab, pick a Steering strategy and send.
Compare vs unsteered: `python steering/compare.py "your message"` (service must be up).
