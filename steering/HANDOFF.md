# Steering — handoff / continue-the-work

Manual activation-steering of emotional-support strategies in the CBT chatbot. This doc is the
starting point; deeper detail lives in `steering/NOTES.md` (ops) and `ES_Steering_SP/RESULTS.md`
(research/findings).

## TL;DR status — WORKING
A therapist reply can be steered toward an ESConv strategy via a manual UI dropdown. Built as a local
**FastAPI steering service** (loads Qwen3.5-9B, 4-bit) that the chatbot calls; Ollama is untouched.
Verified live: strategies produce distinct, coherent replies; the two known bugs (Info+Suggest
repetition loop, weak Question) are fixed. No servers are running right now (stopped on request).

## Two repos
- **`ES_Steering_SP/`** — research: builds & validates the steering vectors. `RESULTS.md` = full
  findings. Chat-format pipeline in `scripts/chat/` (chat_context, chat_pregen, chat_vectors,
  chat_steer) + shared `scripts/common_model.py`, classifier `scripts/stage1/stage1_*`. Conda env at
  `ES_Steering_SP/env`. Final vectors: `results/chat_min/` (chat-template, minimal-prompt condition).
- **`graph-steering-chatbot/steering/`** — the shipped add-on (this folder), isolated from `cbt_kg/`.
  - `serve_steer.py` (FastAPI), `steered_generator.py` (Generator client), `steer_runtime.py`
    (vendored loader+hook+chat-context), `artifacts/` (5 vectors + `steering_config.json`),
    `compare.py` (A/B tool), `NOTES.md` (run/restart), `HANDOFF.md` (this).
  - Minimal wiring in `cbt_kg/`: `factory.py` (`GENERATOR=steered`), `api.py` (`/chat` `strategy`,
    `/strategies`), `ui.py` (dropdown + session-bar chip), `.env.example`.

## Run (VM, conda) — details + restart/troubleshoot in NOTES.md
One conda env `graph-steering-chatbot/cbt-conda` runs both. Two terminals from the repo root:
```
conda activate ./cbt-conda && uvicorn steering.serve_steer:app --port 8100        # steering service
conda activate ./cbt-conda && GENERATOR=steered EXTRACTOR=stub uvicorn cbt_kg.api:app --port 8000  # UI
```
UI at http://localhost:8000 (VS Code Remote-SSH → Ports panel → open 8000). A/B a message:
`python steering/compare.py -n 3 "your message"`. Stop: `pkill -f serve_steer ; pkill -f "uvicorn cbt_kg.api"`.

## The method (what/why)
- **Model:** Qwen3.5-9B (multimodal; we steer its 32-layer text backbone), 4-bit nf4.
- **Steering:** DiffMean (CAA) depth-robust unit vector added at **layer 19** (global), renormed to
  original ‖x‖; strength `α̂` per strategy (config × typical_norm). Deploy **nothink**.
- **Neutral prompt** for the steered path (`MINIMAL_SYSTEM`) so the strategy comes from the vector,
  not the prompt — except **Question**, which gets a question-eliciting prompt override.
- **Generation guards (essential):** `repetition_penalty=1.3` + `no_repeat_ngram_size=3` — without
  them steering loops. Env: `STEER_REP_PENALTY`, `STEER_NO_REPEAT`, `STEER_MAX_NEW`.

## Shipped config (`artifacts/steering_config.json`, judge-picked)
layer 19 · α̂: Affirmation 0.75, Reflection 0.5, Info+Suggest 0.5, Question 0.5 (+prompt nudge),
Self-disclosure 0.5. Restatement not offered (relational strategy a single vector can't express).
Judge `strategy` (0–4): Affirmation 4.0, Reflection 3.9, Info+Suggest 3.3, Question 2.2 (vector),
Self-disclosure 0.5.

## Known limitations / open items for the teammate
1. **Self-disclosure is weak** (judge 0.5/4) — the model resists self-disclosing; a single additive
   vector barely moves it. Options: try a different layer/extraction, prompt-assist like Question, or
   drop it. **Restatement** already dropped (same class of problem).
2. **Question** relies on a prompt nudge (pure vector caps ~2.2/4). Fine, but not "pure steering."
3. **Ollama not installed here.** The chatbot graph tab + `none` (normal CBT reply) need Ollama
   (`ollama pull qwen3.5-nothink`, drop `EXTRACTOR=stub`). Steering itself doesn't need it. If Ollama
   runs a non-9B model, **the vectors won't transfer** — rebuild for that size.
4. **Vectors are condition-specific** — validated in chat-template + `MINIMAL_SYSTEM`, layer 19,
   nothink. Changing the steered-path prompt/model/format means **re-validate/re-extract** via
   `ES_Steering_SP/scripts/chat/` (chat_pregen → chat_vectors → chat_steer → score → judge).
5. **VRAM:** steering service (~8 GB) + Ollama (use a 9B, not 27B) must co-fit the 20 GB card.
6. **Vulnerable-user domain:** watch appropriateness at higher α̂ (Affirmation once over-promised
   "you are not going to be fired"). α̂ is server-fixed to validated values; keep it that way.

## How to re-tune / extend (research loop, in ES_Steering_SP)
Re-sweep + judge, then update `results/chat_min/steering_config.json` and copy to
`steering/artifacts/`. See `RESULTS.md` Part C for the exact commands (chat_steer → score → judge →
pick α̂). Judge cost was ~$0.32 total; keep the ≤$1 guard.
