"""Local FastAPI steering microservice (free — local GPU only).

Loads Qwen3.5-9B (4-bit) once + the chat-format depth vectors, and steers a single therapist reply
toward a chosen ESConv strategy via a forward-hook at layer L. The chatbot's SteeredRemoteGenerator
calls POST /generate; Ollama is untouched (it still supplies CBT technique/phase for the graph).

Endpoints:
  GET  /strategies                     -> {strategies:[...], alphas:{...}, layer, model}
  POST /generate {messages, strategy}  -> {"response": "<steered reply>"}
       messages = [{"role":"user"|"assistant","content":...}, ...] (client=user, therapist=assistant)
       strategy = one of the offered names, or "none" (no steering)

Run (from repo root, with the steering venv):
  uvicorn steering.serve_steer:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .steer_runtime import (load_lm, get_text_layers, make_steer_hook, build_chat_context,
                            new_token_rep_penalty)

import os

ART = Path(__file__).parent / "artifacts"
MAX_NEW = int(os.environ.get("STEER_MAX_NEW", "96"))
TEMPERATURE = float(os.environ.get("STEER_TEMPERATURE", "0.4"))  # match the chatbot's Ollama generator
REP_PENALTY = float(os.environ.get("STEER_REP_PENALTY", "1.3"))  # anti-loop WITHIN a reply; scoped to
# new tokens only (see new_token_rep_penalty) so the multi-turn history (mostly the model's own
# phrasing) is NOT penalized — the whole-context form degenerates into word-salad after a few turns.
NO_REPEAT = int(os.environ.get("STEER_NO_REPEAT", "3"))          # no_repeat_ngram_size (0=off)

app = FastAPI(title="ES steering service")
_S: dict = {}  # lazy-loaded state: model, tok, layer_mod, vectors, cfg, eos_ids
# One model serves both the steered reply (/generate, hook ON) and the extractor (/api/generate,
# hook OFF). A forward hook attaches to the SHARED layer module, and extraction runs as a background
# task that can overlap the next steered reply — so serialize all model access to guarantee the
# steering hook is never live during an extraction forward pass (steering cannot leak into extraction).
_GEN_LOCK = threading.Lock()


def _dirname(strategy: str) -> str:
    return strategy.replace(" ", "_").replace("/", "_")


def _load():
    if _S:
        return
    import torch
    cfg = json.loads((ART / "steering_config.json").read_text(encoding="utf-8"))
    model, tok = load_lm(attn="eager")
    layer = int(cfg["layer"])
    layer_mod = get_text_layers(model)[layer]
    vecs = {}
    for s in cfg["strategies"]:
        z = np.load(ART / "vectors" / _dirname(s) / "vector_depth.npz")
        vecs[s] = torch.tensor(z["v"].astype("float32"), device="cuda")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list({im_end, tok.eos_token_id} - {None})
    _S.update(model=model, tok=tok, layer=layer, layer_mod=layer_mod, vecs=vecs, cfg=cfg,
              eos_ids=eos_ids, typ=float(cfg["typical_norm"]), alphas=cfg["alphas"])
    print(f"[steer] loaded {len(vecs)} vectors @ L{layer}; strategies={cfg['strategies']}", flush=True)


class GenReq(BaseModel):
    messages: list[dict]
    strategy: str = "none"
    system: str | None = None  # optional system-prompt override (baseline/none path passes the
    #                            CBT graph-aware prompt so the unsteered reply stays on-task)


class OllamaGenReq(BaseModel):
    """Ollama-compatible /api/generate body — lets the CBT extractor (cbt_kg/extract.py::
    _ollama_generate) run on THIS single HF model instead of a second Ollama process. Only the
    fields the extractor actually sends are honored; the rest are accepted and ignored."""
    model: str | None = None
    prompt: str = ""
    stream: bool = False
    options: dict | None = None
    system: str | None = None


@app.get("/strategies")
def strategies():
    _load()
    return {"strategies": _S["cfg"]["strategies"], "alphas": _S["alphas"],
            "layer": _S["layer"], "model": _S["model"].name_or_path}


@app.post("/generate")
def generate(req: GenReq):
    import torch
    _load()
    tok, model = _S["tok"], _S["model"]
    strat = req.strategy
    text = build_chat_context(req.messages, tok, strat, system=req.system)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = ids["input_ids"].shape[1]
    rep_proc = new_token_rep_penalty(REP_PENALTY, prompt_len)  # new-tokens-only; None if REP_PENALTY<=1
    procs = [rep_proc] if rep_proc is not None else None

    # Serialize: the steering hook attaches to the shared layer module, so no other model call
    # (e.g. a background extraction request) may run while it is registered.
    with _GEN_LOCK:
        handle = None
        if strat != "none" and strat in _S["vecs"]:
            alpha = float(_S["alphas"].get(strat, 0.0)) * _S["typ"]
            handle = _S["layer_mod"].register_forward_pre_hook(
                make_steer_hook(_S["vecs"][strat], alpha), with_kwargs=True)
        try:
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=TEMPERATURE > 0,
                                     temperature=max(TEMPERATURE, 1e-5),
                                     logits_processor=procs, no_repeat_ngram_size=NO_REPEAT,
                                     pad_token_id=tok.pad_token_id, eos_token_id=_S["eos_ids"])
        finally:
            if handle is not None:
                handle.remove()
    gen = out[0, ids["input_ids"].shape[1]:].tolist()
    gen = [t for t in gen if t not in _S["eos_ids"] and t != tok.pad_token_id]
    return {"response": tok.decode(gen, skip_special_tokens=True).strip()}


@app.post("/api/generate")
def api_generate(req: OllamaGenReq):
    """Ollama-compatible completion on the SAME HF model with NO steering hook — serves the CBT
    extractor / query narration so the whole app runs on one model (point them here via
    OLLAMA_HOST=<steer_url>). Deterministic (greedy) to match the extractor's temperature=0 calls."""
    import torch
    _load()
    tok, model = _S["tok"], _S["model"]
    opts = req.options or {}
    temp = float(opts.get("temperature", 0.0))
    max_new = int(opts.get("num_predict", os.environ.get("STEER_EXTRACT_MAX_NEW", "512")))
    # Ollama's /api/generate applies the model's chat template to a raw prompt; mirror that by
    # wrapping the prompt as a single user turn (plus any system override).
    msgs = ([{"role": "system", "content": req.system}] if req.system else []) + \
           [{"role": "user", "content": req.prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")

    with _GEN_LOCK:  # never overlap a steered /generate that has the hook registered
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                                 temperature=max(temp, 1e-5),
                                 pad_token_id=tok.pad_token_id, eos_token_id=_S["eos_ids"])
    gen = out[0, ids["input_ids"].shape[1]:].tolist()
    gen = [t for t in gen if t not in _S["eos_ids"] and t != tok.pad_token_id]
    return {"response": tok.decode(gen, skip_special_tokens=True).strip(), "done": True}
