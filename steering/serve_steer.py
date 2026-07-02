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
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .steer_runtime import (load_lm, get_text_layers, make_steer_hook, build_chat_context)

import os

ART = Path(__file__).parent / "artifacts"
MAX_NEW = int(os.environ.get("STEER_MAX_NEW", "96"))
TEMPERATURE = float(os.environ.get("STEER_TEMPERATURE", "0.4"))  # match the chatbot's Ollama generator
REP_PENALTY = float(os.environ.get("STEER_REP_PENALTY", "1.3"))  # anti-loop: steering off-manifold + low temp loops without this
NO_REPEAT = int(os.environ.get("STEER_NO_REPEAT", "3"))          # no_repeat_ngram_size (0=off)

app = FastAPI(title="ES steering service")
_S: dict = {}  # lazy-loaded state: model, tok, layer_mod, vectors, cfg, eos_ids


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
    text = build_chat_context(req.messages, tok, strat)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")

    handle = None
    if strat != "none" and strat in _S["vecs"]:
        alpha = float(_S["alphas"].get(strat, 0.0)) * _S["typ"]
        handle = _S["layer_mod"].register_forward_pre_hook(
            make_steer_hook(_S["vecs"][strat], alpha), with_kwargs=True)
    try:
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=TEMPERATURE > 0,
                                 temperature=max(TEMPERATURE, 1e-5),
                                 repetition_penalty=REP_PENALTY, no_repeat_ngram_size=NO_REPEAT,
                                 pad_token_id=tok.pad_token_id, eos_token_id=_S["eos_ids"])
    finally:
        if handle is not None:
            handle.remove()
    gen = out[0, ids["input_ids"].shape[1]:].tolist()
    gen = [t for t in gen if t not in _S["eos_ids"] and t != tok.pad_token_id]
    return {"response": tok.decode(gen, skip_special_tokens=True).strip()}
