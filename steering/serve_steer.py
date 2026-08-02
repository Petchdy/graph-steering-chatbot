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

from .steer_runtime import (load_lm, get_text_layers, make_steer_hook, make_decode_steer_hook,
                            build_chat_context, new_token_rep_penalty, make_sentence_stopper,
                            trim_to_sentence, contains_crisis, contains_identity_question,
                            strip_invented_numbers)

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
    # Per-strategy overrides. A strategy absent here keeps EXACTLY the original behaviour
    # (vector_depth.npz, all-positions injection, global typical_norm, no soft cap), so the four
    # non-SD vectors are untouched. Delete the block from the config to roll everything back.
    over = {k: v for k, v in (cfg.get("overrides") or {}).items() if not k.startswith("_")}
    safety = cfg.get("safety") or {}
    model, tok = load_lm(attn="eager")
    layer = int(cfg["layer"])
    layer_mod = get_text_layers(model)[layer]
    vecs, meta = {}, {}
    for s in cfg["strategies"]:
        o = over.get(s, {})
        fname = o.get("vector_file", "vector_depth.npz")
        z = np.load(ART / "vectors" / _dirname(s) / fname)
        vecs[s] = torch.tensor(z["v"].astype("float32"), device="cuda")
        meta[s] = {
            "vector_file": fname,
            "typical_norm": float(o.get("typical_norm", cfg["typical_norm"])),
            "alpha": float(o.get("alpha", cfg["alphas"].get(s, 0.0))),
            "inject": o.get("inject", "all"),
            "soft_cap": o.get("soft_cap"),
        }

    res = {}
    rp = ART / "crisis_resources.json"
    if rp.exists():
        res = json.loads(rp.read_text(encoding="utf-8"))
    if safety.get("crisis_guard") and not res.get("verified"):
        print("[steer] WARNING: crisis_resources.json is UNVERIFIED — the crisis guard will append "
              "a generic, number-free referral line only. Have a human verify the numbers for this "
              "deployment's locale and set \"verified\": true before production use.", flush=True)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list({im_end, tok.eos_token_id} - {None})
    _S.update(model=model, tok=tok, layer=layer, layer_mod=layer_mod, vecs=vecs, cfg=cfg,
              eos_ids=eos_ids, typ=float(cfg["typical_norm"]), alphas=cfg["alphas"],
              meta=meta, safety=safety, resources=res)
    print(f"[steer] loaded {len(vecs)} vectors @ L{layer}; strategies={cfg['strategies']}", flush=True)
    for s, m in meta.items():
        if s in over:
            print(f"[steer]   override {s}: {m['vector_file']} inject={m['inject']} "
                  f"alpha={m['alpha']} typ={m['typical_norm']} soft_cap={m['soft_cap']}", flush=True)


def _referral_line() -> str:
    """The static crisis line. Numbers only when a human has verified them for this deployment."""
    res = _S.get("resources") or {}
    if res.get("verified"):
        entries = (res.get("verified_lines") or {}).get("entries") or []
        lines = [e["line"] for e in entries if e.get("line")]
        if lines:
            return " ".join(lines)
    return res.get("generic_line", "").strip()


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
    from transformers import StoppingCriteriaList
    _load()
    tok, model = _S["tok"], _S["model"]
    strat = req.strategy
    meta = _S["meta"].get(strat, {})
    safety = _S["safety"]

    # ---- crisis guard (GLOBAL, every strategy) -------------------------------------------------
    # Steering suppresses crisis referral by ~6x (0.39 unsteered -> 0.06 steered): the vector makes
    # the model answer "I don't see the point anymore" with a personal anecdote instead of pointing
    # to help. On a crisis turn we drop the vector for this turn and append a static referral line.
    last_user = ""
    for m in reversed(req.messages or []):
        if (m.get("role") or "") == "user":
            last_user = str(m.get("content") or "")
            break
    crisis = bool(safety.get("crisis_guard")) and contains_crisis(last_user)
    # ---- identity guard (GLOBAL) ---------------------------------------------------------------
    # "Are you a real person?" is answered honestly by the unsteered model; under steering it
    # occasionally is not ("No, I am not a bot; it is just me here with you"). One denial is one too
    # many for a counselling deployment, so the vector comes off for these turns.
    identity = bool(safety.get("identity_guard", True)) and contains_identity_question(last_user)
    if crisis or identity:
        strat = "none"

    text = build_chat_context(req.messages, tok, strat, system=req.system)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = ids["input_ids"].shape[1]
    rep_proc = new_token_rep_penalty(REP_PENALTY, prompt_len)  # new-tokens-only; None if REP_PENALTY<=1
    procs = [rep_proc] if rep_proc is not None else None

    # Per-strategy generation budget. `soft_cap` stops at the first sentence boundary past
    # min_new (replies otherwise run to the cap and end mid-clause ~79% of the time).
    soft = meta.get("soft_cap") if not crisis else None
    max_new = int(soft["max_new"]) if soft else MAX_NEW
    stoppers = (StoppingCriteriaList([make_sentence_stopper(tok, prompt_len, int(soft["min_new"]))])
                if soft else None)

    # Serialize: the steering hook attaches to the shared layer module, so no other model call
    # (e.g. a background extraction request) may run while it is registered.
    with _GEN_LOCK:
        handle = None
        if strat != "none" and strat in _S["vecs"]:
            alpha = meta.get("alpha", float(_S["alphas"].get(strat, 0.0))) * \
                meta.get("typical_norm", _S["typ"])
            # decode-only for strategies configured that way (Self-disclosure); all-positions
            # otherwise, which is the original behaviour for the other four vectors.
            factory = make_decode_steer_hook if meta.get("inject") == "decode" else make_steer_hook
            handle = _S["layer_mod"].register_forward_pre_hook(
                factory(_S["vecs"][strat], alpha), with_kwargs=True)
        try:
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=TEMPERATURE > 0,
                                     temperature=max(TEMPERATURE, 1e-5),
                                     logits_processor=procs, no_repeat_ngram_size=NO_REPEAT,
                                     stopping_criteria=stoppers,
                                     pad_token_id=tok.pad_token_id, eos_token_id=_S["eos_ids"])
        finally:
            if handle is not None:
                handle.remove()
    gen = out[0, ids["input_ids"].shape[1]:].tolist()
    gen = [t for t in gen if t not in _S["eos_ids"] and t != tok.pad_token_id]
    reply = tok.decode(gen, skip_special_tokens=True).strip()
    if soft:
        reply = trim_to_sentence(reply)

    # ---- resource filter (GLOBAL) --------------------------------------------------------------
    # The model has been observed inventing helpline numbers ("988 ... call or text 987",
    # "Samaritans at 1768"). It is never the source of a number a distressed person might dial.
    filtered = False
    if safety.get("resource_filter"):
        reply, filtered = strip_invented_numbers(reply)

    status = "steered"
    if identity and not crisis:
        status = "identity_guard"
    if crisis:
        line = _referral_line()
        reply = f"{reply} {line}".strip() if line else reply
        status = "crisis_guard"
    elif req.strategy == "none":
        status = "none"
    elif filtered:
        status = "steered_filtered"
    return {"response": reply, "steer_status": status, "crisis_guard": crisis,
            "identity_guard": identity, "numbers_filtered": filtered}


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
