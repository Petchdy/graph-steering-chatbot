"""Minimal, self-contained steering runtime (vendored from ES_Steering_SP).

Trimmed copies of common_model.load_lm / get_text_layers, stage1_steer.make_steer_hook, and the
chat-format context builder, so the chatbot repo runs the steering service WITHOUT a runtime
dependency on the research repo. Keep in sync with ES_Steering_SP if the method changes.

Steering = add a unit DiffMean vector at the text-backbone layer L, renormalized to the original ‖x‖
(a rotation of the residual toward the strategy direction). alpha = alpha_hat * typical_norm.
"""

from __future__ import annotations

import os

MODEL_NAME = os.environ.get("STEER_MODEL", "Qwen/Qwen3.5-9B")
FOUR_BIT = os.environ.get("STEER_4BIT", "1") != "0"

# Neutral-but-suitable steered-path prompt: warm/safe counselor TONE, but it does NOT tell the model
# to ask/advise/affirm/etc. — so the strategy that appears is attributable to the steering vector,
# not the prompt (best showcase of steering). Per-strategy overrides can be added to SYSTEM_BY_STRATEGY.
MINIMAL_SYSTEM = (
    "You are a warm, supportive counselor talking with someone going through a hard time. "
    "Respond with a brief, caring 1-2 sentence reply. Be kind, and never give unsafe or harmful advice."
)
# Per-strategy system-prompt overrides. Question is the one strategy where prompt and vector point the
# SAME way and pure-vector caps at ~2/4 (and the base already sometimes asks), so we nudge it to ask —
# the vector still supplies strength. All other strategies stay on the neutral MINIMAL_SYSTEM.
QUESTION_SYSTEM = (
    "You are a warm, supportive counselor talking with someone going through a hard time. "
    "Respond with a brief, caring 1-2 sentence reply that ENDS WITH ONE gentle, open-ended question "
    "inviting them to say more. Be kind, and never give unsafe or harmful advice."
)
SYSTEM_BY_STRATEGY: dict[str, str] = {"Question": QUESTION_SYSTEM}


def system_for(strategy: str) -> str:
    return SYSTEM_BY_STRATEGY.get(strategy, MINIMAL_SYSTEM)


def load_lm(attn: str = "eager"):
    """(model, tokenizer), 4-bit by default, eval mode. Inputs go to cuda."""
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        from transformers import AutoModelForImageTextToText as AutoCls  # Qwen3.5 is multimodal
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForCausalLM as AutoCls  # type: ignore

    kwargs = {"attn_implementation": attn}
    if FOUR_BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
        model = AutoCls.from_pretrained(MODEL_NAME, **kwargs)
    else:
        kwargs["dtype"] = torch.bfloat16
        model = AutoCls.from_pretrained(MODEL_NAME, **kwargs).to(
            "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def n_layers() -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    return int(getattr(cfg, "text_config", cfg).num_hidden_layers)


def get_text_layers(model):
    """The text decoder-layer ModuleList (works for the multimodal wrapper)."""
    import torch.nn as nn
    target = n_layers(); fallback = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == target:
            fallback = mod
            if name.endswith("layers"):
                return mod
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Could not locate a {target}-deep decoder ModuleList in {MODEL_NAME}")


def make_steer_hook(v_t, alpha):
    """forward_pre_hook: x -> renorm(x + alpha*v) per position. v_t unit-norm."""
    def hook(module, args, kwargs):
        if alpha == 0.0:
            return None
        hs = args[0]
        orig = hs.norm(dim=-1, keepdim=True)
        out = hs + alpha * v_t.to(hs.dtype)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8) * orig
        return (out,) + tuple(args[1:]), kwargs
    return hook


def new_token_rep_penalty(penalty: float, prompt_len: int):
    """A LogitsProcessor that applies repetition_penalty to tokens generated in THIS reply only,
    NOT the conversation history.

    HuggingFace's built-in `repetition_penalty` penalizes every token present anywhere in the input
    — including the whole prior conversation. In a live multi-turn chat the history is largely the
    MODEL's own phrasing, so a 1.3x penalty progressively bans its natural register and, after a few
    turns, tips generation into degenerate word-salad (verified: ~20% of samples at turn ~4 with the
    Question vector). Gold-history offline evals never trip this because human text overlaps the
    model's preferred tokens far less. The anti-loop guard is only needed WITHIN a reply, so we scope
    the penalty to freshly generated tokens. Returns None if penalty<=1 (no-op).
    """
    import torch
    from transformers import LogitsProcessor

    if penalty is None or penalty <= 1.0:
        return None

    class _NewTokenRepetitionPenalty(LogitsProcessor):
        def __init__(self, penalty: float, prompt_len: int):
            self.penalty, self.prompt_len = float(penalty), int(prompt_len)

        def __call__(self, input_ids, scores):
            new = input_ids[:, self.prompt_len:]
            if new.shape[1] == 0:
                return scores
            score = torch.gather(scores, 1, new)
            score = torch.where(score < 0, score * self.penalty, score / self.penalty)
            return scores.scatter(1, new, score)

    return _NewTokenRepetitionPenalty(penalty, prompt_len)


def build_chat_context(messages: list[dict], tokenizer, strategy: str,
                       system: str | None = None) -> str:
    """Prepend the (strategy-specific) system prompt to the client/therapist messages and render the
    chat template ending at the assistant generation prompt (thinking off). `system` overrides the
    per-strategy default (used by the baseline/none path to keep the CBT graph-aware system prompt)."""
    sys_msg = system if system is not None else system_for(strategy)
    msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + list(messages)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
