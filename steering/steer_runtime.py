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


def build_chat_context(messages: list[dict], tokenizer, strategy: str) -> str:
    """Prepend the (strategy-specific) system prompt to the client/therapist messages and render the
    chat template ending at the assistant generation prompt (thinking off)."""
    sys_msg = system_for(strategy)
    msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + list(messages)
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
