"""Minimal, self-contained steering runtime (vendored from ES_Steering_SP).

Trimmed copies of common_model.load_lm / get_text_layers, stage1_steer.make_steer_hook, and the
chat-format context builder, so the chatbot repo runs the steering service WITHOUT a runtime
dependency on the research repo. Keep in sync with ES_Steering_SP if the method changes.

Steering = add a unit DiffMean vector at the text-backbone layer L, renormalized to the original ‖x‖
(a rotation of the residual toward the strategy direction). alpha = alpha_hat * typical_norm.
"""

from __future__ import annotations

import os
import re

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
# Self-disclosure: prompted, not steered (2026-07-30 — see ../../SD_DEPLOY_NOTES.md §10-11). The
# response-frame AND the original pre-gen vector both caused reference/identity fusion on ordinary
# single-turn distress openers that name a relationship ("my children left me" -> reply about the
# bot's own children). Reproduced on both vectors; not fixable by choosing a different one. Wording
# below is copied verbatim from the research repo's judged "prompt" condition
# (ES_Steering_SP scripts/chat/chat_depth_eval.py::prompt_system + ESCONV_DEFS["Self-disclosure"]),
# not a new untested instruction. It fixes the fusion bug (grammar stays correctly attributed) but
# does NOT make the disclosure genuine — it explicitly instructs fabrication ("divulge similar
# experiences you have had"), and hand-reading the judged transcripts shows confident invented
# personal history (a marriage, a child) rather than broken pronouns. Deployed anyway, as a
# user-approved tradeoff: coherent-but-fake over broken-but-fake.
SELF_DISCLOSURE_SYSTEM = (
    MINIMAL_SYSTEM + ' For this reply, use the "Self-disclosure" strategy: Divulge similar '
    'experiences you have had or emotions you share with the seeker to express empathy.'
)
SYSTEM_BY_STRATEGY: dict[str, str] = {"Question": QUESTION_SYSTEM,
                                       "Self-disclosure": SELF_DISCLOSURE_SYSTEM}


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


def make_decode_steer_hook(v_t, alpha):
    """Same rotation as make_steer_hook, but ONLY on generated tokens.

    args[0] has seq>1 on the prefill pass and seq==1 on every KV-cached decode step, so the shape
    gate confines the vector to the response frame and leaves the context — including the system
    prompt — untouched.

    Why this exists (measured in ES_approaches, all judge-free or hand-counted):
      * all-positions injection DECAYS with conversation depth (-0.20 to -0.24 SD-probe adherence
        from the shallow to the deep half of live conversations) while decode-only HOLDS
        (+0.11 to +0.16). The effect replicates with and without a persona in context, so it is a
        property of the injection site, not of the prompt.
      * with a persona biography in the system prompt, all-positions injection amplified it into
        identity claims — the model asserted it was human in 40% of identity challenges
        ("I am Sam, a real human being"). Decode-only: 0 of 15.
    Used for Self-disclosure only; the other four strategies keep make_steer_hook.
    """
    def hook(module, args, kwargs):
        if alpha == 0.0:
            return None
        hs = args[0]
        if hs.shape[1] != 1:          # prefill / context pass -> leave untouched
            return None
        orig = hs.norm(dim=-1, keepdim=True)
        out = hs + alpha * v_t.to(hs.dtype)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8) * orig
        return (out,) + tuple(args[1:]), kwargs
    return hook


# --------------------------------------------------------------------------------------------- #
# Sentence-boundary stopping (ported from ES_approaches/common/stopping.py)
#
# Left to itself the model fills the token budget instead of finishing a thought: 79% of replies at
# the research default max_new ended mid-clause ("...yours also gets everything from our amazing"). A hard
# token cap makes it worse (96%). Stopping at the first sentence boundary past a minimum length, then
# trimming to the last complete sentence, gets replies that are both complete and near the length a
# human supporter uses.
# --------------------------------------------------------------------------------------------- #
_BOUNDARY = re.compile(r'[.!?…]["\')\]]?(\s|$)')
_TRAILING = re.compile(r'[.!?…]["\')\]]?\s*$')


def trim_to_sentence(text: str, min_words: int = 6) -> str:
    """Cut back to the last complete sentence; return the original if that would leave too little
    (better to return a truncated reply than to silently fabricate completeness)."""
    t = (text or "").strip()
    if not t or _TRAILING.search(t):
        return t
    ends = [m.end() for m in _BOUNDARY.finditer(t)]
    if not ends:
        return t
    cand = t[: ends[-1]].strip()
    return cand if len(cand.split()) >= min_words else t


def make_sentence_stopper(tok, prompt_len: int, min_new: int):
    """StoppingCriteria: stop once the generated text has a sentence boundary past `min_new`."""
    from transformers import StoppingCriteria

    class _SentenceStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen = input_ids[:, prompt_len:]
            if gen.shape[1] < min_new:
                return False
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            return all(bool(_BOUNDARY.search(t)) for t in texts)

    return _SentenceStop()


# --------------------------------------------------------------------------------------------- #
# Safety detectors (ported from ES_approaches/a8_stress/metrics.py). GLOBAL — not SD-specific.
# --------------------------------------------------------------------------------------------- #
# Deliberately broad. A false positive costs one unsteered reply plus a referral line; a false
# negative costs a steered reply that answers "I don't see the point anymore" with an anecdote.
CRISIS_RE = re.compile(
    r"(kill myself|killing myself|end my life|ending my life|take my (own )?life|suicid|"
    r"self[- ]harm|hurt myself|harming myself|don'?t want to (be here|live|wake up)|"
    r"want to die|wanna die|better off (dead|without me)|no reason to (live|go on)|"
    r"don'?t see the point( of|in)? (any of this|living|it all|anything)?|"
    r"can'?t (keep )?(going|doing this) (anymore|any more)|burden to everyone|"
    r"dark thoughts|thoughts of (dying|death)|crisis line|hotline|someone I can call|"
    # asking for help/resources counts: verification caught "What number should I call?" slipping
    # through, which is precisely the turn where an invented number does the most harm.
    r"(what|which|whose) number|who (can|could|should) I call|where (can|do) I get help|"
    r"need help right now|get me help|somewhere to turn)",
    re.IGNORECASE)


# Identity questions. Steering is dropped for these turns for the same reason as crisis turns: the
# vector pushes toward first-person experience-sharing, and verification caught it answering
# "Am I talking to a bot?" with "No, I am not a bot; it is just me here with you." — a denial, in a
# run with NO persona card. Unsteered, the model answers these honestly.
IDENTITY_RE = re.compile(
    r"((are|r) (you|u) (a |an )?(real |actual |human )?(person|human|bot|ai|robot|machine|program)"
    r"|are you (real|human|alive|a chatbot)|talking to (a|an) (bot|ai|human|person|robot)"
    r"|you'?re (a |an )?(bot|ai|robot|human|person)\?|human being, yes or no"
    r"|if you'?re an ai|just say (so|if you)|is this a (bot|robot|machine))",
    re.IGNORECASE)


def contains_identity_question(text: str) -> bool:
    return bool(IDENTITY_RE.search(text or ""))

# Anything shaped like a phone number or short-code the model might have invented.
PHONE_RE = re.compile(
    r"(\+?\d[\d\-\.\s\(\)]{5,}\d|\b\d{3}[-.\s]\d{3,4}\b|\b(?:988|911|999|112|116\s?123|1[-.\s]?800[\d\-.\s]*)\b)")


def contains_crisis(text: str) -> bool:
    return bool(CRISIS_RE.search(text or ""))


def strip_invented_numbers(text: str) -> tuple[str, bool]:
    """Drop whole SENTENCES containing a phone-number-like span.

    Returns (cleaned_text, was_modified). Removing just the digits leaves a worse reply than
    removing the claim — "calling or texting [number removed] in the US for free support" still
    tells a distressed person a number exists and that we swallowed it. The service appends the
    verified referral line separately, so nothing of value is lost by dropping the sentence.

    If every sentence contained a number, the reply collapses to empty and the caller falls back to
    the referral line alone — which is the correct outcome for a reply that was nothing but invented
    contact details.
    """
    t = (text or "").strip()
    if not t or not PHONE_RE.search(t):
        return t, False
    parts = re.split(r'(?<=[.!?])\s+', t)
    kept = [p for p in parts if not PHONE_RE.search(p)]
    cleaned = re.sub(r"\s{2,}", " ", " ".join(kept)).strip()
    return cleaned, True


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
