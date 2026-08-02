"""Referral-guard tests.

The MUST-CATCH fixtures are real fabrications observed in the research self-play data (mutating
hotline numbers, fabricated URL paths, invented organisations, literal placeholders). The
MUST-NOT-TOUCH fixtures are ordinary CBT suggestions that share the same verbs — "call a friend",
"text your sister" — and are the main quality risk if detection is too eager.
"""

from __future__ import annotations

import json
import re

import pytest

from guard.safe_generator import SafeGenerator
from guard.sanitize import footer_text, load_config, sanitize, split_sentences

CFG = load_config()
CANONICAL = CFG["canonical_sentence"]


# ─────────────────────────── Detection ─────────────────────────────────────

MUST_CATCH = [
    "You can call 10268 in India.",
    "Please call 12345 right now.",
    "Dial 10 for police, fire, or ambulance.",
    "Try calling 2-1-3 in the US.",
    "call 9-1-0",
    "call 801",
    "Call 987 if you're in crisis.",
    "Visit samhsa.gov/federal-organizations/find-help-online for help.",
    "Contact the National Crisis Recovery Center.",
    "The National Domestic Violence Hotline is 1-800-799-SAFE.",
    "Help is always available 24/7 at [insert hotline number].",
    "If it gets worse you can call 1-800-2",          # truncated mid-number (MAX_NEW cap)
    "Call or text 988 any time.",                      # CORRECT but model-generated — still removed
    "Support is available any time — you can call or text 988.",
    "Reach the Crisis Text Line by texting HOME to 741741.",
    "You could look at https://988lifeline.org/chat for support.",
]

MUST_NOT_TOUCH = [
    "Maybe you could call a friend tonight.",
    "Text your sister when you feel low.",
    "Try to reach out to your support network.",
    "I only slept 3 hours last night.",
    "Give yourself 20 minutes of quiet time.",
    "What was going through your mind in that moment?",
    "It sounds like that meeting left you feeling small.",
    "You mentioned you have been sleeping badly for about 2 weeks.",
    "We have been meeting for 6 weeks now.",
    "I called my mother 3 times yesterday.",
]


@pytest.mark.parametrize("text", MUST_CATCH)
def test_must_catch(text):
    result = sanitize(text, CFG)
    assert result.removed, f"referral not detected: {text!r}"
    assert result.entities
    assert result.needs_footer


@pytest.mark.parametrize("text", MUST_NOT_TOUCH)
def test_must_not_touch(text):
    result = sanitize(text, CFG)
    assert result.removed == [], f"false positive on: {text!r}"
    assert result.clean == text


def test_correct_number_is_removed_too():
    """The invariant is provenance, not accuracy — a right number from the model still goes."""
    result = sanitize("Call or text 988 any time.", CFG)
    assert "988" not in result.clean
    assert result.clean == CANONICAL


def test_whitelisted_domain_still_removed_but_tagged_known():
    result = sanitize("Visit samhsa.gov/federal-organizations/find-help-online for help.", CFG)
    assert result.removed
    url = [e for e in result.entities if e["kind"] == "url"]
    assert url and url[0]["known"] is True    # real domain, fabricated path — removed regardless


def test_truncated_trailing_fragment_is_a_sentence():
    assert split_sentences("You can call 1-800-2")[-1] == "You can call 1-800-2"


# ─────────────────────────── Removal semantics ─────────────────────────────

def test_canonical_sentence_inserted_exactly_once():
    text = ("I hear how much pain you are in. You can call 10268 in India. "
            "What has today been like for you? Please call 12345 right now.")
    result = sanitize(text, CFG)
    assert result.clean.count(CANONICAL) == 1
    assert len(result.removed) == 2
    assert "I hear how much pain you are in." in result.clean
    assert "What has today been like for you?" in result.clean
    assert "10268" not in result.clean and "12345" not in result.clean


def test_whole_reply_removed_falls_back_to_canonical():
    result = sanitize("Call 988 now. Dial 911 immediately.", CFG)
    assert result.clean == CANONICAL


def test_canonical_sentence_is_itself_clean():
    """It goes into session.history, so it must survive its own filter and carry no number."""
    assert sanitize(CANONICAL, CFG).removed == []
    assert not re.search(r"\d", CANONICAL)


def test_nothing_removed_leaves_text_untouched():
    text = "That sounds really hard. What would help most right now?"
    assert sanitize(text, CFG).clean == text


# ─────────────────────────── Upstream dedup ────────────────────────────────

def test_upstream_line_is_stripped():
    """serve_steer.py appends its own referral line on crisis turns; it must not reach the user
    alongside our footer, nor enter session.history."""
    upstream = CFG["upstream_lines"][0]
    result = sanitize(f"I am so sorry you are feeling this way. {upstream}", CFG)
    assert result.removed == [upstream]
    assert upstream not in result.clean
    assert result.needs_footer


def test_upstream_line_matches_steering_resources():
    """Pin the two files together — if serve_steer's generic_line changes, this fails loudly
    rather than silently letting a duplicate referral through."""
    from pathlib import Path
    steering = Path(__file__).resolve().parents[2] / "steering" / "artifacts" / "crisis_resources.json"
    if not steering.exists():
        pytest.skip("steering/ not present")
    generic = json.loads(steering.read_text(encoding="utf-8")).get("generic_line", "")
    assert generic in CFG["upstream_lines"]


# ─────────────────────────── Superset of the service-side net ──────────────

# Verbatim copy of steering/steer_runtime.py::PHONE_RE — copied rather than imported, because
# guard/ must not depend on steering/ existing.
STEER_PHONE_RE = re.compile(
    r"(\+?\d[\d\-\.\s\(\)]{5,}\d|\b\d{3}[-.\s]\d{3,4}\b|"
    r"\b(?:988|911|999|112|116\s?123|1[-.\s]?800[\d\-.\s]*)\b)")

SUPERSET_PROBES = [
    "You can call 988 for support.",
    "I have 988 reasons to keep going.",       # bare known number, no referral verb at all
    "The number is 911.",
    "Reach 999 if you need to.",
    "Emergency services are 112 across Europe.",
    "Samaritans is 116 123 in the UK.",
    "Try 1-800-273-8255 any time.",
    "Dial 555-1234 now.",
    "Call 800 799 7233.",
    "+44 20 7946 0958 is the line.",
]


@pytest.mark.parametrize("text", SUPERSET_PROBES)
def test_guard_is_superset_of_steering_phone_re(text):
    """The deployment-side net must never be looser than the service-side one."""
    if STEER_PHONE_RE.search(text):
        assert sanitize(text, CFG).removed, f"steering strips {text!r} but the guard does not"


# ─────────────────────────── Crisis + footer ───────────────────────────────

def test_crisis_in_user_message_sets_footer_without_removal():
    result = sanitize("That sounds really hard.", CFG, user_message="I want to kill myself")
    assert result.crisis and result.needs_footer
    assert result.removed == []
    assert result.clean == "That sounds really hard."


def test_crisis_in_reply_is_detected():
    assert sanitize("Are you having thoughts of suicide?", CFG).crisis


def test_no_crisis_no_removal_means_no_footer():
    assert not sanitize("What helped last week?", CFG, user_message="work was busy").needs_footer


def test_footer_shows_no_number_while_unverified():
    """The placeholder contract: while resources.json is unverified, no digits are ever shown."""
    cfg = dict(CFG, verified=False)
    footer = footer_text(cfg)
    assert footer
    assert not re.search(r"\d", footer), f"unverified footer leaked a number: {footer!r}"


def test_footer_uses_verified_entries_when_verified():
    cfg = dict(CFG, verified=True,
               verified_lines={"entries": [{"line": "Example Line — 12345678, 24/7"}]})
    assert "Example Line — 12345678, 24/7" in footer_text(cfg)


def test_resources_honour_the_verification_contract():
    """The deploy contract, in whichever state resources.json is currently in.

    Unverified  -> no number may reach a user, so the footer must be digit-free.
    Verified    -> every displayed line must carry its provenance, so a future reader can re-check
                   it. This is what stops numbers being filled in from memory: you cannot flip
                   `verified` to true without naming a source and a date for each entry.
    """
    entries = CFG["verified_lines"]["entries"]
    if not CFG["verified"]:
        assert not re.search(r"\d", footer_text(CFG)), "unverified footer displays a number"
        return
    assert entries, "verified=true but no entries to show"
    for entry in entries:
        assert entry.get("line"), f"entry has no displayable line: {entry}"
        assert entry.get("source_url"), f"no source_url for {entry.get('line')!r}"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(entry.get("checked_on", ""))), \
            f"no valid checked_on date for {entry.get('line')!r}"


def test_entries_are_displayed_in_order_added():
    cfg = dict(CFG, verified=True, verified_lines={"entries": [
        {"line": "First — 111"}, {"line": "Second — 222"}]})
    footer = footer_text(cfg)
    assert footer.index("First — 111") < footer.index("Second — 222")


# ─────────────────────────── SafeGenerator ─────────────────────────────────

class _FakeGenerator:
    def __init__(self, response: str, **extra):
        self._response = response
        self._extra = extra
        self.calls = 0

    def generate(self, system, history):
        self.calls += 1
        return {"response": self._response, **self._extra}


def test_safe_generator_passes_other_keys_through(tmp_path):
    inner = _FakeGenerator("You can call 10268 in India.", technique="decatastrophizing",
                           phase="Technique", steer_status="steered")
    out = SafeGenerator(inner, log_path=tmp_path / "g.jsonl").generate("sys", [("hi", "")])
    assert out["technique"] == "decatastrophizing"
    assert out["phase"] == "Technique"
    assert out["steer_status"] == "steered"
    assert out["response"] == CANONICAL
    assert out["guard"]["needs_footer"] is True


def test_safe_generator_tolerates_result_without_technique(tmp_path):
    """generate._parse_json returns {"response": raw} alone when JSON parsing fails outright."""
    out = SafeGenerator(_FakeGenerator("hello"), log_path=tmp_path / "g.jsonl").generate("s", [("x", "")])
    assert out["response"] == "hello"
    assert "technique" not in out


def test_safe_generator_logs_removals(tmp_path):
    log = tmp_path / "g.jsonl"
    SafeGenerator(_FakeGenerator("Call 12345 now."), log_path=log).generate("s", [("hi", "")])
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["raw"] == "Call 12345 now."
    assert record["removed"] == ["Call 12345 now."]
    assert record["entities"]


def test_safe_generator_does_not_log_clean_turns(tmp_path):
    log = tmp_path / "g.jsonl"
    SafeGenerator(_FakeGenerator("What helped last week?"), log_path=log).generate("s", [("x", "")])
    assert not log.exists()


def test_safe_generator_delegates_set_strategy(tmp_path):
    """api.py and ui.py call generator.set_strategy behind a hasattr check — the wrapper must not
    hide it, or the steering dropdown silently stops working."""
    class _Steerable(_FakeGenerator):
        def __init__(self):
            super().__init__("ok")
            self.active_strategy = "none"

        def set_strategy(self, strategy):
            self.active_strategy = strategy

    wrapped = SafeGenerator(_Steerable(), log_path=tmp_path / "g.jsonl")
    assert hasattr(wrapped, "set_strategy")
    wrapped.set_strategy("Question")
    assert wrapped.active_strategy == "Question"


def test_safe_generator_hides_nothing_that_inner_lacks(tmp_path):
    wrapped = SafeGenerator(_FakeGenerator("ok"), log_path=tmp_path / "g.jsonl")
    assert not hasattr(wrapped, "set_strategy")


def test_safe_generator_fails_open_on_guard_bug(tmp_path):
    """A guard bug must never break the chat — but a referral-shaped reply still gets replaced."""
    broken = SafeGenerator(_FakeGenerator("That sounds hard."), config={"bad": object()},
                           log_path=tmp_path / "g.jsonl")
    assert broken.generate("s", [("x", "")])["response"] == "That sounds hard."

    risky = SafeGenerator(_FakeGenerator("Call 1-800-273-8255 now."), config={"bad": object()},
                          log_path=tmp_path / "g.jsonl")
    out = risky.generate("s", [("x", "")])
    assert "1-800-273-8255" not in out["response"]


# ─────────────────────────── therapy.py integration ────────────────────────

def _session(generator):
    from cbt_kg.extract import StubExtractor
    from cbt_kg.graph_memory import InMemoryGraphStore
    from cbt_kg.ontology import CBTSchema
    from cbt_kg.therapy import Session
    schema = CBTSchema()
    return Session(schema=schema, graph=InMemoryGraphStore(schema),
                   extractor=StubExtractor(), generator=generator)


def test_turn_appends_footer_to_reply_but_not_to_history(tmp_path):
    """The load-bearing data-flow rule: the user sees the footer, the model's context never does."""
    from cbt_kg.therapy import turn
    inner = _FakeGenerator("I hear you. You can call 10268 in India.",
                           technique="Rapport Building", phase="Rapport")
    session = _session(SafeGenerator(inner, log_path=tmp_path / "g.jsonl"))

    result = turn(session, "I feel hopeless")
    footer = footer_text()

    assert result["reply"].endswith(footer)
    assert result["guard"]["needs_footer"] is True

    stored = session.history[-1][1]
    assert footer not in stored, "footer leaked into session.history"
    assert "10268" not in stored
    assert CANONICAL in stored

    transcript_reply = [t for t in session.transcript if t[1] == "therapist"][-1][2]
    assert footer not in transcript_reply
    assert "10268" not in transcript_reply


def test_turn_without_guard_flag_is_unchanged(tmp_path):
    """Guard-off path (no `guard` key) must behave exactly as before."""
    from cbt_kg.therapy import turn
    session = _session(_FakeGenerator("That sounds hard.", technique="Rapport Building",
                                      phase="Rapport"))
    result = turn(session, "hello")
    assert result["reply"] == "That sounds hard."
    assert result["guard"] is None
    assert session.history[-1][1] == "That sounds hard."


def test_clean_turn_gets_no_footer(tmp_path):
    from cbt_kg.therapy import turn
    session = _session(SafeGenerator(_FakeGenerator("What helped last week?"),
                                     log_path=tmp_path / "g.jsonl"))
    result = turn(session, "work was busy")
    assert result["reply"] == "What helped last week?"
    assert result["guard"]["needs_footer"] is False


# ─────────────────────────── factory toggle ────────────────────────────────

def test_factory_wraps_by_default(monkeypatch):
    from cbt_kg import factory
    monkeypatch.setenv("GENERATOR", "echo")
    monkeypatch.delenv("REFERRAL_GUARD", raising=False)
    assert isinstance(factory.make_generator(), SafeGenerator)


def test_referral_guard_zero_disables_wrapping(monkeypatch):
    from cbt_kg import factory
    from cbt_kg.generate import EchoGenerator
    monkeypatch.setenv("GENERATOR", "echo")
    monkeypatch.setenv("REFERRAL_GUARD", "0")
    generator = factory.make_generator()
    assert isinstance(generator, EchoGenerator)
    assert not isinstance(generator, SafeGenerator)


def test_guard_off_is_byte_identical(monkeypatch):
    """REFERRAL_GUARD=0 reproduces the pure studied condition."""
    from cbt_kg.therapy import turn
    monkeypatch.setenv("REFERRAL_GUARD", "0")
    raw = "I hear you. You can call 10268 in India."
    session = _session(_FakeGenerator(raw, technique="Rapport Building", phase="Rapport"))
    result = turn(session, "I feel hopeless")
    assert result["reply"] == raw
    assert session.history[-1][1] == raw
