"""End-to-end async-turn-loop tests using the offline stub + echo generator.

Exercises:
  - async_turn returns {reply, technique, phase, extraction_mode, ...}
  - per-turn extraction (via StubExtractor "Label: text" lines) fills the graph
  - validate_phase refuses to advance until the V4_flat node-class minimums hold
  - reset clears the graph + history
"""

from __future__ import annotations

import asyncio

from cbt_kg.extract import StubExtractor
from cbt_kg.generate import EchoGenerator
from cbt_kg.graph_memory import InMemoryGraphStore
from cbt_kg.interfaces import GraphNode
from cbt_kg.ontology import CBTSchema
from cbt_kg.therapy import Session, apply_graph, async_turn, turn, validate_phase


def _make_session() -> Session:
    schema = CBTSchema()
    return Session(
        schema=schema,
        graph=InMemoryGraphStore(schema),
        extractor=StubExtractor(),
        generator=EchoGenerator(),
    )


def test_async_turn_returns_expected_keys():
    session = _make_session()
    result = asyncio.run(async_turn(session, "Situation: exam tomorrow"))
    assert "reply" in result
    assert "phase" in result
    assert "technique" in result
    assert "extraction_mode" in result
    assert result["extraction_mode"] in ("sync", "async")


def test_stub_extractor_creates_nodes():
    session = _make_session()
    turn(session, "Situation: exam tomorrow\nAutomaticThought: I will fail")
    counts = session.graph.snapshot()["counts"]
    assert counts.get("Situation", 0) >= 1
    assert counts.get("AutomaticThought", 0) >= 1


def test_phase_gates_block_when_classes_missing():
    g = InMemoryGraphStore(CBTSchema())
    # No Problem yet — cannot advance to Exploration.
    assert validate_phase("Exploration", "Rapport", g, turn_count=5) == "Rapport"
    g.upsert_node("Problem", {"description": "work stress", "domain": "work"}, 1)
    # Problem present + 2 turns → Exploration allowed.
    assert validate_phase("Exploration", "Rapport", g, turn_count=2) == "Exploration"
    # Technique still blocked (no AT + Situation yet).
    assert validate_phase("Technique", "Exploration", g, turn_count=10) == "Exploration"
    # Add AT + Situation, advance.
    g.upsert_node("Situation", {"description": "exam"}, 3)
    g.upsert_node("AutomaticThought", {"content": "I will fail"}, 3)
    assert validate_phase("Technique", "Exploration", g, turn_count=5) == "Technique"


def test_phase_gates_block_consolidation_until_adaptive_response():
    g = InMemoryGraphStore(CBTSchema())
    g.upsert_node("Problem", {"description": "x", "domain": "work"}, 1)
    g.upsert_node("Situation", {"description": "y"}, 2)
    g.upsert_node("AutomaticThought", {"content": "z"}, 3)
    # ≥12 turns but no AdaptiveResponse → still Technique.
    assert validate_phase("Consolidation", "Technique", g, turn_count=12) == "Technique"
    g.upsert_node("AdaptiveResponse", {"content": "I can handle this"}, 13)
    assert validate_phase("Consolidation", "Technique", g, turn_count=12) == "Consolidation"


def test_validate_phase_allows_staying_or_going_back():
    g = InMemoryGraphStore(CBTSchema())
    assert validate_phase("Rapport", "Rapport", g, 0) == "Rapport"
    assert validate_phase("Rapport", "Technique", g, 0) == "Rapport"


def test_reset_clears_graph_and_history():
    session = _make_session()
    turn(session, "Situation: exam")
    assert session.graph.count_found("Situation") >= 1
    session.graph.reset()
    session.history.clear()
    session.transcript.clear()
    session.turn_count = 0
    assert session.graph.count_found("Situation") == 0


def test_extraction_lock_exists():
    session = _make_session()
    assert isinstance(session.extraction_lock, asyncio.Lock)


def test_apply_graph_keeps_placeholder_scaffold():
    """A found-only correction must not delete the 'not yet discovered' scaffold.

    LiveGraphReader and the canvas saveJSON export both emit `found` items only,
    so applying one with a bare replace_all erased every placeholder — a
    load-then-apply round trip made untouched classes vanish from the panel.
    """
    session = _make_session()
    turn(session, "AutomaticThought: I will fail")

    before = {n.node_id: n.status for n in session.graph.nodes()}
    assert any(n.label == "Situation" and n.status == "missing"
               for n in session.graph.nodes()), "expected an unfound Situation"

    found_nodes = [n for n in session.graph.nodes() if n.status == "found"]
    found_edges = [e for e in session.graph.edges() if e.status == "found"]
    apply_graph(session.graph, found_nodes, found_edges)

    after = {n.node_id: n.status for n in session.graph.nodes()}
    assert after == before, "load-then-apply round trip must be lossless"
    assert any(n.label == "Situation" and n.status == "missing"
               for n in session.graph.nodes())


def test_apply_graph_still_honours_deletions():
    """Keeping the scaffold must not resurrect a node the user deleted."""
    session = _make_session()
    turn(session, "Situation: exam tomorrow\nAutomaticThought: I will fail")
    sit = next(n for n in session.graph.nodes()
               if n.label == "Situation" and n.status == "found")

    kept = [n for n in session.graph.nodes()
            if n.status == "found" and n.node_id != sit.node_id]
    apply_graph(session.graph, kept, [])

    assert not any(n.node_id == sit.node_id for n in session.graph.nodes())


def test_apply_graph_preserves_node_order():
    """Placement must not shift on apply.

    The canvas derives a node's slot from its index in this list, so re-sorting
    the graph — e.g. hoisting the found nodes ahead of the placeholders — makes
    the drawing jump around even though the graph is otherwise unchanged.
    """
    session = _make_session()
    turn(session, "Situation: exam tomorrow\nAutomaticThought: I will fail")

    before_nodes = [n.node_id for n in session.graph.nodes()]
    before_edges = [e.edge_id for e in session.graph.edges()]

    found_nodes = [n for n in session.graph.nodes() if n.status == "found"]
    found_edges = [e for e in session.graph.edges() if e.status == "found"]
    apply_graph(session.graph, found_nodes, found_edges)

    assert [n.node_id for n in session.graph.nodes()] == before_nodes
    assert [e.edge_id for e in session.graph.edges()] == before_edges


def test_apply_graph_appends_new_nodes_at_the_end():
    """A node created in the canvas has no prior position, so it goes last."""
    session = _make_session()
    turn(session, "AutomaticThought: I will fail")
    before = [n.node_id for n in session.graph.nodes()]

    found = [n for n in session.graph.nodes() if n.status == "found"]
    newcomer = GraphNode(node_id="sit_99", label="Situation",
                         props={"description": "added by hand"}, status="found")
    apply_graph(session.graph, found + [newcomer], [])

    after = [n.node_id for n in session.graph.nodes()]
    assert after[-1] == "sit_99"
    assert [i for i in after if i != "sit_99"] == before


def test_edit_cancel_does_not_replay_the_turn():
    """Dismissing the inline editor hands back the ORIGINAL text.

    Rewriting on that would spend a whole generate call — the visible symptom
    being a spinner that runs after pressing cancel — only to land back where it
    started. Unchanged text must short-circuit.
    """
    import types
    from cbt_kg import ui

    session = ui._new_session()
    ui._bot_respond("I felt anxious at work", [], session, "none")
    history = ui._history_to_chat(session)
    idx = next(i for i, m in enumerate(history) if m["role"] == "user")

    calls = []
    real = ui.edit_turn
    ui.edit_turn = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        same = types.SimpleNamespace(index=idx, value=history[idx]["content"])
        out = ui._on_edit_message(session, history, "none", same)
        assert calls == [], "cancel must not replay the turn"
        assert len(out) == 6, "must match edit_outputs arity"

        changed = types.SimpleNamespace(index=idx, value="I felt calm instead")
        ui._on_edit_message(session, history, "none", changed)
        assert len(calls) == 1, "a genuine edit must still replay"
    finally:
        ui.edit_turn = real


def test_edit_cancel_uses_gradios_previous_value():
    """Cancel is detected from EditData.previous_value, the shape Gradio sends.

    Indexing the chat history is only the fallback: the transcript opens with the
    therapist's INTRO and a failed turn contributes no assistant message, so the
    index can misalign — while previous_value is authoritative.
    """
    import gradio as gr
    from cbt_kg import ui

    session = ui._new_session()
    ui._bot_respond("I felt anxious at work", [], session, "none")
    chat = ui._history_to_chat(session)
    idx = next(i for i, m in enumerate(chat) if m["role"] == "user")
    text = chat[idx]["content"]

    calls = []
    real = ui.edit_turn
    ui.edit_turn = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        cancel = gr.EditData(None, {"index": idx, "previous_value": text,
                                    "value": text})
        out = ui._on_edit_message(session, chat, "none", cancel)
        assert calls == []
        assert len(out) == 6

        # a deliberately EMPTY history is supplied: previous_value must still win
        edited = gr.EditData(None, {"index": idx, "previous_value": text,
                                    "value": "a different sentence"})
        ui._on_edit_message(session, [], "none", edited)
        assert len(calls) == 1, "real edit must replay even with no history to index"
    finally:
        ui.edit_turn = real


def test_edit_event_shows_no_loading_indicator():
    """Pressing cancel must not put a spinner on screen.

    Gradio fires `edit` when the inline editor closes, cancel included, so the
    event itself is unavoidable — what has to stay off is the indicator. Two
    independent sources, both load-bearing:
      * show_progress="hidden" suppresses the progress overlay;
      * the handler must NOT be a generator, or Gradio registers the event as
        streaming and the chatbot shows a pending state that show_progress does
        not govern.
    """
    import inspect
    from cbt_kg import ui

    assert not inspect.isgeneratorfunction(ui._on_edit_message), (
        "a generator handler makes Gradio stream this event, which re-introduces "
        "the pending indicator on cancel")

    cfg = ui.demo.get_config_file()
    edit_deps = [d for d in cfg["dependencies"]
                 if any(ev == "edit" for _, ev in (d.get("targets") or []))]
    assert len(edit_deps) == 1
    dep = edit_deps[0]
    assert dep["show_progress"] == "hidden"
    assert dep["types"]["generator"] is False


def _contrast(a: str, b: str) -> float:
    def lum(h):
        h = h.lstrip("#")
        chan = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, bl = (f(c) for c in chan)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_dark_palette_covers_every_class():
    """The two palettes are hand-maintained; they must not drift apart."""
    from cbt_kg import ui
    assert set(ui.NODE_COLORS) == set(ui.NODE_COLORS_DARK)
    assert set(ui._COLOR) == set(ui._COLOR_D)      # includes the 'missing' key
    for label, triple in ui.NODE_COLORS_DARK.items():
        assert len(triple) == 3, label
        assert all(c.startswith("#") for c in triple), label


def test_dark_palette_is_readable_on_a_dark_panel():
    """A dark node must show its text and still be findable against the panel."""
    from cbt_kg import ui
    panel = "#181c22"                      # --c-panel in the canvas dark block
    for label, (fill, ring, text) in ui.NODE_COLORS_DARK.items():
        assert _contrast(text, fill) >= 4.5, f"{label}: label text too faint"
        assert _contrast(ring, panel) >= 2.0, f"{label}: ring invisible on panel"


def test_legend_swatches_are_tagged_for_retheming():
    """redrawLegend() finds swatches by data-label; untagged ones stay light."""
    from cbt_kg import ui
    html = ui._legend_html()
    assert html.count("data-label=") == html.count('class="leg"') > 0


def test_edit_event_never_renders_a_pending_bubble():
    """The chatbot must not be an output of the inline-edit event.

    Gradio draws a pending bubble under the last message whenever the Chatbot is
    an output of a running event — regardless of show_progress or how fast the
    handler returns. That was the spinner appearing under the therapist's reply
    after pressing cancel. The chat is refreshed indirectly by edit_token.change,
    and a cancel returns gr.skip() for that token so nothing fires.
    """
    import gradio as gr
    from cbt_kg import ui

    cfg = ui.demo.get_config_file()
    comp = {c["id"]: (c.get("props") or {}).get("elem_id") for c in cfg["components"]}
    chat_id = next(i for i, e in comp.items() if e == "therapy_chat")
    token_id = next(i for i, e in comp.items() if e == "edit_token")

    edit_deps = [d for d in cfg["dependencies"]
                 if any(ev == "edit" for _, ev in (d.get("targets") or []))]
    assert len(edit_deps) == 1
    assert chat_id not in edit_deps[0]["outputs"], (
        "chatbot is an output of .edit again — that re-introduces the pending "
        "bubble on cancel")

    refresh = [d for d in cfg["dependencies"]
               if any(c == token_id for c, _ in (d.get("targets") or []))]
    assert refresh and chat_id in refresh[0]["outputs"], (
        "nothing refreshes the chat after a real rewrite")

    # and a cancel must skip the token, or the refresh fires anyway
    session = ui._new_session()
    ui._bot_respond("I felt anxious at work", [], session, "none")
    chat = ui._history_to_chat(session)
    idx = next(i for i, m in enumerate(chat) if m["role"] == "user")
    text = chat[idx]["content"]
    out = ui._on_edit_message(session, chat, "none",
                              gr.EditData(None, {"index": idx,
                                                 "previous_value": text,
                                                 "value": text}))
    assert isinstance(out[-1], type(gr.skip())), "cancel must gr.skip() the token"


def test_shell_css_has_no_hardcoded_light_colours():
    """Every colour in the app rules must come from a variable.

    The dark theme works by flipping one variable block, so any literal left in a
    rule stays light in dark mode — which is exactly how the selected tab kept a
    white background behind it.
    """
    import re
    from cbt_kg import ui

    css = ui._UI_CSS
    rules = css[css.index("body, .gradio-container"):]
    hexes = re.findall(r"#[0-9a-fA-F]{3,6}", rules)
    assert hexes == [], f"hardcoded colours in shell rules: {sorted(set(hexes))}"
    # rgba is allowed only for focus rings / shadows, never as a surface colour
    surfaces = re.findall(r"(?:background|color)\s*:[^;]*rgba\([^)]*\)", rules)
    assert surfaces == [], f"hardcoded surface colours: {surfaces}"


def test_theme_variable_blocks_stay_in_sync():
    """Light and dark must define the same variables, or dark inherits a light one."""
    import re
    from cbt_kg import ui

    css = ui._UI_CSS
    light = css[css.index(":root {"):css.index("}", css.index(":root {"))]
    dark_start = css.index("html.dark, body.dark")
    dark = css[dark_start:css.index("}", dark_start)]
    names = lambda block: {m.group(1) for m in re.finditer(r"(--[a-z-]+)\s*:", block)}
    missing = names(light) - names(dark)
    assert not missing, f"dark block is missing: {sorted(missing)}"
