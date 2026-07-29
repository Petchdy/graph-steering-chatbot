"""Graph repair (fix → back to the conversation) and client-turn branching."""

from __future__ import annotations

import pytest

from cbt_kg import factory
from cbt_kg.interfaces import GraphNode
from cbt_kg.therapy import (Session, edit_turn, editable_turns, list_branches,
                            switch_branch, turn)


def _session() -> Session:
    schema = factory.make_schema()
    s = Session(schema=schema, graph=factory.make_graph(schema),
                extractor=factory.make_extractor(),
                generator=factory.make_generator())
    s.graph.reset()
    return s


# ── Graph repair ──────────────────────────────────────────────────────────

def test_update_node_feeds_the_next_reply():
    """The point of the round-trip: a correction changes the therapist's context."""
    s = _session()
    turn(s, "I froze in the meeting.")
    node = s.graph.upsert_node("Problem", {"description": "fear of spiders"}, 1)
    assert "fear of spiders" in s.graph.cbt_context()
    s.graph.update_node(node.node_id, props={"description": "performance anxiety"})
    assert "performance anxiety" in s.graph.cbt_context()
    assert "fear of spiders" not in s.graph.cbt_context()


def test_delete_node_takes_its_edges_with_it():
    s = _session()
    a = s.graph.upsert_node("Situation", {"description": "meeting"}, 1)
    b = s.graph.upsert_node("AutomaticThought", {"content": "I will fail"}, 1)
    s.graph.add_edge(a.node_id, "triggers", b.node_id)
    assert s.graph.delete_node(a.node_id) is True
    assert all(e.subject_id != a.node_id and e.object_id != a.node_id
               for e in s.graph.edges())
    assert s.graph.delete_node("nope_1") is False


def test_replace_all_keeps_new_ids_from_colliding():
    """Counters must be rebuilt from incoming ids or extraction overwrites them."""
    s = _session()
    s.graph.replace_all(
        [GraphNode(node_id="prob_7", label="Problem", props={"description": "x"})], [])
    fresh = s.graph.upsert_node("Problem", {"description": "second problem"}, 1)
    assert fresh.node_id != "prob_7"
    assert {n.node_id for n in s.graph.nodes()} == {"prob_7", fresh.node_id}


def test_conversation_continues_after_replace_all():
    s = _session()
    turn(s, "I froze in the meeting.")
    s.graph.replace_all(
        [GraphNode(node_id="prob_1", label="Problem",
                   props={"description": "performance anxiety"})], [])
    result = turn(s, "That's closer to it.")
    assert s.turn_count == 2
    assert result["reply"]
    assert "performance anxiety" in s.graph.cbt_context()


# ── Branching ─────────────────────────────────────────────────────────────

def test_edit_turn_replays_against_original_context():
    s = _session()
    turn(s, "I froze in the meeting.")
    turn(s, "It happened again Tuesday.")
    edit_turn(s, 2, "Actually Tuesday went well.")
    assert s.turn_count == 2                      # replaced, not appended
    assert [h[0] for h in s.history] == ["I froze in the meeting.",
                                         "Actually Tuesday went well."]


def test_both_versions_are_kept_and_switchable():
    s = _session()
    turn(s, "I froze in the meeting.")
    turn(s, "It happened again Tuesday.")
    edit_turn(s, 2, "Actually Tuesday went well.")
    branches = list_branches(s)
    assert len(branches) == 2
    original = next(b for b in branches if "happened again" in b["message"])
    switch_branch(s, original["id"])
    assert s.history[-1][0] == "It happened again Tuesday."
    assert next(b["active"] for b in list_branches(s) if b["id"] == original["id"])


def test_switching_restores_the_graph_too_not_just_the_text():
    s = _session()
    turn(s, "I froze in the meeting.")
    turn(s, "It happened again Tuesday.")
    edit_turn(s, 2, "Actually Tuesday went well.")

    def utterances():
        return {n.props.get("text") for n in s.graph.nodes()
                if n.label == "Utterance" and n.status == "found"}

    assert "Actually Tuesday went well." in utterances()
    original = next(b for b in list_branches(s) if "happened again" in b["message"])
    switch_branch(s, original["id"])
    assert "It happened again Tuesday." in utterances()
    assert "Actually Tuesday went well." not in utterances()


def test_edit_turn_rejects_a_turn_with_no_checkpoint():
    s = _session()
    turn(s, "only one turn")
    with pytest.raises(ValueError, match="no checkpoint"):
        edit_turn(s, 99, "never happened")


def test_editable_turns_tracks_the_conversation():
    s = _session()
    assert editable_turns(s) == []
    turn(s, "one")
    turn(s, "two")
    assert editable_turns(s) == [1, 2]


# ── Inline chat edit (pencil button next to copy) ─────────────────────────

def test_message_index_maps_to_the_right_turn():
    """The chat opens with the therapist's INTRO, so positions are offset by one."""
    from cbt_kg.ui import _message_index_to_turn
    chat = [
        {"role": "assistant", "content": "intro"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "reply two"},
    ]
    assert _message_index_to_turn(chat, 1) == 1
    assert _message_index_to_turn(chat, 3) == 2
    assert _message_index_to_turn(chat, (3, 0)) == 2   # multimodal tuple index


def test_message_index_survives_a_missing_assistant_reply():
    """A turn whose generation failed contributes no assistant message."""
    from cbt_kg.ui import _message_index_to_turn
    chat = [
        {"role": "assistant", "content": "intro"},
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "reply two"},
    ]
    assert _message_index_to_turn(chat, 2) == 2


def test_inline_edit_rewrites_that_turn():
    import gradio as gr
    from cbt_kg import ui
    s = _session()
    turn(s, "I froze in the meeting.")
    turn(s, "It happened again Tuesday.")
    chat = ui._history_to_chat(s)
    data = gr.EditData(None, {"index": 3,
                              "previous_value": "It happened again Tuesday.",
                              "value": "Actually Tuesday went fine."})
    _chat, s, _bar, _gh, _dd, status = ui._on_edit_message(s, chat, "none", data)
    assert "Replayed turn 2" in status
    assert s.history[-1][0] == "Actually Tuesday went fine."
    assert len(list_branches(s)) == 2
