"""Tests for the deterministic query executor + readers."""

from __future__ import annotations

import json
from pathlib import Path

from cbt_kg.graph_memory import InMemoryGraphStore
from cbt_kg.graph_reader import JsonGraphReader, LiveGraphReader
from cbt_kg.ontology import CBTSchema
from cbt_kg.query import execute


def _seeded_graph() -> InMemoryGraphStore:
    g = InMemoryGraphStore(CBTSchema())
    g.upsert_node("Problem", {"description": "exam anxiety", "domain": "academic"}, 1)
    sit = g.upsert_node("Situation", {"description": "exam tomorrow",
                                       "kind": "externalSituation"}, 1)
    at = g.upsert_node("AutomaticThought",
                       {"content": "I will fail", "modality": "verbal",
                        "distortionType": "catastrophizing"}, 1)
    re = g.upsert_node("Reaction", {"content": "anxious", "channel": "emotional",
                                     "valence": "negative"}, 1)
    g.add_edge(sit.node_id, "triggers", at.node_id, evidence=[1])
    g.add_edge(at.node_id, "leadsTo", re.node_id,
               props={"reportedIntensity": "8/10"}, evidence=[1])
    return g


def test_live_reader_emits_canonical_nodes_and_edges():
    g = _seeded_graph()
    nodes, edges = LiveGraphReader(g).load()
    labels = {n.label for n in nodes}
    assert "Situation" in labels
    assert "AutomaticThought" in labels
    preds = {e.predicate for e in edges}
    assert "triggers" in preds
    assert "leadsTo" in preds


def test_execute_count_intent():
    g = _seeded_graph()
    nodes, edges = LiveGraphReader(g).load()
    rs = execute({"intent": "count", "node_labels": ["AutomaticThought"]},
                 nodes, edges)
    assert rs["counts"]["AutomaticThought"] == 1


def test_execute_list_intent_with_property_filter():
    g = _seeded_graph()
    nodes, edges = LiveGraphReader(g).load()
    rs = execute({
        "intent": "list",
        "node_labels": ["AutomaticThought"],
        "property_filters": {"distortionType": "catastrophizing"},
    }, nodes, edges)
    assert rs["intent"] == "list"
    assert len(rs["nodes"]) == 1
    assert rs["nodes"][0]["text"] == "I will fail"


def test_execute_trace_intent_walks_chain():
    g = _seeded_graph()
    nodes, edges = LiveGraphReader(g).load()
    rs = execute({
        "intent": "trace",
        "node_labels": ["Situation"],
        "predicates": ["triggers", "leadsTo"],
    }, nodes, edges)
    labels_walked = {n["label"] for n in rs["nodes"]}
    assert "Situation" in labels_walked
    assert "AutomaticThought" in labels_walked
    assert "Reaction" in labels_walked


def test_execute_summarize_intent():
    g = _seeded_graph()
    nodes, edges = LiveGraphReader(g).load()
    rs = execute({"intent": "summarize"}, nodes, edges)
    assert rs["counts"]["Situation"] >= 1
    assert rs["counts"]["AutomaticThought"] >= 1


def test_json_reader_round_trip(tmp_path: Path):
    """Hand-built V4_flat Stage-5-shaped JSON loads correctly."""
    payload = {
        "meta": {"schema_version": "ontology_v4_flat"},
        "tbox_nodes": [], "tbox_edges": [],
        "nodes": [
            {"id": "client_1", "label": "Client", "parent": None,
             "properties": {}, "evidence": []},
            {"id": "session_1", "label": "Session", "parent": None,
             "properties": {"sessionType": "therapy"}, "evidence": []},
            {"id": "sit_1", "label": "Situation", "parent": None,
             "properties": {"description": "exam tomorrow",
                            "kind": "externalSituation"},
             "evidence": [1]},
            {"id": "at_1", "label": "AutomaticThought", "parent": None,
             "properties": {"content": "I will fail", "modality": "verbal"},
             "evidence": [1]},
        ],
        "edges": [
            {"type": "triggers", "from": "sit_1", "to": "at_1", "evidence": [1]},
        ],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    nodes, edges = JsonGraphReader(str(path)).load()
    ids = {n.node_id for n in nodes}
    assert {"sit_1", "at_1"}.issubset(ids)
    preds = {e.predicate for e in edges}
    assert preds == {"triggers"}


# ── Dialogue provenance: Utterance nodes -> quotable evidence ──────────────

def _graph_with_dialogue() -> InMemoryGraphStore:
    g = _seeded_graph()
    g.record_utterance(1, "client", "I have an exam tomorrow and I will fail.")
    g.record_utterance(1, "therapist", "What goes through your mind then?")
    return g


def test_record_utterance_materializes_dialogue_nodes():
    g = _graph_with_dialogue()
    utts = [n for n in g.nodes() if n.label == "Utterance" and n.status == "found"]
    assert {u.node_id for u in utts} == {"utt_1", "utt_1_t"}
    client = next(u for u in utts if u.node_id == "utt_1")
    assert client.props["text"].startswith("I have an exam")
    assert client.props["speaker"] == "client"
    assert client.props["turnIndex"] == 1


def test_evidence_turn_indices_resolve_to_quotes():
    nodes, edges = LiveGraphReader(_graph_with_dialogue()).load()
    rs = execute({"intent": "list", "node_labels": ["AutomaticThought"]}, nodes, edges)
    quotes = rs["nodes"][0]["evidence_quotes"]
    assert [q["speaker"] for q in quotes] == ["client", "therapist"]
    assert "exam tomorrow" in quotes[0]["text"]


def test_utterances_excluded_from_unlabelled_list_but_available_by_label():
    nodes, edges = LiveGraphReader(_graph_with_dialogue()).load()
    unlabelled = execute({"intent": "list", "node_labels": []}, nodes, edges)
    assert "Utterance" not in {n["label"] for n in unlabelled["nodes"]}
    asked = execute({"intent": "list", "node_labels": ["Utterance"]}, nodes, edges)
    assert {n["label"] for n in asked["nodes"]} == {"Utterance"}


def test_summarize_carries_transcript_in_spoken_order():
    nodes, edges = LiveGraphReader(_graph_with_dialogue()).load()
    rs = execute({"intent": "summarize"}, nodes, edges)
    assert [t["speaker"] for t in rs["transcript"]] == ["client", "therapist"]


def test_graph_without_utterances_degrades_quietly():
    """Sessions recorded before this feature must still answer, just without quotes."""
    nodes, edges = LiveGraphReader(_seeded_graph()).load()
    rs = execute({"intent": "list", "node_labels": ["AutomaticThought"]}, nodes, edges)
    assert rs["nodes"] and "evidence_quotes" not in rs["nodes"][0]


def test_qualified_property_filter_keys_are_normalized():
    """The parse prompt lists enums as `Class.prop`; unnormalized those match nothing."""
    from cbt_kg.query import _normalize_filters
    assert _normalize_filters({"AutomaticThought.distortionType": "catastrophizing"}) == \
        {"distortionType": "catastrophizing"}
    nodes, edges = LiveGraphReader(_seeded_graph()).load()
    rs = execute({"intent": "list", "node_labels": ["AutomaticThought"],
                  "property_filters": _normalize_filters(
                      {"AutomaticThought.distortionType": "catastrophizing"})},
                 nodes, edges)
    assert len(rs["nodes"]) == 1


def test_dialogue_survives_json_round_trip(tmp_path):
    nodes, _ = LiveGraphReader(_graph_with_dialogue()).load()
    payload = {"nodes": [{"id": n.node_id, "label": n.label,
                          "properties": n.props, "evidence": n.evidence}
                         for n in nodes], "edges": []}
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rn, re_ = JsonGraphReader(str(path)).load()
    rs = execute({"intent": "list", "node_labels": ["AutomaticThought"]}, rn, re_)
    assert "exam tomorrow" in rs["nodes"][0]["evidence_quotes"][0]["text"]
