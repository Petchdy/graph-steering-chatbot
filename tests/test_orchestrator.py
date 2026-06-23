"""Proves the turn loop works with zero external services:
InMemoryGraphStore + StubExtractor + EchoGenerator."""

import asyncio

from extract import StubExtractor
from generate import EchoGenerator
from graph import InMemoryGraphStore
from orchestrator import Session, turn, async_turn, validate_phase
from schema import CBTSchema, CBT_NODE_CLASSES

CLINICAL_FIELDS = [
    "presenting_problem", "emotion", "negative_thought", "cognitive_pattern",
    "trigger_situation", "physical_symptoms", "past_coping",
    "reframe_attempt", "coping_strategies",
]


def _make_session() -> Session:
    schema = CBTSchema()
    return Session(
        schema=schema,
        graph=InMemoryGraphStore(schema),
        extractor=StubExtractor(),
        generator=EchoGenerator(),
    )


def test_turn_fills_graph_and_shrinks_missing():
    session = _make_session()

    assert session.graph.missing() == CLINICAL_FIELDS

    result1 = turn(session, "presenting_problem: work stress")
    assert result1["deltas"] == {"presenting_problem": "work stress"}
    assert result1["slots"]["presenting_problem"]["acquired"] is True

    missing_after_1 = session.graph.missing()
    assert "presenting_problem" not in missing_after_1
    assert len(missing_after_1) == len(CLINICAL_FIELDS) - 1

    result2 = turn(session, "emotion: anxious")
    missing_after_2 = session.graph.missing()
    assert "emotion" not in missing_after_2
    assert len(missing_after_2) < len(missing_after_1)

    for result in (result1, result2):
        assert isinstance(result["reply"], str) and result["reply"]


def test_turn_ignores_unknown_fields():
    session = _make_session()
    result = turn(session, "unknown_field: some value")
    assert result["deltas"] == {}
    assert session.graph.missing() == CLINICAL_FIELDS


def test_reset_clears_acquired_fields():
    session = _make_session()
    turn(session, "presenting_problem: loneliness")
    assert "presenting_problem" not in session.graph.missing()

    session.graph.reset()
    assert session.graph.missing() == CLINICAL_FIELDS


def test_session_state_not_in_missing():
    session = _make_session()
    missing = session.graph.missing()
    assert "session_phase" not in missing
    assert "active_technique" not in missing


def test_session_state_updates():
    session = _make_session()
    result = turn(session, "presenting_problem: exam anxiety")
    # Not enough turns/fields yet for Exploration — should stay Rapport
    assert result["phase"] == "Rapport"
    snap = session.graph.snapshot()
    assert snap["session_phase"]["value"] == "Rapport"
    assert snap["active_technique"]["acquired"] is True


def test_validate_phase_enforces_minimums():
    snap_empty = {f: {"acquired": False} for f in CLINICAL_FIELDS}
    snap_with_problem = {**snap_empty, "presenting_problem": {"acquired": True}}
    snap_with_thought = {
        **snap_with_problem,
        "negative_thought": {"acquired": True},
    }

    # Can't advance to Exploration without presenting_problem
    assert validate_phase("Exploration", "Rapport", snap_empty, 5) == "Rapport"

    # Can advance to Exploration with presenting_problem + 2 turns
    assert validate_phase("Exploration", "Rapport", snap_with_problem, 2) == "Exploration"

    # Can't advance to Technique without 5 turns
    assert validate_phase("Technique", "Exploration", snap_with_thought, 4) == "Exploration"

    # Can advance to Technique with both fields + 5 turns
    assert validate_phase("Technique", "Exploration", snap_with_thought, 5) == "Technique"

    # Always allowed to stay in same phase or go back
    assert validate_phase("Rapport", "Exploration", snap_empty, 0) == "Rapport"


# ── New v6 tests ──────────────────────────────────────────────────────────────

def test_async_turn_returns_expected_keys():
    session = _make_session()
    result = asyncio.run(async_turn(session, "presenting_problem: work stress"))
    assert "reply" in result
    assert "phase" in result
    assert "technique" in result
    assert "extraction_mode" in result
    assert result["extraction_mode"] in ("sync", "async")


def test_rich_graph_has_placeholder_nodes():
    session = _make_session()
    nodes = session.graph.nodes()
    node_labels = {n.label for n in nodes}
    for cls in CBT_NODE_CLASSES:
        assert cls["label"] in node_labels, f"Missing placeholder for {cls['label']}"


def test_rich_graph_upsert_creates_found_node():
    session = _make_session()
    node = session.graph.upsert_node(
        "Situation",
        {"description": "exam tomorrow", "kind": "externalSituation"},
        turn_id=1,
    )
    assert node.status == "found"
    assert node.label == "Situation"
    placeholders = [n for n in session.graph.nodes()
                    if n.label == "Situation" and n.status == "missing"]
    assert len(placeholders) == 0


def test_rich_graph_multi_instance():
    session = _make_session()
    n1 = session.graph.upsert_node("Situation", {"description": "exam tomorrow"}, turn_id=1)
    n2 = session.graph.upsert_node("Situation", {"description": "fight with friend"}, turn_id=2)
    assert n1.node_id != n2.node_id
    found = [n for n in session.graph.nodes()
             if n.label == "Situation" and n.status == "found"]
    assert len(found) == 2


def test_resolve_edge_marks_found():
    session = _make_session()
    sit = session.graph.upsert_node("Situation", {"description": "exam"}, 1)
    at  = session.graph.upsert_node("AutomaticThought", {"content": "I will fail"}, 1)
    edge = session.graph.resolve_edge(sit.node_id, "triggers", at.node_id, 1)
    assert edge.status == "found"
    found_edges = [e for e in session.graph.edges() if e.status == "found"]
    assert any(e.predicate == "triggers" for e in found_edges)


def test_extraction_lock_exists():
    session = _make_session()
    assert isinstance(session.extraction_lock, asyncio.Lock)
