"""Stable contract shared by every module.

Only factory.py may import concrete implementations. Every other module
imports only from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class GraphNode:
    """A node in the V4_flat CBT knowledge graph."""
    node_id: str
    label: str                                  # V4_flat class name
    props: dict
    status: str = "found"                       # "found" | "missing"
    evidence: list[int] = field(default_factory=list)
    turn_acquired: int | None = None


@dataclass
class GraphEdge:
    """A directed edge in the V4_flat CBT knowledge graph."""
    subject_id: str
    predicate: str
    object_id: str
    props: dict = field(default_factory=dict)   # e.g. {"reportedIntensity": "8/10"}
    status: str = "found"                       # "found" | "missing"
    evidence: list[int] = field(default_factory=list)
    turn_acquired: int | None = None

    @property
    def edge_id(self) -> str:
        return f"{self.subject_id}__{self.predicate}__{self.object_id}"


@runtime_checkable
class Schema(Protocol):
    def node_classes(self) -> list[dict]: ...
    def edge_map(self) -> list[tuple[str, str, str]]: ...
    def subject_edges(self) -> dict[str, list[tuple[str, str]]]: ...
    def anchor_families(self) -> dict[str, list[tuple[str, str, str]]]: ...
    def class_definitions(self) -> dict[str, str]: ...
    def render_ontology(self) -> str: ...


@runtime_checkable
class GraphStore(Protocol):
    def reset(self) -> None: ...
    def upsert_node(self, label: str, props: dict, turn_index: int) -> GraphNode: ...
    def merge_into(self, existing_id: str, props: dict, turn_index: int) -> GraphNode: ...
    def record_utterance(self, turn_index: int, speaker: str,
                         text: str) -> GraphNode: ...
    def add_edge(self, subj_id: str, predicate: str, obj_id: str,
                 props: dict | None = None,
                 evidence: list[int] | None = None) -> GraphEdge: ...
    def nodes(self) -> list[GraphNode]: ...
    def edges(self) -> list[GraphEdge]: ...
    def count_found(self, label: str) -> int: ...
    def cbt_context(self) -> str: ...
    def apply_session_state(self, phase: str, technique: str) -> None: ...
    def snapshot(self) -> dict: ...
    def cytoscape(self) -> dict: ...


@runtime_checkable
class Extractor(Protocol):
    def process_turn(self, client_msg: str, window: list[tuple[str, str]],
                     graph: GraphStore, turn_index: int) -> dict: ...
    def consolidate(self, transcript: list[tuple[int, str, str]],
                    graph: GraphStore) -> dict: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, system: str, history: list[tuple[str, str]]) -> dict: ...


@runtime_checkable
class GraphReader(Protocol):
    def load(self) -> tuple[list[GraphNode], list[GraphEdge]]: ...
    def label(self) -> str: ...
