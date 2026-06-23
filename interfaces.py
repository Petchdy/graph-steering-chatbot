"""Stable contract shared by every module.

Only factory.py may import concrete implementations. Everyone else depends only
on the Protocols defined here.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OntologyField:
    key: str
    description: str
    priority: int = 1


@dataclass
class GraphNode:
    """A node in the CBT knowledge graph."""
    node_id: str
    label: str
    status: str           # "missing" | "found"
    props: dict
    turn_acquired: int | None = None


@dataclass
class GraphEdge:
    """A directed edge in the CBT knowledge graph."""
    edge_id: str
    predicate: str
    subject_id: str
    object_id: str
    status: str           # "missing" | "found"
    turn_acquired: int | None = None


@runtime_checkable
class Schema(Protocol):
    def fields(self) -> list[OntologyField]: ...
    def render(self) -> str: ...
    def render_ontology(self) -> str: ...
    def node_classes(self) -> list[dict]: ...
    def edge_map(self) -> list[tuple[str, str, str]]: ...
    def subject_edges(self) -> dict[str, list[tuple[str, str]]]: ...


@runtime_checkable
class GraphStore(Protocol):
    def apply_deltas(self, deltas: dict[str, str], turn_id: int) -> None: ...
    def missing(self) -> list[str]: ...
    def acquired_summary(self) -> str: ...
    def snapshot(self) -> dict: ...
    def reset(self) -> None: ...
    def cbt_context(self) -> str: ...
    def apply_session_state(self, phase: str, technique: str) -> None: ...
    def nodes(self) -> list[GraphNode]: ...
    def edges(self) -> list[GraphEdge]: ...
    def upsert_node(self, label: str, props: dict, turn_id: int) -> GraphNode: ...
    def resolve_edge(self, subject_id: str, predicate: str,
                     object_id: str, turn_id: int) -> GraphEdge: ...


@runtime_checkable
class Extractor(Protocol):
    def extract(self, message: str, schema_text: str) -> dict[str, str]: ...
    def extract_nodes(self, message: str, window: list[tuple[str, str]],
                      schema_text: str) -> list[dict]: ...
    def resolve_edges(self, new_node: GraphNode, existing_nodes: list[GraphNode],
                      window_text: str, subject_edges: dict) -> list[tuple[str, str, str]]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, system: str, history: list[tuple[str, str]]) -> dict: ...
