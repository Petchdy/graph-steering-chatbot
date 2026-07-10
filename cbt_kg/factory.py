"""The ONLY file allowed to import concrete implementations.
Env vars select which backend is wired in at construction time.
"""

from __future__ import annotations

import os

from .interfaces import Extractor, Generator, GraphReader, GraphStore, Schema

from .ontology import CBTSchema
from .graph_memory import InMemoryGraphStore
from .extract import StubExtractor, TurnPipeline
from .generate import EchoGenerator, LocalLLMGenerator, OpenRouterGenerator
from .graph_reader import JsonGraphReader, LiveGraphReader, Neo4jGraphReader
from .query import QueryEngine


def make_schema() -> Schema:
    return CBTSchema()


def make_graph(schema: Schema, session_id: str = "default") -> GraphStore:
    backend = os.environ.get("GRAPH_BACKEND", "memory")
    if backend == "neo4j":
        from .graph_neo4j import Neo4jGraphStore
        return Neo4jGraphStore(
            schema,
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
            session_id=session_id,
        )
    return InMemoryGraphStore(schema)


def make_extractor() -> Extractor:
    kind = os.environ.get("EXTRACTOR", "local")
    if kind == "local":
        return TurnPipeline(
            model=os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            language=os.environ.get("EXTRACT_LANGUAGE", "English"),
        )
    return StubExtractor()


def make_generator() -> Generator:
    kind = os.environ.get("GENERATOR", "local")
    if kind == "openrouter":
        return OpenRouterGenerator(
            model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6"),
        )
    if kind == "steered":
        # Steering overlay: reply steered by the local HF service.
        # STEER_NO_OLLAMA=1 (GPU-lean single-model mode): no Ollama at all — the one HF model behind
        # the steering service produces both steered and baseline replies; technique/phase default
        # (therapy.py advances phases deterministically). Point the extractor at the same service via
        # OLLAMA_HOST=<steer_url> so the CBT graph keeps working on one model.
        # Otherwise (default): two-model mode — CBT technique/phase come from Ollama.
        from steering.steered_generator import SteeredRemoteGenerator
        no_ollama = os.environ.get("STEER_NO_OLLAMA", "0") == "1"
        fallback = None if no_ollama else LocalLLMGenerator(
            model=os.environ.get("LOCAL_LLM_MODEL",
                                 os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink")),
            base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        )
        return SteeredRemoteGenerator(
            steer_url=os.environ.get("STEER_URL", "http://localhost:8100"),
            fallback=fallback,
            default_strategy=os.environ.get("STEER_DEFAULT_STRATEGY", "none"),
        )
    if kind == "local":
        return LocalLLMGenerator(
            model=os.environ.get("LOCAL_LLM_MODEL",
                                  os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink")),
            base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        )
    return EchoGenerator()


def make_query_engine() -> QueryEngine:
    return QueryEngine(
        generator=None,           # use plain ollama for narration
        model=os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink"),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )


# ─────────────────────── Part 2 — graph readers ────────────────────────────

def make_reader_live(graph: GraphStore, label: str = "Live session") -> GraphReader:
    return LiveGraphReader(graph, label=label)


def make_reader_json(path: str) -> GraphReader:
    return JsonGraphReader(path)


def make_reader_neo4j(uri: str | None = None, user: str | None = None,
                       password: str | None = None) -> GraphReader:
    return Neo4jGraphReader(
        uri=uri or os.environ["NEO4J_URI"],
        user=user or os.environ["NEO4J_USER"],
        password=password or os.environ["NEO4J_PASSWORD"],
    )
