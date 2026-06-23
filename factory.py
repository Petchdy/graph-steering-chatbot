"""The ONLY file allowed to import concrete implementations.
Env vars select which backend is wired in at construction time.
"""

import os

from interfaces import Schema, GraphStore, Extractor, Generator
from schema import CBTSchema
from graph import InMemoryGraphStore, Neo4jGraphStore
from extract import StubExtractor, CBTExtractor
from generate import EchoGenerator, LocalLLMGenerator, OpenRouterGenerator


def make_schema() -> Schema:
    return CBTSchema()


def make_graph(schema: Schema, session_id: str = "default") -> GraphStore:
    backend = os.environ.get("GRAPH_BACKEND", "memory")
    if backend == "neo4j":
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
        return CBTExtractor(
            model=os.environ.get("OLLAMA_MODEL", "qwen3.5-nothink"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    return StubExtractor()


def make_generator() -> Generator:
    kind = os.environ.get("GENERATOR", "local")
    if kind == "openrouter":
        return OpenRouterGenerator(
            model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6"),
        )
    if kind == "local":
        return LocalLLMGenerator(
            model=os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:9b"),
            base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        )
    return EchoGenerator()
