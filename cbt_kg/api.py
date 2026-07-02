"""FastAPI backend — Part 1 (therapy) + Part 2 (query) routes.

Depends only on interfaces.py, ontology.py, factory.py, therapy.py.
Sessions and loaded query-graphs are process-local dicts; restart wipes them.
"""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from . import factory
from .graph_memory import cytoscape_render
from .interfaces import GraphEdge, GraphNode
from .therapy import Session, async_turn

app = FastAPI(title="CBT V4_flat Chatbot")

_sessions: dict[str, Session] = {}
_query_graphs: dict[str, tuple[list[GraphNode], list[GraphEdge], str]] = {}


def _get_or_create_session(session_id: str) -> Session:
    if session_id not in _sessions:
        schema = factory.make_schema()
        _sessions[session_id] = Session(
            schema=schema,
            graph=factory.make_graph(schema, session_id=session_id),
            extractor=factory.make_extractor(),
            generator=factory.make_generator(),
        )
    return _sessions[session_id]


# ─────────────────────────── Part 1 — Therapy ────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    strategy: Optional[str] = None   # steering strategy for this turn (GENERATOR=steered); None = leave as-is


class ChatResponse(BaseModel):
    reply: str
    technique: str = ""
    phase: str = ""
    extraction_mode: str = "async"
    new_nodes: list[str] = []
    new_edges: list[list[str]] = []
    graph_snapshot: dict = {}


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session = _get_or_create_session(req.session_id)
    if req.strategy is not None and hasattr(session.generator, "set_strategy"):
        session.generator.set_strategy(req.strategy)   # manual steering button
    result = await async_turn(session, req.message)
    return ChatResponse(
        reply=result["reply"],
        technique=result["technique"],
        phase=result["phase"],
        extraction_mode=result["extraction_mode"],
        new_nodes=result.get("new_nodes", []),
        new_edges=[list(e) for e in result.get("new_edges", [])],
        graph_snapshot=result.get("graph_snapshot", {}),
    )


@app.get("/strategies")
def strategies() -> dict:
    """Offered steering strategies (proxied from the steering service). 'none' always available."""
    import os
    import requests
    try:
        r = requests.get(os.environ.get("STEER_URL", "http://localhost:8100") + "/strategies", timeout=5)
        r.raise_for_status()
        data = r.json()
        return {"strategies": ["none"] + list(data.get("strategies", [])), "alphas": data.get("alphas", {})}
    except Exception:  # noqa: BLE001 — service down / not steered mode
        return {"strategies": ["none"], "alphas": {}}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    session = _sessions.get(req.session_id)
    if session is not None:
        session.graph.reset()
        session.history.clear()
        session.transcript.clear()
        session.turn_count = 0
    return {"ok": True}


@app.get("/graph/{session_id}")
def get_graph(session_id: str) -> dict:
    if session_id not in _sessions:
        return {"nodes": [], "edges": []}
    return _sessions[session_id].graph.cytoscape()


# ─────────────────────────── Part 2 — Query ──────────────────────────────

class LoadGraphLiveRequest(BaseModel):
    source: str = "live"
    session_id: str


class LoadGraphNeo4jRequest(BaseModel):
    source: str = "neo4j"
    uri: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None


class LoadGraphResponse(BaseModel):
    handle: str
    label: str
    counts: dict
    total_nodes: int
    total_edges: int


class QueryRequest(BaseModel):
    handle: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    intent: str = ""
    n_result_nodes: int = 0


def _summarize(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict:
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.label] = counts.get(n.label, 0) + 1
    return counts


@app.post("/load_graph/live", response_model=LoadGraphResponse)
def load_graph_live(req: LoadGraphLiveRequest) -> LoadGraphResponse:
    if req.session_id not in _sessions:
        raise HTTPException(404, "session_id not found")
    reader = factory.make_reader_live(
        _sessions[req.session_id].graph,
        label=f"Live: {req.session_id}",
    )
    nodes, edges = reader.load()
    handle = uuid.uuid4().hex[:12]
    _query_graphs[handle] = (nodes, edges, reader.label())
    return LoadGraphResponse(
        handle=handle, label=reader.label(),
        counts=_summarize(nodes, edges),
        total_nodes=len(nodes), total_edges=len(edges),
    )


@app.post("/load_graph/json", response_model=LoadGraphResponse)
async def load_graph_json(file: UploadFile = File(...)) -> LoadGraphResponse:
    import json
    import tempfile
    raw = await file.read()
    # Write to temp and parse via JsonGraphReader (so the same code path is used).
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    reader = factory.make_reader_json(tmp_path)
    nodes, edges = reader.load()
    handle = uuid.uuid4().hex[:12]
    _query_graphs[handle] = (nodes, edges, f"JSON: {file.filename}")
    return LoadGraphResponse(
        handle=handle, label=f"JSON: {file.filename}",
        counts=_summarize(nodes, edges),
        total_nodes=len(nodes), total_edges=len(edges),
    )


@app.post("/load_graph/neo4j", response_model=LoadGraphResponse)
def load_graph_neo4j(req: LoadGraphNeo4jRequest) -> LoadGraphResponse:
    reader = factory.make_reader_neo4j(
        uri=req.uri, user=req.user, password=req.password,
    )
    nodes, edges = reader.load()
    handle = uuid.uuid4().hex[:12]
    _query_graphs[handle] = (nodes, edges, "Neo4j")
    return LoadGraphResponse(
        handle=handle, label="Neo4j",
        counts=_summarize(nodes, edges),
        total_nodes=len(nodes), total_edges=len(edges),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if req.handle not in _query_graphs:
        raise HTTPException(404, "graph handle not found")
    nodes, edges, _ = _query_graphs[req.handle]
    engine = factory.make_query_engine()
    result = engine.answer(req.question, nodes, edges)
    rs = result.get("result_set", {})
    return QueryResponse(
        answer=result.get("answer", ""),
        intent=rs.get("intent", ""),
        n_result_nodes=len(rs.get("nodes", []) or []),
    )


@app.get("/graph_preview/{handle}")
def graph_preview(handle: str) -> dict:
    if handle not in _query_graphs:
        return {"nodes": [], "edges": []}
    nodes, edges, _ = _query_graphs[handle]
    return cytoscape_render(nodes, edges)


# ─────────────────────────── Mount Gradio UI ────────────────────────────

import gradio as gr  # noqa: E402
from . import ui  # noqa: E402

app = gr.mount_gradio_app(app, ui.demo, path="/")
