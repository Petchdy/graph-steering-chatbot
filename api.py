"""FastAPI backend. Depends only on interfaces.py and factory.py."""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

import factory
from orchestrator import Session, async_turn

app = FastAPI(title="CACTUS CBT Chatbot")

_sessions: dict[str, Session] = {}

_LABEL_TYPE_MAP = {
    "Session": "session", "Client": "session",
    "Problem": "session_structure", "Goal": "session_structure",
    "Intervention": "session_structure", "Homework": "session_structure",
    "CoreBelief": "cognitive", "IntermediateBelief": "cognitive",
    "Situation": "cognitive", "AutomaticThought": "cognitive",
    "Reaction": "cognitive", "AdaptiveResponse": "cognitive",
    "Utterance": "provenance",
}


def _get_or_create(session_id: str) -> Session:
    if session_id not in _sessions:
        schema = factory.make_schema()
        _sessions[session_id] = Session(
            schema=schema,
            graph=factory.make_graph(schema, session_id=session_id),
            extractor=factory.make_extractor(),
            generator=factory.make_generator(),
        )
    return _sessions[session_id]


def _node_type(label: str) -> str:
    return _LABEL_TYPE_MAP.get(label, "field")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    technique: str = ""
    phase: str = ""
    deltas: dict[str, str]
    slots: dict
    extraction_mode: str = "async"


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session = _get_or_create(request.session_id)
    result = await async_turn(session, request.message)
    return ChatResponse(**result)


@app.post("/reset")
def reset(request: ResetRequest) -> dict:
    session = _sessions.get(request.session_id)
    if session is not None:
        session.graph.reset()
        session.history.clear()
        session.turn_count = 0
    return {"ok": True}


@app.get("/graph/{session_id}")
def get_graph(session_id: str) -> dict:
    """Cytoscape-compatible node/edge JSON from the rich graph."""
    if session_id not in _sessions:
        return {"nodes": [], "edges": []}

    graph = _sessions[session_id].graph
    nodes = []
    edges = []

    for node in graph.nodes():
        ntype = _node_type(node.label) if node.status == "found" else "missing"
        content = (node.props.get("content") or node.props.get("description") or "")[:25]
        label = f"{node.label}\n{content}" if node.status == "found" and content else node.label
        nodes.append({"data": {
            "id": node.node_id,
            "label": label,
            "type": ntype,
            "status": node.status,
        }})

    for edge in graph.edges():
        edges.append({"data": {
            "source": edge.subject_id,
            "target": edge.object_id,
            "label": edge.predicate if edge.status == "found" else "",
            "status": edge.status,
        }})

    return {"nodes": nodes, "edges": edges}


import gradio as gr  # noqa: E402
import ui  # noqa: E402

app = gr.mount_gradio_app(app, ui.demo, path="/")
