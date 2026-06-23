"""Gradio UI — two-column: chat (left) + live Cytoscape graph (right)."""

import json
from dotenv import load_dotenv
load_dotenv()

import gradio as gr

import factory
from orchestrator import Session, turn

_LABEL_TYPE_MAP = {
    "Session": "session", "Client": "session",
    "Problem": "session_structure", "Goal": "session_structure",
    "Intervention": "session_structure", "Homework": "session_structure",
    "CoreBelief": "cognitive", "IntermediateBelief": "cognitive",
    "Situation": "cognitive", "AutomaticThought": "cognitive",
    "Reaction": "cognitive", "AdaptiveResponse": "cognitive",
    "Utterance": "provenance",
}

_CONTENT_KEYS = ("content", "description", "statement", "taskDescription", "text")

INTRO = (
    "Hello, and welcome. I'm glad you're here today. "
    "This is a safe space to talk about whatever is on your mind. "
    "What's been weighing on you lately, or what would you most like to explore today?"
)

CYTOSCAPE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"

NODE_STYLES = json.dumps([
    {"selector": 'node[type="session"]',
     "style": {"background-color": "#2d6a4f", "color": "#fff",
                "label": "data(label)", "text-wrap": "wrap",
                "text-valign": "center", "font-size": "11px",
                "width": 80, "height": 80, "shape": "ellipse"}},
    {"selector": 'node[type="session_structure"]',
     "style": {"background-color": "#74c69d", "label": "data(label)",
                "text-wrap": "wrap", "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "round-rectangle"}},
    {"selector": 'node[type="cognitive"]',
     "style": {"background-color": "#9b72cf", "color": "#fff",
                "label": "data(label)", "text-wrap": "wrap",
                "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "ellipse"}},
    {"selector": 'node[type="provenance"]',
     "style": {"background-color": "#6c757d", "color": "#fff",
                "label": "data(label)", "text-wrap": "wrap",
                "text-valign": "center", "font-size": "10px",
                "width": 60, "height": 60, "shape": "tag"}},
    {"selector": 'node[type="session_state"]',
     "style": {"background-color": "#f4a261", "label": "data(label)",
                "text-wrap": "wrap", "text-valign": "center", "font-size": "10px",
                "width": 70, "height": 70, "shape": "diamond"}},
    {"selector": 'node[type="missing"]',
     "style": {"background-color": "#dee2e6", "label": "data(label)",
                "text-valign": "center", "font-size": "9px",
                "width": 50, "height": 50,
                "border-style": "dashed", "border-color": "#adb5bd", "border-width": 2}},
    {"selector": 'edge[status="found"]',
     "style": {"label": "data(label)", "font-size": "8px", "curve-style": "bezier",
                "target-arrow-shape": "triangle", "line-color": "#74c69d",
                "target-arrow-color": "#74c69d", "arrow-scale": 0.7}},
    {"selector": 'edge[status="missing"]',
     "style": {"curve-style": "bezier", "target-arrow-shape": "triangle",
                "line-color": "#dee2e6", "target-arrow-color": "#dee2e6",
                "line-style": "dashed", "arrow-scale": 0.5}},
])


def _node_type(label: str, status: str) -> str:
    if status == "missing":
        return "missing"
    return _LABEL_TYPE_MAP.get(label, "field")


def _content_preview(props: dict) -> str:
    for key in _CONTENT_KEYS:
        val = props.get(key)
        if val:
            return str(val)[:25]
    return ""


def _build_elements(graph) -> list:
    elements = []
    for n in graph.nodes():
        ntype = _node_type(n.label, n.status)
        preview = _content_preview(n.props)
        label = f"{n.label}\n{preview}" if n.status == "found" and preview else n.label
        elements.append({"data": {
            "id": n.node_id, "label": label, "type": ntype, "status": n.status,
        }})
    for e in graph.edges():
        elements.append({"data": {
            "id": e.edge_id, "source": e.subject_id, "target": e.object_id,
            "label": e.predicate if e.status == "found" else "",
            "status": e.status,
        }})
    return elements


def _render_graph(graph) -> str:
    elements_json = json.dumps(_build_elements(graph))
    styles_json = NODE_STYLES
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<script src="{CYTOSCAPE_CDN}"></script>
</head>
<body style="margin:0; padding:0; background:#fafafa;">
<div id="cy" style="width:100%;height:470px;background:#fafafa;
     border:1px solid #dee2e6;border-radius:8px;box-sizing:border-box;"></div>
<script>
(function() {{
  function init() {{
    var el = document.getElementById('cy');
    if (!el || typeof cytoscape === 'undefined') {{
      setTimeout(init, 100); return;
    }}
    el.innerHTML = '';
    cytoscape({{
      container: el,
      style: {styles_json},
      layout: {{ name: 'cose', animate: false, randomize: false, nodeRepulsion: 8000 }},
      elements: {elements_json}
    }}).fit(undefined, 20);
  }}
  init();
}})();
</script>
</body>
</html>
"""
    import html
    escaped_html = html.escape(html_content)
    return f'<iframe srcdoc="{escaped_html}" style="width:100%; height:480px; border:none; border-radius:8px;"></iframe>'


def _new_session() -> Session:
    schema = factory.make_schema()
    return Session(
        schema=schema,
        graph=factory.make_graph(schema),
        extractor=factory.make_extractor(),
        generator=factory.make_generator(),
    )


def _add_user(message: str, history: list):
    return history + [{"role": "user", "content": message}], "", message


def _bot_respond(message: str, history: list, session: Session):
    if session is None:
        session = _new_session()
    result = turn(session, message)
    history = history + [{"role": "assistant", "content": result["reply"]}]
    graph_html = _render_graph(session.graph)
    return history, session, result["phase"], result["technique"], graph_html


def _reset():
    session = _new_session()
    history = [{"role": "assistant", "content": INTRO}]
    graph_html = _render_graph(session.graph)
    return history, session, "Rapport", "—", graph_html


with gr.Blocks(title="CACTUS CBT Therapy", fill_height=True) as demo:
    session_state = gr.State(None)
    pending_msg = gr.State("")

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.Markdown("## CACTUS CBT Therapy")
            with gr.Row():
                phase_box = gr.Textbox(label="Phase", value="Rapport",
                                       interactive=False, scale=1)
                technique_box = gr.Textbox(label="Technique", value="—",
                                           interactive=False, scale=3)
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Share what's on your mind…",
                                     show_label=False, scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            reset_btn = gr.Button("New session")

        with gr.Column(scale=2):
            gr.Markdown("## Knowledge Graph")
            graph_panel = gr.HTML()

    _outputs = [chatbot, session_state, phase_box, technique_box, graph_panel]

    send_btn.click(
        _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
    ).then(
        _bot_respond, [pending_msg, chatbot, session_state], _outputs
    )
    msg_box.submit(
        _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
    ).then(
        _bot_respond, [pending_msg, chatbot, session_state], _outputs
    )
    reset_btn.click(_reset, [], _outputs)
    demo.load(_reset, [], _outputs)
