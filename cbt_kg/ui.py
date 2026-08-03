"""Gradio UI — two tabs: Therapy (Part 1) and Query (Part 2).

Tab 1: Canvas-based read-only knowledge graph with inspector.
Tab 2: Canvas-based editable knowledge graph with inspector (add/edit/delete nodes+edges).
"""

from __future__ import annotations

import html
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import gradio as gr

from . import factory
from .interfaces import GraphEdge, GraphNode
from .therapy import (Session, turn, edit_turn, editable_turns, list_branches,
                      switch_branch, apply_graph, _client_message_at)

# ─────────────────────────────────────────────────────────────────────────
# Color / style constants
# ─────────────────────────────────────────────────────────────────────────

# The stroke (middle value) is each class's identity colour and is the *original*
# pre-"new design" palette — restored deliberately, so don't re-hue it. Only the
# fill and text were re-derived, as a pale tint / dark shade of that same hue, to
# keep the soft blended look the rest of the canvas uses.
NODE_COLORS: dict[str, tuple[str, str, str]] = {
    # label: (fill, stroke, text)
    "Client":              ("#EFF0F2", "#D1D5DB", "#1F2937"),
    "Session":             ("#EFF0F2", "#D1D5DB", "#1F2937"),
    "Problem":             ("#FDECEC", "#EF4444", "#7F1D1D"),
    "Goal":                ("#E3F7F0", "#10B981", "#065F46"),
    "Intervention":        ("#EFEAFE", "#8B5CF6", "#4C1D95"),
    "Homework":            ("#FEF3DC", "#F59E0B", "#78350F"),
    "CoreBelief":          ("#F7E6EE", "#831843", "#611030"),
    "IntermediateBelief":  ("#FAE7EF", "#9D174D", "#6B0F35"),
    "Situation":           ("#FEF8DC", "#FACC15", "#713F12"),
    "AutomaticThought":    ("#E4F8F0", "#34D399", "#065F46"),
    "Reaction":            ("#FDEDED", "#F87171", "#7F1D1D"),
    "AdaptiveResponse":    ("#E6FAF1", "#6EE7B7", "#065F46"),
    "Utterance":           ("#F1F2F4", "#9CA3AF", "#374151"),
}
_MISSING_COLORS = ("#F8FAFC", "#CBD5E1", "#94A3B8")

# Dark-theme counterpart. Same identity hue per class wherever it still reads on a
# dark ground — only CoreBelief (#831843) and IntermediateBelief (#9D174D) are
# lightened, because those two are near-black maroons that vanish otherwise. Fill
# becomes a deep tint of the hue and text a pale one, mirroring the light palette's
# structure so the two themes feel like one design.
NODE_COLORS_DARK: dict[str, tuple[str, str, str]] = {
    # label: (fill, stroke, text)
    "Client":              ("#242830", "#D1D5DB", "#E5E7EB"),
    "Session":             ("#242830", "#D1D5DB", "#E5E7EB"),
    "Problem":             ("#3A1D1D", "#EF4444", "#FCA5A5"),
    "Goal":                ("#0E2C24", "#10B981", "#6EE7B7"),
    "Intervention":        ("#262046", "#8B5CF6", "#C4B5FD"),
    "Homework":            ("#3A2A0C", "#F59E0B", "#FCD34D"),
    "CoreBelief":          ("#35152A", "#C2417A", "#F0A5C8"),
    "IntermediateBelief":  ("#38162A", "#D4548A", "#F5AFCB"),
    "Situation":           ("#3A3410", "#FACC15", "#FEF08A"),
    "AutomaticThought":    ("#12302A", "#34D399", "#86EFC5"),
    "Reaction":            ("#3B2020", "#F87171", "#FECACA"),
    "AdaptiveResponse":    ("#113329", "#6EE7B7", "#A7F3D0"),
    "Utterance":           ("#24282F", "#9CA3AF", "#D1D5DB"),
}
_MISSING_COLORS_DARK = ("#1F232A", "#3F4753", "#6B7280")

_COLOR = {k: v[1] for k, v in NODE_COLORS.items()}    # stroke
_BADGE_BG = {k: v[0] for k, v in NODE_COLORS.items()}  # fill
_BADGE_COLOR = {k: v[2] for k, v in NODE_COLORS.items()}  # text
_COLOR["missing"] = _MISSING_COLORS[1]

_COLOR_D = {k: v[1] for k, v in NODE_COLORS_DARK.items()}
_BADGE_BG_D = {k: v[0] for k, v in NODE_COLORS_DARK.items()}
_BADGE_COLOR_D = {k: v[2] for k, v in NODE_COLORS_DARK.items()}
_COLOR_D["missing"] = _MISSING_COLORS_DARK[1]

_PREDICATES = [
    "triggers", "leadsTo", "stemsFrom", "givesRiseTo",
    "influencesPerceptionOf", "manifestsAs", "becomesSituation",
    "reinforces", "hasAdaptiveResponse", "associatedWith",
    "targetsProblem", "targets", "appliedTo", "produces",
    "hasSession", "hasProblem", "hasIntervention", "hasHomework",
    "evidencedBy", "inSession",
]

_NODE_CLASSES = [
    "Problem", "Goal", "Intervention", "Homework",
    "CoreBelief", "IntermediateBelief", "Situation",
    "AutomaticThought", "Reaction", "AdaptiveResponse",
    "Client", "Session",
]

# ─────────────────────────────────────────────────────────────────────────
# Demo script — a client side that yields a complete, correct graph
# ─────────────────────────────────────────────────────────────────────────
#
# Ordered deliberately, because the pipeline can only draw an edge when BOTH
# endpoints already exist and a node may only be a relation subject on the turn
# it is created (plus the bounded orphan retry). So the concrete Situation →
# AutomaticThought → Reaction chain comes first and the beliefs it feeds come
# after, giving `stemsFrom` / `reinforces` real objects to attach to.
#
# Each message is written to be extractable: one clear unit per class, in the
# client's own voice, with the distortion stated rather than implied.
# Intervention / Homework / Goal nodes are only ever created by Tier B's
# session-level pass, which fires every CONSOLIDATE_EVERY (6) turns — so the
# graph is not complete until turn 6, and Consolidation phase additionally
# needs an AdaptiveResponse plus 12 turns.

DEMO_SCRIPT: list[str] = [
    # 1 — Situation + AutomaticThought + Reaction, one concrete episode
    "Last Tuesday in the team meeting my manager asked me a direct question and "
    "my mind went completely blank. I thought 'everyone can see I'm a fraud'. "
    "My face burned and I just froze.",
    # 2 — a second instance, behavioural Reaction (gives the pattern something to generalise from)
    "It happens most weeks now. Yesterday I stayed quiet in the standup even "
    "though I knew the answer, because I was sure I'd say something stupid.",
    # 3 — CoreBelief (self / worthless), now that the evidence for it exists
    "Underneath it I think I've always believed I'm not good enough. Like I got "
    "this job by luck and sooner or later they'll find me out.",
    # 4 — IntermediateBelief: ONE rule. Stating a rule and a should-statement
    #     together makes Stage 1.2 atomize them into near-duplicate IB nodes.
    "My rule is that if I don't answer perfectly, people will decide I'm "
    "incompetent.",
    # 5 — Goal only. The Problem is already stated in turn 1; restating it here
    #     produces a second Problem node that Jaccard merging will not catch.
    "What I want is to speak up in meetings without my heart racing, even if "
    "what I say isn't perfect.",
    # 6 — the session wrap-up, and the LAST turn that matters for completeness:
    #     Tier B fires here (turn_count % CONSOLIDATE_EVERY == 0) and reads the
    #     transcript *so far*, so AdaptiveResponse / Intervention / Homework must
    #     already have been said. Left to turn 7 they would not be consolidated
    #     until turn 12, which is why this turn carries all three.
    #     The reframe is phrased as the balanced alternative itself, rather than
    #     as recollection, which reads as just another AutomaticThought.
    "A more balanced way to see it is this: going blank in one meeting doesn't "
    "make me a fraud, it means I was caught off guard — I have answered plenty "
    "of hard questions well before. In today's session we worked through a "
    "thought record together, and I'll keep filling one in after each meeting "
    "this week.",
]


INTRO = (
    "Hello, and welcome. I'm glad you're here today. "
    "This is a safe space to talk about whatever is on your mind. "
    "What's been weighing on you lately, or what would you most like to explore today?"
)

# ─────────────────────────────────────────────────────────────────────────
# Data conversion helper
# ─────────────────────────────────────────────────────────────────────────

# Never drawn on either canvas: Client/Session are deterministic scaffold and
# Utterance is raw provenance — none of them are things the model *found*. They
# stay in the graph itself, and remain editable in Tab 2's repair dropdowns.
SCAFFOLD_LABELS = ("Client", "Session")

# Tab 1 is sized to land inside a single ~700px viewport: the left column's
# stacked controls and the right column's canvas finish at about the same height.
# Tab 2 keeps the taller canvas — it is a working surface, not a dashboard.
THERAPY_CHAT_H = 300
THERAPY_CANVAS_H = 520

# Hidden from the therapy panel's canvas *and* its legend. Part 2 still draws
# and lists all three, so its legend keeps the original full set.
THERAPY_HIDDEN_LABELS = {"Client", "Session", "Utterance"}


def _build_canvas_data(
    graph_nodes: list[GraphNode],
    graph_edges: list[GraphEdge],
    skip_utterance: bool = True,
    skip_scaffold: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Convert GraphNode/GraphEdge lists to canvas-friendly dicts.

    Filters out Utterance nodes (too noisy) and edges where BOTH endpoints
    are missing (placeholder-only noise at startup).

    `skip_scaffold` additionally drops Client/Session. They are never extracted
    from what the client says — `_structure_edges` creates them deterministically
    — so on the therapy panel they are just fixed furniture that implies the model
    found something. Tab 2 keeps them, since the structural edges hanging off them
    (hasSession / hasProblem / hasIntervention) are legitimately editable there.
    """
    filtered_nodes = [
        n for n in graph_nodes
        if not (skip_utterance and n.label == "Utterance")
        and not (skip_scaffold and n.label in SCAFFOLD_LABELS)
    ]
    node_ids = {n.node_id for n in filtered_nodes}
    node_status = {n.node_id: n.status for n in filtered_nodes}

    canvas_nodes = []
    for n in filtered_nodes:
        canvas_nodes.append({
            "id": n.node_id,
            "label": n.label,
            "x": 0,
            "y": 0,
            "status": n.status,
            "props": n.props,
            "evidence": n.evidence,
        })

    canvas_edges = []
    for e in graph_edges:
        if e.subject_id not in node_ids or e.object_id not in node_ids:
            continue
        if e.status != "found":
            continue                  # suppress placeholder edges (§UI-1.1 §2.2)
        canvas_edges.append({
            "id": e.edge_id,
            "from": e.subject_id,
            "to": e.object_id,
            "predicate": e.predicate,
            "status": e.status,
            "props": e.props,
            "evidence": e.evidence,
        })

    return canvas_nodes, canvas_edges


# ─────────────────────────────────────────────────────────────────────────
# Canvas HTML template (NOT an f-string — uses __PLACEHOLDER__ substitution)
# ─────────────────────────────────────────────────────────────────────────

# One shared template; __EDIT_MODE__ switches edit buttons on/off in JS.
_CANVAS_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
/* Both themes are expressed as one variable set; the drawing code reads the
   matching node palette (see THEME/PALETTE in the script). The iframe cannot see
   Gradio's .dark class on the parent page, so the theme arrives two ways:
   prefers-color-scheme as the default, and an explicit html[data-theme] that the
   parent sets via postMessage — the latter wins, because Gradio's toggle can
   disagree with the OS. */
:root {
  --c-app: #f6f8fb;        --c-panel: #ffffff;      --c-panel-soft: #fbfdff;
  --c-border: #dbe3ee;     --c-border-soft: #e2e8f0;
  --c-line: #d5deea;       --c-line-hover: #b8c7d9;
  --c-text: #273142;       --c-text-strong: #1f2a37;
  --c-btn-text: #44546a;   --c-btn-hover: #eef4fb;
  --c-muted: #64748b;      --c-subtle: #94a3b8;     --c-chip: #eef2f7;
  --c-grid: rgba(148,163,184,0.18);
  --c-legend-bg: rgba(255,255,255,0.96);
  --c-shadow: rgba(15,23,42,0.06);
  --c-danger: #b4232a;   --c-danger-line: #f3b9be;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --c-app: #12151a;      --c-panel: #181c22;      --c-panel-soft: #1d222a;
    --c-border: #2a313b;   --c-border-soft: #262d36;
    --c-line: #333c48;     --c-line-hover: #47525f;
    --c-text: #e5e9ef;     --c-text-strong: #f3f5f8;
    --c-btn-text: #c2cad6; --c-btn-hover: #232a33;
    --c-muted: #94a3b8;    --c-subtle: #6b7787;     --c-chip: #232a33;
    --c-grid: rgba(148,163,184,0.14);
    --c-legend-bg: rgba(24,28,34,0.96);
    --c-shadow: rgba(0,0,0,0.35);
    --c-danger: #f98a90;   --c-danger-line: #6d2a2f;
  }
}
:root[data-theme="dark"] {
  --c-app: #12151a;      --c-panel: #181c22;      --c-panel-soft: #1d222a;
  --c-border: #2a313b;   --c-border-soft: #262d36;
  --c-line: #333c48;     --c-line-hover: #47525f;
  --c-text: #e5e9ef;     --c-text-strong: #f3f5f8;
  --c-btn-text: #c2cad6; --c-btn-hover: #232a33;
  --c-muted: #94a3b8;    --c-subtle: #6b7787;     --c-chip: #232a33;
  --c-grid: rgba(148,163,184,0.14);
  --c-legend-bg: rgba(24,28,34,0.96);
  --c-shadow: rgba(0,0,0,0.35);
  --c-danger: #f98a90;   --c-danger-line: #6d2a2f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; background: var(--c-app); }
.shell { display: flex; flex-direction: column; height: __SHELL_H__px; background: var(--c-panel); overflow: hidden; border: 1px solid var(--c-border); border-radius: 8px; }
.graph-header { display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--c-border-soft); background: var(--c-panel-soft); flex-shrink: 0; }
.graph-title { font-size: 12px; font-weight: 650; color: var(--c-text); letter-spacing: 0; white-space: nowrap; }
.graph-actions { display: flex; gap: 6px; align-items: center; }
.btn-sm { font-size: 11px; font-weight: 600; padding: 5px 9px; border-radius: 6px;
  border: 1px solid var(--c-line); background: var(--c-panel); cursor: pointer; color: var(--c-btn-text); box-shadow: 0 1px 1px var(--c-shadow); }
.btn-sm:hover { background: var(--c-btn-hover); border-color: var(--c-line-hover); color: var(--c-text-strong); }
.btn-sm.primary { background: #2f7dd1; color: var(--c-panel); border-color: #2f7dd1; }
.btn-sm.primary:hover { background: #2465aa; }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #24A47A;
  display: inline-block; margin-right: 5px; vertical-align: middle; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.workspace { display: flex; flex: 1; overflow: hidden; position: relative; }
.graph-panel { flex: 1; display: flex; flex-direction: column; position: relative; background: radial-gradient(circle at 18px 18px, var(--c-grid) 1px, transparent 1px) 0 0/28px 28px, var(--c-panel); }
canvas { position: absolute; top: 0; left: 0; cursor: pointer; }
.legend { padding: 8px 14px; border-top: 1px solid var(--c-border-soft);
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  flex-shrink: 0; background: var(--c-legend-bg); margin-top: auto; }
.leg { display: flex; align-items: center; gap: 5px; font-size: 10px; color: var(--c-muted); white-space: nowrap; }
.ld { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.detail-panel { width: 240px; flex-shrink: 0; display: flex; flex-direction: column;
  border-left: 1px solid var(--c-border-soft); background: var(--c-panel-soft); overflow-y: auto; }
.dp-header { padding: 11px 14px 9px; border-bottom: 1px solid var(--c-border-soft);
  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.dp-title { font-size: 12px; font-weight: 650; color: var(--c-text); }
.dp-close { font-size: 15px; color: var(--c-subtle); cursor: pointer; border: none; background: none; padding: 0; }
.dp-close:hover { color: var(--c-text); }
.dp-empty { padding: 28px 14px; text-align: center; color: var(--c-subtle); font-size: 12px; line-height: 1.6; }
.dp-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; flex: 1; }
.dp-label-badge { display: inline-block; font-size: 10px; font-weight: 650;
  padding: 3px 8px; border-radius: 999px; margin-bottom: 4px; }
.dp-field { display: flex; flex-direction: column; gap: 3px; }
.dp-field label { font-size: 10px; color: var(--c-muted); font-weight: 650;
  text-transform: uppercase; letter-spacing: 0.04em; }
.dp-field input, .dp-field select, .dp-field textarea {
  font-size: 12px; padding: 6px 8px; border-radius: 6px;
  border: 1px solid var(--c-line); background: var(--c-panel); color: var(--c-text);
  width: 100%; font-family: inherit; resize: none; }
.dp-field textarea { min-height: 52px; }
.dp-field input:focus, .dp-field select:focus, .dp-field textarea:focus
  { outline: none; border-color: #2f7dd1; box-shadow: 0 0 0 3px rgba(47,125,209,0.12); }
.dp-actions { padding: 10px 14px; border-top: 1px solid var(--c-border-soft);
  display: flex; gap: 6px; flex-shrink: 0; }
.dp-actions button { flex: 1; font-size: 11px; padding: 6px; border-radius: 6px;
  border: 1px solid var(--c-line); background: var(--c-panel); cursor: pointer; color: var(--c-btn-text); }
.dp-actions button.save { background: #2f7dd1; color: var(--c-panel); border-color: #2f7dd1; }
.dp-actions button.del { color: var(--c-danger); border-color: var(--c-danger-line); }
.dp-actions button:hover { filter: brightness(0.92); }
.create-modal { position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(15,23,42,0.35); display: flex; align-items: center;
  justify-content: center; z-index: 100; }
.modal-box { background: var(--c-panel); border-radius: 8px; border: 1px solid var(--c-line);
  padding: 16px; width: 240px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 18px 45px var(--c-shadow); }
.modal-title { font-size: 13px; font-weight: 650; color: var(--c-text); }
.modal-field { display: flex; flex-direction: column; gap: 4px; }
.modal-field label { font-size: 10px; color: var(--c-muted); font-weight: 650;
  text-transform: uppercase; letter-spacing: 0.04em; }
.modal-field select, .modal-field input, .modal-field textarea {
  font-size: 12px; padding: 6px 8px; border-radius: 6px;
  border: 1px solid var(--c-line); background: var(--c-panel); color: var(--c-text);
  width: 100%; font-family: inherit; }
.modal-field textarea { min-height: 48px; resize: none; }
.modal-actions { display: flex; gap: 6px; }
.modal-actions button { flex: 1; font-size: 11px; padding: 6px; border-radius: 6px;
  border: 1px solid var(--c-line); background: var(--c-panel); cursor: pointer; color: var(--c-btn-text); }
.modal-actions button.confirm { background: #2f7dd1; color: var(--c-panel); border-color: #2f7dd1; }
</style>
</head>
<body>
<div class="shell">
  <div class="graph-header">
    <span class="graph-title" id="gTitle"><span class="live-dot"></span>Knowledge graph</span>
    <div class="graph-actions" id="gActions"></div>
  </div>
  <div class="workspace">
    <div class="graph-panel" id="gp">
      <canvas id="gc"></canvas>
      <div class="legend">
__LEGEND__
        <div class="leg"><div style="width:16px;height:1.5px;background:#24A47A;"></div>Found</div>
        <div class="leg" id="legPlaceholderEdge"><div style="width:16px;height:1.5px;background:repeating-linear-gradient(90deg,#bbb 0,#bbb 3px,transparent 3px,transparent 6px);"></div>Placeholder</div>
      </div>
    </div>
    <div class="detail-panel" id="dp">
      <div class="dp-header">
        <span class="dp-title">Inspector</span>
        <button class="dp-close" onclick="clearSelection()">&#x2715;</button>
      </div>
      <div class="dp-empty" id="dpEmpty">Click any node or edge<br>to inspect</div>
      <div id="dpContent" style="display:none;flex:1;flex-direction:column;">
        <div class="dp-body" id="dpBody"></div>
        <div class="dp-actions" id="dpActions" style="display:none;"></div>
      </div>
    </div>
  </div>
</div>
<div class="create-modal" id="createModal" style="display:none;">
  <div class="modal-box" id="modalBox"></div>
</div>
<script>
(function() {
const EDIT_MODE = __EDIT_MODE__;
// Both palettes ship with the document; which one is live follows the theme.
// The iframe cannot read Gradio's .dark class on the parent, so the theme comes
// from prefers-color-scheme by default and from an explicit data-theme that the
// parent postMessages (Gradio's toggle can disagree with the OS).
const COLOR_L = __COLOR__, BADGE_BG_L = __BADGE_BG__, BADGE_COLOR_L = __BADGE_CLR__;
const COLOR_D = __COLOR_D__, BADGE_BG_D = __BADGE_BG_D__, BADGE_COLOR_D = __BADGE_CLR_D__;
// Values the canvas paints itself, which have no CSS rule to inherit from.
const CHROME_L = {missingFill:'#F5F5F5', missingText:'#9aa0a6', edgeIdle:'#94a3b8',
                  fallbackStroke:'#aaa', fallbackFill:'#eee', fallbackText:'#1F2937'};
const CHROME_D = {missingFill:'#1F232A', missingText:'#6B7280', edgeIdle:'#5b6675',
                  fallbackStroke:'#6b7787', fallbackFill:'#242830', fallbackText:'#e5e9ef'};
let DARK = false, themePinned = false;
let COLOR = COLOR_L, BADGE_BG = BADGE_BG_L, BADGE_COLOR = BADGE_COLOR_L, CHROME = CHROME_L;

// The legend is server-rendered with the light palette inlined; re-colour its
// swatches from whichever palette is live so it cannot contradict the canvas.
function redrawLegend() {
  const sw = document.querySelectorAll('.ld[data-label]');
  for (let i = 0; i < sw.length; i++) {
    const lb = sw[i].getAttribute('data-label');
    if (BADGE_BG[lb]) sw[i].style.background = BADGE_BG[lb];
    if (COLOR[lb]) sw[i].style.border = '1px solid ' + COLOR[lb];
  }
}

function applyTheme(dark) {
  DARK = !!dark;
  COLOR = DARK ? COLOR_D : COLOR_L;
  BADGE_BG = DARK ? BADGE_BG_D : BADGE_BG_L;
  BADGE_COLOR = DARK ? BADGE_COLOR_D : BADGE_COLOR_L;
  CHROME = DARK ? CHROME_D : CHROME_L;
  document.documentElement.setAttribute('data-theme', DARK ? 'dark' : 'light');
}
const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
applyTheme(mq ? mq.matches : false);
if (mq && mq.addEventListener) {
  // Only while the parent has not pinned a theme: an explicit choice outranks the OS.
  mq.addEventListener('change', function(e) {
    if (!themePinned) { applyTheme(e.matches); redrawLegend(); draw(); }
  });
}
window.addEventListener('message', function(e) {
  const d = e.data;
  if (!d || d.kind !== 'cbt_theme') return;
  themePinned = true;
  applyTheme(d.dark);
  redrawLegend(); draw();
});
const NODE_CLASSES = __NODE_CLASSES__;
const PREDICATES = __PREDICATES__;

let nodes = __NODES__;
let edges = __EDGES__;
// Nodes kept out of the drawing but still written by saveJSON — Utterance
// provenance, which is too noisy to render yet is what lets a re-loaded export
// quote the dialogue in the Query tab.
const hiddenNodes = __HIDDEN_NODES__;

// ── Layout constants ──────────────────────────────────────────────────────
const LAYERS = {
  Client: 0, Session: 1,
  Problem: 2, Goal: 2,
  CoreBelief: 3, IntermediateBelief: 3,
  Situation: 4, AutomaticThought: 4,
  Reaction: 5, AdaptiveResponse: 5,
};
const RIGHT_SIDE = new Set(["Intervention", "Homework"]);
const RECT_LABELS = new Set(["Problem", "Goal"]);
const RADIUS_CIRCLE = 28;
const RADIUS_RECT_H = 22;
const RADIUS_RECT_W = 38;
const ARROW_CLEARANCE = 8;
const CURVE = 28;
const MAX_PER_ROW = 6;
// Sub-rows must clear a full circle (2*RADIUS_CIRCLE = 56) plus PAD_Y, or the
// wrapped rows of a dense layer overlap before separation even starts.
const SUBROW_GAP = 78;
// Separation is measured between node *boxes*, not centres — a circle is
// 56x56 and a Problem/Goal rect is 76x44, so a single centre-distance
// threshold cannot describe both.
const PAD_X = 16;   // min horizontal gap between two node boxes
const PAD_Y = 12;   // min vertical gap
const SEP_PASSES = 14;

function halfW(n) { return RECT_LABELS.has(n.label) ? RADIUS_RECT_W : RADIUS_CIRCLE; }
function halfH(n) { return RECT_LABELS.has(n.label) ? RADIUS_RECT_H : RADIUS_CIRCLE; }

function applyLayout(W, H) {
  const MARGIN = 48;
  const RIGHT_W = 160;

  // Virtual canvas scales with node count so dense graphs breathe
  const SCALE = Math.max(1, Math.sqrt(nodes.length / 18));
  const VW = W * SCALE, VH = H * SCALE;
  const vM = MARGIN * SCALE;

  const layerGroups = {}, rightGroups = {};
  for (const n of nodes) {
    if (RIGHT_SIDE.has(n.label)) {
      (rightGroups[n.label] = rightGroups[n.label] || []).push(n);
    } else {
      const l = LAYERS[n.label] !== undefined ? LAYERS[n.label] : 6;
      (layerGroups[l] = layerGroups[l] || []).push(n);
    }
  }

  // The right-hand region is sized from what it must hold, not a fixed 160px:
  // a long session yields more Intervention/Homework nodes than fit in one
  // column, and the old single column simply stacked the overflow.
  const rightAll = [];
  Object.keys(rightGroups).forEach(function(label) {
    rightGroups[label].forEach(function(n) { rightAll.push(n); });
  });
  const cellH = RADIUS_CIRCLE * 2 + PAD_Y;
  const cellW = RADIUS_CIRCLE * 2 + PAD_X;
  const rightRowsFit = Math.max(1, Math.floor((VH - vM * 2) / cellH));
  const rightCols = rightAll.length ? Math.ceil(rightAll.length / rightRowsFit) : 0;
  // Never let the column grid eat more than a third of the canvas; past that the
  // main hierarchy is what matters and separation can spill the rest.
  // The VW/3 cap limits how much *padding* the region takes, but must never
  // squeeze the grid itself below the width its columns need.
  const rightW = Math.max(rightCols * cellW, Math.min(RIGHT_W * SCALE, VW / 3));
  const vMAIN_W = VW - rightW - vM * 2;

  const mainLayers = Object.keys(layerGroups).map(Number).sort(function(a,b){return a-b;});
  const totalLayers = mainLayers.length;
  const nodeBaseY = {}, nodeBaseX = {};

  // How many nodes actually fit across, sized off the widest node (a rect) so a
  // narrow panel wraps sooner instead of packing a row tighter than separation
  // can ever undo. Caps at MAX_PER_ROW to keep wide panels from going flat.
  const perRow = Math.max(2, Math.min(
    MAX_PER_ROW, Math.floor(vMAIN_W / (RADIUS_RECT_W * 2 + PAD_X))));

  // Every sub-row anywhere in the main area is spaced ROW_H apart, counted across
  // layers rather than per layer. That is what stops a dense layer's wrapped rows
  // from spilling into its neighbours (the old even split gave each layer the same
  // band no matter its size), while still spreading rows over the available height
  // the way that split did.
  //
  // ROW_H fills the panel when there is room and falls back to MIN_ROW_H when there
  // is not, growing the canvas only then. Pinning it at the minimum instead made a
  // short graph — 7 layers, one node each — a tall thin ribbon that zoomToFit had
  // to shrink to fit. Rows are fixed (Y_BAND = 0), so MIN_ROW_H needs no drift
  // allowance and every leftover collision is horizontal, where perRow guarantees
  // the row fits.
  const Y_BAND = 0;
  const MIN_ROW_H = RADIUS_CIRCLE * 2 + PAD_Y;
  let totalSubRows = 0;
  mainLayers.forEach(function(l) {
    totalSubRows += Math.ceil(layerGroups[l].length / perRow);
  });
  const ROW_H = totalSubRows > 1
    ? Math.max(MIN_ROW_H, (VH - vM * 2) / (totalSubRows - 1))
    : MIN_ROW_H;
  const needH = Math.max(0, totalSubRows - 1) * ROW_H + vM * 2;
  const VH2 = Math.max(VH, needH);

  // Hierarchical slot assignment — dense layers wrap into sub-rows
  let yCursor = vM + Math.max(0, (VH2 - needH) / 2);
  mainLayers.forEach(function(l) {
    const row = layerGroups[l];
    const subRows = Math.ceil(row.length / perRow);
    for (let i = 0; i < row.length; i++) {
      const sr = Math.floor(i / perRow);
      const idxInSr = i % perRow;
      const cntInSr = Math.min(perRow, row.length - sr * perRow);
      const slotW = vMAIN_W / cntInSr;
      row[i].x = vM + slotW * idxInSr + slotW / 2;
      row[i].y = yCursor + sr * ROW_H;
      nodeBaseY[row[i].id] = row[i].y;
    }
    // No extra gap between layers: rows already carry full clearance, and the
    // gap was pure inflation — it stretched the graph without separating anything.
    yCursor += subRows * ROW_H;
  });

  // Right-side nodes (Intervention, Homework) — column-major grid, filled top to
  // bottom then wrapping to a new column. rightAll keeps same-label nodes
  // adjacent, so a label still reads as one block.
  if (rightAll.length) {
    const rows = Math.ceil(rightAll.length / Math.max(1, rightCols));
    const spanH = VH2 - vM * 2;
    const stepY = rows > 1 ? Math.max(cellH, spanH / (rows - 1)) : 0;
    const yStart = vM + Math.max(0, (spanH - stepY * (rows - 1)) / 2);
    const xStart = VW - rightW + cellW / 2;
    rightAll.forEach(function(n, i) {
      n.x = xStart + Math.floor(i / rows) * cellW;
      n.y = yStart + (i % rows) * stepY;
      // Held exactly here. Columns sit one cell apart and rows one cellH apart —
      // precisely the minimum separation — so there is no slack to drift into:
      // any force nudge closes the gap, and the separation pass cannot undo it
      // (a whole column pinned against the x-clamp collapses onto one x).
      nodeBaseX[n.id] = n.x;
      nodeBaseY[n.id] = n.y;
    });
  }

  // Spring-force refinement (80 iterations, annealed) — stronger repulsion
  for (let iter = 0; iter < 80; iter++) {
    const step = 0.4 * Math.pow(0.97, iter);
    const force = {};
    for (const n of nodes) force[n.id] = {x: 0, y: 0};

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const rep = Math.min(12000 / (dist * dist), 90);
        const ux = dx / dist, uy = dy / dist;
        force[a.id].x -= ux * rep; force[a.id].y -= uy * rep * 0.15;
        force[b.id].x += ux * rep; force[b.id].y += uy * rep * 0.15;
      }
    }

    const nodeMap = {};
    for (const n of nodes) nodeMap[n.id] = n;
    for (const e of edges) {
      const a = nodeMap[e.from], b = nodeMap[e.to];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.max(Math.hypot(dx, dy), 1);
      const att = (dist - 130) * 0.03;
      const ux = dx / dist, uy = dy / dist;
      force[a.id].x += ux * att; force[a.id].y += uy * att;
      force[b.id].x -= ux * att; force[b.id].y -= uy * att;
    }

    for (const n of nodes) {
      if (RIGHT_SIDE.has(n.label)) {
        if (nodeBaseX[n.id] !== undefined) { n.x = nodeBaseX[n.id]; n.y = nodeBaseY[n.id]; }
      } else {
        const yBase = nodeBaseY[n.id] !== undefined ? nodeBaseY[n.id] : VH2 / 2;
        n.x = Math.max(vM + 20, Math.min(vM + vMAIN_W - 20, n.x + force[n.id].x * step));
        n.y = Math.max(yBase - Y_BAND, Math.min(yBase + Y_BAND, n.y + force[n.id].y * step));
      }
    }
  }

  // Hard-separation relaxation — iterated, and box-aware rather than
  // centre-distance, so wide rects separate as far as they actually are wide.
  // Resolving a<->b can push a into c, so this repeats until a pass is clean
  // (SEP_PASSES caps the work on pathological graphs).
  for (let pass = 0; pass < SEP_PASSES; pass++) {
    let moved = false;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const minX = halfW(a) + halfW(b) + PAD_X;
        const minY = halfH(a) + halfH(b) + PAD_Y;
        const dx = b.x - a.x, dy = b.y - a.y;
        const ovX = minX - Math.abs(dx);
        const ovY = minY - Math.abs(dy);
        if (ovX <= 0 || ovY <= 0) continue;   // boxes already clear
        moved = true;
        // Resolve along whichever axis the node is actually free to move on,
        // which the two regions disagree about: main-area nodes are pinned to a
        // ~50px layer band but span the full width, so they can only go
        // sideways. The right-side grid is free on both axes, so there we take
        // the axis of least penetration — the shortest way out of the overlap.
        if (RIGHT_SIDE.has(a.label) && RIGHT_SIDE.has(b.label)) {
          if (ovY <= ovX) {
            const dir = dy === 0 ? (i % 2 ? 1 : -1) : Math.sign(dy);
            const s = dir * ovY / 2;
            a.y -= s; b.y += s;
          } else {
            const dir = dx === 0 ? (i % 2 ? 1 : -1) : Math.sign(dx);
            const s = dir * ovX / 2;
            a.x -= s; b.x += s;
          }
        } else {
          const dir = dx === 0 ? (i % 2 ? 1 : -1) : Math.sign(dx);
          const s = dir * ovX / 2;
          a.x -= s; b.x += s;
        }
      }
    }
    // Re-apply the bounds the force phase honoured; the old single pass skipped
    // this and let nodes escape both the canvas and their own layer.
    for (const n of nodes) {
      if (RIGHT_SIDE.has(n.label)) {
        if (nodeBaseX[n.id] !== undefined) { n.x = nodeBaseX[n.id]; n.y = nodeBaseY[n.id]; }
      } else {
        const yBase = nodeBaseY[n.id] !== undefined ? nodeBaseY[n.id] : VH2 / 2;
        n.x = Math.max(vM + 20, Math.min(vM + vMAIN_W - 20, n.x));
        n.y = Math.max(yBase - Y_BAND, Math.min(yBase + Y_BAND, n.y));
      }
    }
    if (!moved) break;
  }
}

// ── Canvas setup ──────────────────────────────────────────────────────────
const gp = document.getElementById('gp');
const cv = document.getElementById('gc');
const ctx = cv.getContext('2d');
let dpr = window.devicePixelRatio || 1;
let lastNodeCount = 0;
// Panel size at the last layout. Tracked because a canvas that first laid out
// at one size must re-flow when that size changes (see resize()).
let lastW = 0, lastH = 0;
let W = 1, H = 1;

// ── Viewport state ────────────────────────────────────────────────────────
let view = { scale: 1, tx: 0, ty: 0 };
let panning = null;
// Once the user zooms or pans, the viewport is theirs — auto-fit stops fighting
// them. Until then the canvas keeps the graph fitted, so a structural edit that
// changes the layout's extent cannot leave the drawing drifted or half off-view.
let userAdjusted = false;

function toWorld(sx, sy) {
  return { x: (sx - view.tx) / view.scale, y: (sy - view.ty) / view.scale };
}

function zoomToFit(pad) {
  pad = pad !== undefined ? pad : 40;
  if (!nodes.length) return;
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const n of nodes) {
    // Bound the node's drawn extent, not its centre — fitting centre-to-centre
    // let the outermost nodes hang over the edge by their own radius.
    const hwN = halfW(n), hhN = halfH(n);
    minX=Math.min(minX,n.x-hwN); minY=Math.min(minY,n.y-hhN);
    maxX=Math.max(maxX,n.x+hwN); maxY=Math.max(maxY,n.y+hhN);
  }
  const gw=(maxX-minX)||1, gh=(maxY-minY)||1;
  // Fit shrinks to fit but never magnifies. The old 3x ceiling blew a freshly
  // loaded graph — a handful of found nodes, tiny spread — up to 168px nodes
  // with labels to match. Manual ＋ still goes to 3x for a deliberate close-up.
  view.scale = Math.max(0.2, Math.min(1, Math.min((W-pad*2)/gw, (H-pad*2)/gh)));
  view.tx = (W - gw*view.scale)/2 - minX*view.scale;
  view.ty = (H - gh*view.scale)/2 - minY*view.scale;
  draw();
}

function resize() {
  const rect = gp.getBoundingClientRect();
  const legendEl = gp.querySelector('.legend');
  const legendH = legendEl ? legendEl.offsetHeight : 0;
  const w = Math.max(rect.width, 10);
  const h = Math.max(rect.height - legendH, 10);
  cv.style.width = w + 'px'; cv.style.height = h + 'px'; cv.style.top = '0px';
  cv.width = w * dpr; cv.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  W = w; H = h;
  // Re-lay-out when the node set changes OR the panel changes size. Size used to
  // be ignored, so a canvas that first laid out at one width kept coordinates
  // computed for that width forever — and the two ways that happens are exactly
  // the reported ones: the iframe renders while its tab is hidden (zero width),
  // and Apply re-renders the panel while the column is still reflowing. The
  // result was a graph drawn off-centre with nodes past the edge.
  const structural = nodes.length !== lastNodeCount;
  const resized = Math.abs(w - lastW) > 2 || Math.abs(h - lastH) > 2;
  if (nodes.length > 0 && w > 50 && h > 50 && (structural || resized)) {
    applyLayout(w, h);
    lastNodeCount = nodes.length; lastW = w; lastH = h;
    if (!userAdjusted) zoomToFit(); else draw();
  } else {
    draw();
  }
}

// ── Toolbar (zoom controls always; edit controls only in EDIT_MODE) ────────
const gActions = document.getElementById('gActions');
gActions.innerHTML =
  '<button class="btn-sm" id="btnFit">Fit</button>' +
  '<button class="btn-sm" id="btnZoomIn">＋</button>' +
  '<button class="btn-sm" id="btnZoomOut">－</button>' +
  '<button class="btn-sm" id="btnReset">Reset</button>' +
  (EDIT_MODE ?
    '<button class="btn-sm" id="btnNode">+ Node</button>' +
    '<button class="btn-sm" id="btnEdge">+ Edge</button>' +
    '<button class="btn-sm primary" id="btnSave">Save JSON</button>' : '');

document.getElementById('btnFit').addEventListener('click', function() { userAdjusted = false; zoomToFit(); });
document.getElementById('btnZoomIn').addEventListener('click', function() {
  const before = toWorld(W/2, H/2);
  userAdjusted = true;
  view.scale = Math.min(3, view.scale * 1.2);
  view.tx = W/2 - before.x * view.scale; view.ty = H/2 - before.y * view.scale; draw();
});
document.getElementById('btnZoomOut').addEventListener('click', function() {
  const before = toWorld(W/2, H/2);
  userAdjusted = true;
  view.scale = Math.max(0.2, view.scale / 1.2);
  view.tx = W/2 - before.x * view.scale; view.ty = H/2 - before.y * view.scale; draw();
});
document.getElementById('btnReset').addEventListener('click', function() {
  userAdjusted = false;
  if (nodes.length > 0) { applyLayout(W, H); zoomToFit(); } else { draw(); }
});
if (EDIT_MODE) {
  document.getElementById('btnNode').addEventListener('click', showCreateNode);
  document.getElementById('btnEdge').addEventListener('click', startEdgeMode);
  document.getElementById('btnSave').addEventListener('click', saveJSON);
}

// Hide placeholder-edge legend entry in read-only mode
const legPE = document.getElementById('legPlaceholderEdge');
if (legPE && !EDIT_MODE) legPE.style.display = 'none';

// ── State ─────────────────────────────────────────────────────────────────
let selected = null;
let edgeMode = false;
let edgeFrom = null;
let drag = null, dragOff = {x: 0, y: 0};

// ── Drawing ───────────────────────────────────────────────────────────────
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x+r, y); c.lineTo(x+w-r, y); c.arcTo(x+w, y, x+w, y+r, r);
  c.lineTo(x+w, y+h-r); c.arcTo(x+w, y+h, x+w-r, y+h, r);
  c.lineTo(x+r, y+h); c.arcTo(x, y+h, x, y+h-r, r);
  c.lineTo(x, y+r); c.arcTo(x, y, x+r, y, r);
  c.closePath();
}

function nodeAt(wx, wy) {
  for (const n of nodes) {
    if (RECT_LABELS.has(n.label)) {
      if (wx>=n.x-RADIUS_RECT_W && wx<=n.x+RADIUS_RECT_W && wy>=n.y-RADIUS_RECT_H && wy<=n.y+RADIUS_RECT_H) return n;
    }
  }
  for (const n of nodes) {
    if (!RECT_LABELS.has(n.label) && Math.hypot(n.x-wx, n.y-wy) < RADIUS_CIRCLE) return n;
  }
  return null;
}

function edgeAt(wx, wy) {
  const nmap = {};
  for (const n of nodes) nmap[n.id] = n;
  for (const e of edges) {
    const a = nmap[e.from], b = nmap[e.to];
    if (!a || !b) continue;
    if (Math.hypot((a.x+b.x)/2 - wx, (a.y+b.y)/2 - wy) < 16) return e;
  }
  return null;
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.translate(view.tx, view.ty);
  ctx.scale(view.scale, view.scale);

  const nmap = {};
  for (const n of nodes) nmap[n.id] = n;

  // Draw edges
  for (const e of edges) {
    const a = nmap[e.from], b = nmap[e.to];
    if (!a || !b) continue;
    const sel = selected && selected.type === 'edge' && selected.id === e.id;
    const isFound = e.status === 'found';
    const col = sel ? '#2f7dd1' : (isFound ? '#24A47A' : CHROME.edgeIdle);
    ctx.save();
    ctx.strokeStyle = col; ctx.lineWidth = sel ? 2.2 : 1.4;
    if (!isFound) ctx.setLineDash([5, 3]);
    const dx=b.x-a.x, dy=b.y-a.y, dist=Math.max(Math.hypot(dx,dy),1);
    const ux=dx/dist, uy=dy/dist;
    const startR = RECT_LABELS.has(a.label) ? RADIUS_RECT_H : RADIUS_CIRCLE;
    const endR   = RECT_LABELS.has(b.label) ? RADIUS_RECT_H : RADIUS_CIRCLE;
    const sx=a.x+ux*startR, sy=a.y+uy*startR;
    const ex=b.x-ux*(endR+ARROW_CLEARANCE), ey=b.y-uy*(endR+ARROW_CLEARANCE);
    const cx1=sx+uy*CURVE, cy1=sy-ux*CURVE, cx2=ex+uy*CURVE, cy2=ey-ux*CURVE;
    ctx.beginPath(); ctx.moveTo(sx,sy); ctx.bezierCurveTo(cx1,cy1,cx2,cy2,ex,ey); ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(ey-cy2, ex-cx2);
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.moveTo(ex,ey);
    ctx.lineTo(ex-9*Math.cos(ang-0.4), ey-9*Math.sin(ang-0.4));
    ctx.lineTo(ex-9*Math.cos(ang+0.4), ey-9*Math.sin(ang+0.4));
    ctx.closePath(); ctx.fill();
    const mx=(sx+ex)/2+uy*CURVE*0.5, my=(sy+ey)/2-ux*CURVE*0.5;
    ctx.font='9px system-ui,sans-serif';
    ctx.fillStyle = sel ? '#2f7dd1' : (isFound ? '#12613F' : '#94a3b8');
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(e.predicate, mx, my-7);
    ctx.restore();
  }

  // Draw nodes
  for (const n of nodes) {
    const sel   = selected && selected.type === 'node' && selected.id === n.id;
    const efrom = edgeMode && edgeFrom === n.id;
    ctx.save();
    const isMissing = n.status === 'missing';
    const col    = isMissing ? COLOR['missing'] : (COLOR[n.label] || CHROME.fallbackStroke);
    const bgCol  = isMissing ? CHROME.missingFill : (BADGE_BG[n.label] || CHROME.fallbackFill);
    const isRect = RECT_LABELS.has(n.label);
    // §UI-1.1 §2.1: use BADGE_COLOR for both text lines, not stroke colour
    const nodeTextCol = isMissing ? CHROME.missingText : (BADGE_COLOR[n.label] || CHROME.fallbackText);

    if (sel || efrom) { ctx.shadowColor = efrom ? '#2f7dd1' : '#24A47A'; ctx.shadowBlur = 10; }
    ctx.fillStyle = bgCol; ctx.strokeStyle = sel ? '#2f7dd1' : col; ctx.lineWidth = sel ? 2.2 : 1.5;
    if (isMissing) ctx.setLineDash([4, 3]);
    if (isRect) { roundRect(ctx, n.x-RADIUS_RECT_W, n.y-RADIUS_RECT_H, RADIUS_RECT_W*2, RADIUS_RECT_H*2, 6); }
    else { ctx.beginPath(); ctx.arc(n.x, n.y, RADIUS_CIRCLE, 0, Math.PI*2); }
    ctx.fill(); ctx.stroke(); ctx.setLineDash([]); ctx.shadowBlur = 0;

    // Class label — bold
    ctx.fillStyle = nodeTextCol; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = '600 9px system-ui,sans-serif';
    ctx.fillText(n.label.length > 13 ? n.label.slice(0,11)+'…' : n.label, n.x, n.y-7);
    // Property preview — same colour at 82% alpha
    const rawProp = Object.values(n.props||{})[0];
    const mainProp = rawProp !== undefined && rawProp !== null ? String(rawProp) : (isMissing ? 'missing' : '');
    ctx.save();
    ctx.globalAlpha = isMissing ? 1 : 0.82;
    ctx.fillStyle = nodeTextCol; ctx.font = '8px system-ui,sans-serif';
    ctx.fillText(mainProp.slice(0,13)+(mainProp.length>13?'…':''), n.x, n.y+6);
    ctx.restore();

    ctx.restore();
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // reset transform after drawing
}

// ── Mouse interaction ─────────────────────────────────────────────────────
cv.addEventListener('mousedown', function(e) {
  const r = cv.getBoundingClientRect();
  const w = toWorld(e.clientX - r.left, e.clientY - r.top);
  const n = nodeAt(w.x, w.y);
  if (edgeMode) {
    if (n) {
      if (!edgeFrom) { edgeFrom = n.id; draw(); }
      else if (edgeFrom !== n.id) { showCreateEdge(edgeFrom, n.id); edgeFrom = null; edgeMode = false; }
    }
    return;
  }
  if (n) { drag = n; dragOff = {x: w.x-n.x, y: w.y-n.y}; selectItem('node', n.id); return; }
  const ed = edgeAt(w.x, w.y);
  if (ed) { selectItem('edge', ed.id); return; }
  userAdjusted = true;
  panning = { sx: e.clientX, sy: e.clientY, tx0: view.tx, ty0: view.ty };
  clearSelection();
});
cv.addEventListener('mousemove', function(e) {
  const r = cv.getBoundingClientRect();
  if (drag) {
    const w = toWorld(e.clientX - r.left, e.clientY - r.top);
    drag.x = w.x - dragOff.x; drag.y = w.y - dragOff.y; draw(); return;
  }
  if (panning) {
    view.tx = panning.tx0 + (e.clientX - panning.sx);
    view.ty = panning.ty0 + (e.clientY - panning.sy);
    draw();
  }
});
cv.addEventListener('mouseup', function() { drag = null; panning = null; });
cv.addEventListener('wheel', function(e) {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const before = toWorld(mx, my);
  userAdjusted = true;
  view.scale = Math.max(0.2, Math.min(3, view.scale * (e.deltaY < 0 ? 1.1 : 1/1.1)));
  view.tx = mx - before.x * view.scale; view.ty = my - before.y * view.scale;
  draw();
}, { passive: false });

function selectItem(type, id) {
  selected = {type, id}; draw();
  if (type === 'node') showNodePanel(nodes.find(n => n.id === id));
  else showEdgePanel(edges.find(e => e.id === id));
}
function clearSelection() { selected = null; showEmpty(); draw(); }
function showEmpty() {
  document.getElementById('dpEmpty').style.display = '';
  document.getElementById('dpContent').style.display = 'none';
}
function renderDP(body, actions) {
  document.getElementById('dpEmpty').style.display = 'none';
  const dc = document.getElementById('dpContent');
  dc.style.display = 'flex'; dc.style.flexDirection = 'column';
  document.getElementById('dpBody').innerHTML = body;
  const da = document.getElementById('dpActions');
  if (actions && EDIT_MODE) { da.style.display = 'flex'; da.innerHTML = actions; }
  else { da.style.display = 'none'; da.innerHTML = ''; }
}

function showNodePanel(n) {
  if (!n) return;
  const bb = BADGE_BG[n.label]||'#eef2f7', bc = BADGE_COLOR[n.label]||'#273142';
  const sBg = n.status==='found'?'#E4F7F0':'#F1F5F9', sFg = n.status==='found'?'#0F6046':'#64748b';
  const badge = '<span class="dp-label-badge" style="background:'+bb+';color:'+bc+';">'+n.label+'</span>';
  const stag  = '<span style="font-size:10px;padding:2px 7px;border-radius:10px;background:'+sBg+';color:'+sFg+';">'+n.status+'</span>';
  let fields = '<div class="dp-field">'+badge+' '+stag+'</div>';
  if (EDIT_MODE) {
    fields += '<div class="dp-field"><label>Node ID</label><input id="dpId" value="'+esc(n.id)+'" /></div>';
    fields += '<div class="dp-field"><label>Class</label><select id="dpClass">'+
      NODE_CLASSES.map(function(c){return '<option'+(c===n.label?' selected':'')+'>'+c+'</option>';}).join('')+'</select></div>';
    fields += '<div class="dp-field"><label>Status</label><select id="dpStatus">'+
      '<option'+(n.status==='found'?' selected':'')+'>found</option>'+
      '<option'+(n.status==='missing'?' selected':'')+'>missing</option></select></div>';
    const propStr = Object.keys(n.props||{}).length>0 ? JSON.stringify(n.props,null,1) : '';
    fields += '<div class="dp-field"><label>Properties (JSON)</label>'+
      '<textarea id="dpProps" placeholder="{&quot;content&quot;:&quot;...&quot;}">'+esc(propStr)+'</textarea></div>';
  } else {
    fields += '<div class="dp-field"><label>Node ID</label><input value="'+esc(n.id)+'" readonly /></div>';
    if (Object.keys(n.props||{}).length>0)
      fields += '<div class="dp-field"><label>Properties</label><textarea readonly>'+esc(JSON.stringify(n.props,null,1))+'</textarea></div>';
  }
  fields += '<div class="dp-field"><label>Evidence turns</label><input value="'+esc((n.evidence||[]).join(', ')||'—')+'" readonly /></div>';
  renderDP(fields,
    '<button class="save" onclick="saveNode(&quot;'+esc(n.id)+'&quot;)">Save</button>'+
    '<button class="del" onclick="deleteNode(&quot;'+esc(n.id)+'&quot;)">Delete</button>');
}

function showEdgePanel(e) {
  if (!e) return;
  const sBg = e.status==='found'?'#E4F7F0':'#F1F5F9', sFg = e.status==='found'?'#0F6046':'#64748b';
  const stag = '<span style="font-size:10px;padding:2px 7px;border-radius:10px;background:'+sBg+';color:'+sFg+';">'+e.status+'</span>';
  let fields = '<div class="dp-field"><span style="font-size:11px;font-weight:500;color:#222;">Edge</span> '+stag+'</div>';
  if (EDIT_MODE) {
    fields += '<div class="dp-field"><label>Predicate</label><select id="dpPred">'+
      PREDICATES.map(function(p){return '<option'+(p===e.predicate?' selected':'')+'>'+p+'</option>';}).join('')+'</select></div>';
    fields += '<div class="dp-field"><label>From</label><select id="dpFrom">'+
      nodes.map(function(n){return '<option value="'+n.id+'"'+(n.id===e.from?' selected':'')+'>'+n.label+' ('+n.id+')</option>';}).join('')+'</select></div>';
    fields += '<div class="dp-field"><label>To</label><select id="dpTo">'+
      nodes.map(function(n){return '<option value="'+n.id+'"'+(n.id===e.to?' selected':'')+'>'+n.label+' ('+n.id+')</option>';}).join('')+'</select></div>';
    fields += '<div class="dp-field"><label>Status</label><select id="dpEdgeStatus">'+
      '<option'+(e.status==='found'?' selected':'')+'>found</option>'+
      '<option'+(e.status==='placeholder'?' selected':'')+'>placeholder</option></select></div>';
  } else {
    fields += '<div class="dp-field"><label>Predicate</label><input value="'+esc(e.predicate)+'" readonly /></div>';
    fields += '<div class="dp-field"><label>From → To</label><input value="'+esc(e.from+' → '+e.to)+'" readonly /></div>';
  }
  fields += '<div class="dp-field"><label>Evidence turns</label><input value="'+esc((e.evidence||[]).join(', ')||'—')+'" readonly /></div>';
  renderDP(fields,
    '<button class="save" onclick="saveEdge(&quot;'+esc(e.id)+'&quot;)">Save</button>'+
    '<button class="del" onclick="deleteEdge(&quot;'+esc(e.id)+'&quot;)">Delete</button>');
}

// Push the edited graph up to the Gradio page. The canvas lives in an iframe
// srcdoc, so this postMessage is the only channel back to Python — without it
// edits would redraw here and be lost on the next re-render.
function syncGraph() {
  if (!EDIT_MODE) return;
  try {
    parent.postMessage({kind:'cbt_graph_sync',
                        nodes: nodes.concat(hiddenNodes), edges: edges}, '*');
  } catch (err) {}
}

// Re-run the layout after a change that moves a node between rows. A node's
// class picks its layer (LAYERS), and a newly created node has no slot at all,
// so without this the edit lands in the data but the drawing keeps the old
// position — which reads as the edit not having taken. `resize()` cannot cover
// it: that only re-lays-out when the node COUNT changes, and re-classing a node
// leaves the count alone.
function relayout() {
  if (nodes.length > 0 && W > 50 && H > 50) {
    applyLayout(W, H);
    lastNodeCount = nodes.length;
    // Re-fit: re-classing a node can add or drop a whole layer row, so the new
    // layout's extent differs from the one the view was fitted to. Without this
    // the graph drifts inside the viewport and reads as "the position broke".
    if (!userAdjusted) { zoomToFit(); return; }
  }
  draw();
}

function saveNode(id) {
  const n = nodes.find(function(x){return x.id===id;});
  if (!n) return;
  const prevLabel = n.label;
  n.label = document.getElementById('dpClass').value;
  n.status = document.getElementById('dpStatus').value;
  try { n.props = JSON.parse(document.getElementById('dpProps').value||'{}'); } catch(err) {}
  // Only on a class change — re-laying out after a props-only edit would make
  // the whole graph twitch for no reason.
  if (n.label !== prevLabel) relayout(); else draw();
  syncGraph(); showNodePanel(n);
}
function saveEdge(id) {
  const e = edges.find(function(x){return x.id===id;});
  if (!e) return;
  e.predicate = document.getElementById('dpPred').value;
  e.from = document.getElementById('dpFrom').value;
  e.to   = document.getElementById('dpTo').value;
  e.status = document.getElementById('dpEdgeStatus').value;
  draw(); syncGraph(); showEdgePanel(e);
}
function deleteNode(id) {
  nodes = nodes.filter(function(n){return n.id!==id;});
  edges = edges.filter(function(e){return e.from!==id && e.to!==id;});
  clearSelection(); relayout(); syncGraph();
}
function deleteEdge(id) {
  edges = edges.filter(function(e){return e.id!==id;});
  clearSelection(); draw(); syncGraph();
}

// ── Edge creation modal ───────────────────────────────────────────────────
let edgeModeTimeout = null;
function startEdgeMode() {
  edgeMode = true; edgeFrom = null;
  document.getElementById('gTitle').innerHTML = 'Click source node, then target node…';
  if (edgeModeTimeout) clearTimeout(edgeModeTimeout);
  edgeModeTimeout = setTimeout(function(){ edgeMode=false; edgeFrom=null; updateTitle(); }, 10000);
}
function showCreateEdge(fromId, toId) {
  if (edgeModeTimeout) clearTimeout(edgeModeTimeout);
  updateTitle();
  document.getElementById('modalBox').innerHTML =
    '<div class="modal-title">Add edge</div>'+
    '<div class="modal-field"><label>From</label><input value="'+esc(fromId)+'" readonly/></div>'+
    '<div class="modal-field"><label>To</label><input value="'+esc(toId)+'" readonly/></div>'+
    '<div class="modal-field"><label>Predicate</label><select id="mPred">'+
    PREDICATES.map(function(p){return '<option>'+p+'</option>';}).join('')+'</select></div>'+
    '<div class="modal-actions"><button onclick="closeModal()">Cancel</button>'+
    '<button class="confirm" onclick="confirmEdge(&quot;'+esc(fromId)+'&quot;,&quot;'+esc(toId)+'&quot;)">Add</button></div>';
  document.getElementById('createModal').style.display = 'flex';
}
function confirmEdge(from, to) {
  edges.push({id:'e'+Date.now(), from:from, to:to, predicate:document.getElementById('mPred').value, status:'found', evidence:[], props:{}});
  closeModal(); updateTitle(); draw(); syncGraph();
}

// ── Node creation modal ───────────────────────────────────────────────────
function showCreateNode() {
  document.getElementById('modalBox').innerHTML =
    '<div class="modal-title">Add node</div>'+
    '<div class="modal-field"><label>Class</label><select id="mClass">'+
    NODE_CLASSES.map(function(c){return '<option>'+c+'</option>';}).join('')+'</select></div>'+
    '<div class="modal-field"><label>Main text / content</label>'+
    '<textarea id="mText" placeholder="e.g. I will never succeed"></textarea></div>'+
    '<div class="modal-actions"><button onclick="closeModal()">Cancel</button>'+
    '<button class="confirm" onclick="confirmNode()">Add</button></div>';
  document.getElementById('createModal').style.display = 'flex';
}
function confirmNode() {
  const cls = document.getElementById('mClass').value;
  const txt = document.getElementById('mText').value;
  const propKeys = {Problem:'description',Goal:'statement',CoreBelief:'content',
    IntermediateBelief:'content',Situation:'description',AutomaticThought:'content',
    Reaction:'content',AdaptiveResponse:'content',Intervention:'description',
    Homework:'taskDescription',Client:'',Session:''};
  const propKey = propKeys[cls]||'content';
  // No random scatter: relayout() below gives it a proper slot in its layer.
  // The old random drop landed it anywhere, overlapping whatever was there.
  nodes.push({id:cls.toLowerCase().slice(0,3)+'_'+Date.now(), label:cls,
    x:0, y:0,
    status:'found', props:propKey?{[propKey]:txt}:{}, evidence:[]});
  closeModal(); updateTitle(); relayout(); syncGraph();
}

function saveJSON() {
  // V4_flat Stage 5 export shape (matches JsonGraphReader in graph_reader.py) so a saved
  // file can be re-loaded via "Load JSON" without corruption. Only 'found' items are
  // exported — 'missing' placeholders aren't real extracted facts.
  const outNodes = nodes.concat(hiddenNodes)
    .filter(function(n) { return n.status === 'found'; })
    .map(function(n) { return {id: n.id, label: n.label, properties: n.props || {},
                                evidence: n.evidence || []}; });
  const outEdges = edges
    .filter(function(e) { return e.status === 'found'; })
    .map(function(e) {
      const out = {type: e.predicate, from: e.from, to: e.to, evidence: e.evidence || []};
      if (e.props && e.props.reportedIntensity) out.reportedIntensity = e.props.reportedIntensity;
      return out;
    });
  const blob = new Blob([JSON.stringify({nodes:outNodes,edges:outEdges},null,2)],{type:'application/json'});
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='cbt_graph.json'; a.click();
}
function closeModal() { document.getElementById('createModal').style.display='none'; }
function updateTitle() {
  document.getElementById('gTitle').innerHTML =
    '<span class="live-dot"></span>Knowledge graph · '+nodes.length+' nodes · '+edges.length+' edges';
}
function esc(s) {
  if (s===undefined||s===null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Expose globals for inline onclick ─────────────────────────────────────
window.saveNode=saveNode; window.saveEdge=saveEdge;
window.deleteNode=deleteNode; window.deleteEdge=deleteEdge;
window.confirmEdge=confirmEdge; window.confirmNode=confirmNode;
window.closeModal=closeModal; window.clearSelection=clearSelection;

new ResizeObserver(resize).observe(gp);
resize();
updateTitle();
})();
</script>
</body>
</html>'''


# Full legend, in the original grouping — `None` marks a full-width row break.
# Part 2 renders all of it; Part 1 hides the classes it does not draw.
_LEGEND_ORDER = [
    "Client", "Session", "Problem", "Goal", "Intervention", "Homework", None,
    "CoreBelief", "IntermediateBelief", "Situation", "AutomaticThought",
    "Reaction", "AdaptiveResponse", None, "Utterance",
]
_RECT_LEGEND = {"Problem", "Goal"}


def _legend_html(hide_labels: set[str] | None = None,
                 row_breaks: bool = True) -> str:
    """Build the legend from NODE_COLORS so it can't drift from what is drawn.

    `row_breaks` keeps the original fixed grouping (Part 2, full list). Part 1
    filters classes out and turns them off, letting `.legend`'s flex-wrap
    re-flow instead — fixed breaks leave the rows lopsided once entries go.
    """
    hide_labels = hide_labels or set()
    out: list[str] = []
    for label in _LEGEND_ORDER:
        if label is None:
            if row_breaks:
                out.append('<div style="width:100%;height:0;"></div>')
            continue
        if label in hide_labels:
            continue
        fill, stroke, _text = NODE_COLORS[label]
        radius = "border-radius:2px;" if label in _RECT_LEGEND else ""
        out.append(
            # data-label lets redrawLegend() re-colour the swatch from the live
            # palette; the inline colours are just the light-theme default.
            f'<div class="leg"><div class="ld" data-label="{label}" '
            f'style="background:{fill};border:1px solid {stroke};{radius}"></div>'
            f'{label}</div>'
        )
    return "\n        ".join(out)


def _render_canvas(
    canvas_nodes: list[dict],
    canvas_edges: list[dict],
    edit_mode: bool = False,
    height: int = 610,
    hidden_nodes: list[dict] | None = None,
    hide_legend_labels: set[str] | None = None,
) -> str:
    """Render the canvas graph as an iframe srcdoc string.

    `hidden_nodes` are excluded from the drawing but still written by saveJSON.
    """
    nodes_json = json.dumps(canvas_nodes)
    edges_json = json.dumps(canvas_edges)
    hidden_json = json.dumps(hidden_nodes or [])
    edit_str = "true" if edit_mode else "false"
    color_json = json.dumps(_COLOR)
    badge_bg_json = json.dumps(_BADGE_BG)
    badge_clr_json = json.dumps(_BADGE_COLOR)
    color_d_json = json.dumps(_COLOR_D)
    badge_bg_d_json = json.dumps(_BADGE_BG_D)
    badge_clr_d_json = json.dumps(_BADGE_COLOR_D)
    classes_json = json.dumps(_NODE_CLASSES)
    predicates_json = json.dumps(_PREDICATES)

    filled = (
        _CANVAS_TEMPLATE
        .replace("__NODES__", nodes_json)
        .replace("__EDGES__", edges_json)
        .replace("__HIDDEN_NODES__", hidden_json)
        .replace("__SHELL_H__", str(height))
        .replace("__LEGEND__", _legend_html(hide_legend_labels,
                                            row_breaks=not hide_legend_labels))
        .replace("__EDIT_MODE__", edit_str)
        .replace("__COLOR__", color_json)
        .replace("__BADGE_BG__", badge_bg_json)
        .replace("__BADGE_CLR__", badge_clr_json)
        .replace("__COLOR_D__", color_d_json)
        .replace("__BADGE_BG_D__", badge_bg_d_json)
        .replace("__BADGE_CLR_D__", badge_clr_d_json)
        .replace("__NODE_CLASSES__", classes_json)
        .replace("__PREDICATES__", predicates_json)
    )
    escaped = html.escape(filled)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:{height}px; border:none; border-radius:8px;"></iframe>'
    )


# ─────────────────────────────────────────────────────────────────────────
# Session bar HTML helper
# ─────────────────────────────────────────────────────────────────────────

def _session_bar_html(phase: str, technique: str, turn_count: int, strategy: str = "none",
                       steer_status: str | None = None) -> str:
    steer = ""
    if strategy and strategy != "none":
        if steer_status == "steered":
            steer = (f'<span class="session-chip session-chip-warn">'
                     f'steer: {html.escape(strategy)}</span>')
        elif steer_status == "fallback":
            # The steering service errored/returned nothing for THIS turn and
            # SteeredRemoteGenerator silently fell back to the plain Ollama reply.
            steer = (f'<span class="session-chip session-chip-caution" '
                     f'title="Steering service unavailable this turn — used the plain reply instead.">'
                     f'steer: {html.escape(strategy)} (fallback)</span>')
        else:
            # steer_status is None — the active generator has no steering concept at all
            # (GENERATOR != steered), so the dropdown selection was never even attempted, not
            # just unavailable for one turn. Distinct from "fallback" so this doesn't read as a
            # transient hiccup — it means steering isn't wired up for this session at all.
            steer = (f'<span class="session-chip session-chip-muted" '
                     f'title="This session\'s generator has no steering support — set GENERATOR=steered '
                     f'and restart the chatbot for the dropdown to take effect.">'
                     f'steer: {html.escape(strategy)} (inactive)</span>')
    return (
        '<div class="session-strip">'
        f'<span class="session-chip session-chip-phase">{html.escape(phase)}</span>'
        f'<span class="session-chip session-chip-technique">{html.escape(technique)}</span>'
        f'{steer}'
        f'<span class="session-turn">Turn {turn_count}</span>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────
# Tab 1 — Therapy (Part 1)
# ─────────────────────────────────────────────────────────────────────────

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


def _bot_respond(message: str, history: list, session: Session, strategy: str = "none"):
    if session is None:
        session = _new_session()
    if hasattr(session.generator, "set_strategy"):   # manual steering button (GENERATOR=steered)
        session.generator.set_strategy(strategy)
    result = turn(session, message)
    history = history + [{"role": "assistant", "content": result["reply"]}]
    phase = result["phase"]
    technique = result["technique"]
    bar_html = _session_bar_html(phase, technique, session.turn_count, strategy,
                                  result.get("steer_status"))
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges(),
                                      skip_scaffold=True)
    graph_html = _render_canvas(nodes, edges, edit_mode=False,
                           hide_legend_labels=THERAPY_HIDDEN_LABELS,
                           height=THERAPY_CANVAS_H)
    return history, session, bar_html, graph_html


def _reset_therapy():
    session = _new_session()
    history = [{"role": "assistant", "content": INTRO}]
    bar_html = _session_bar_html("Rapport", "Rapport Building", 0)
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges(),
                                      skip_scaffold=True)
    graph_html = _render_canvas(nodes, edges, edit_mode=False,
                           hide_legend_labels=THERAPY_HIDDEN_LABELS,
                           height=THERAPY_CANVAS_H)
    return history, session, bar_html, graph_html


# ─────────────────────────────────────────────────────────────────────────
# Tab 2 — Query (Part 2)
# ─────────────────────────────────────────────────────────────────────────

# Loaded canonical graphs keyed by handle (process-local).
_loaded_graphs: dict = {}


def _summary_text(nodes: list[GraphNode], edges: list[GraphEdge], label: str) -> str:
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.label] = counts.get(n.label, 0) + 1
    rows = [f"- {k}: {v}" for k, v in sorted(counts.items())]
    return (
        f"Loaded **{label}** — {len(nodes)} nodes, {len(edges)} edges.\n\n"
        + ("\n".join(rows) if rows else "(empty)")
    )


def _make_query_graph_html(gnodes: list[GraphNode], gedges: list[GraphEdge]) -> str:
    cn, ce = _build_canvas_data(gnodes, gedges)
    # Utterances are filtered out of the drawing by _build_canvas_data, but must
    # survive a save→re-load round-trip or the re-loaded graph can no longer
    # quote the dialogue.
    hidden, _ = _build_canvas_data(
        [n for n in gnodes if n.label == "Utterance"], [], skip_utterance=False
    )
    return _render_canvas(cn, ce, edit_mode=True, hidden_nodes=hidden)


def _load_live(therapy_session: Session):
    if therapy_session is None:
        return (
            None,
            "No active therapy session yet — go to Tab 1 first.",
            _render_canvas([], [], edit_mode=True),
            [],
            "",
        )
    reader = factory.make_reader_live(therapy_session.graph, label="Live therapy session")
    gnodes, gedges = reader.load()
    handle = uuid.uuid4().hex[:12]
    _loaded_graphs[handle] = (gnodes, gedges, reader.label())
    return (
        handle,
        _summary_text(gnodes, gedges, reader.label()),
        _make_query_graph_html(gnodes, gedges),
        [],
        "",
    )


def _load_json(file_obj):
    if file_obj is None:
        return None, "Upload a V4_flat JSON export first.", _render_canvas([], [], edit_mode=True), [], ""
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    reader = factory.make_reader_json(path)
    gnodes, gedges = reader.load()
    handle = uuid.uuid4().hex[:12]
    _loaded_graphs[handle] = (gnodes, gedges, reader.label())
    return (
        handle,
        _summary_text(gnodes, gedges, reader.label()),
        _make_query_graph_html(gnodes, gedges),
        [],
        "",
    )


def _load_neo4j(uri: str, user: str, password: str):
    try:
        reader = factory.make_reader_neo4j(
            uri=uri or os.environ.get("NEO4J_URI"),
            user=user or os.environ.get("NEO4J_USER"),
            password=password or os.environ.get("NEO4J_PASSWORD"),
        )
        gnodes, gedges = reader.load()
    except Exception as exc:
        return None, f"Connect failed: {exc}", _render_canvas([], [], edit_mode=True), [], ""
    handle = uuid.uuid4().hex[:12]
    _loaded_graphs[handle] = (gnodes, gedges, reader.label())
    return (
        handle,
        _summary_text(gnodes, gedges, reader.label()),
        _make_query_graph_html(gnodes, gedges),
        [],
        "",
    )


def _query_ask(handle: str, question: str, chat_history: list):
    if not handle or handle not in _loaded_graphs:
        return chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Load a graph first."},
        ], ""
    gnodes, gedges, _ = _loaded_graphs[handle]
    engine = factory.make_query_engine()
    try:
        result = engine.answer(question, gnodes, gedges)
        answer = result.get("answer", "(no answer)")
    except Exception as exc:
        answer = f"Query failed: {exc}"
    chat_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""


# ─────────────────────────────────────────────────────────────────────────
# Compose the UI
# ─────────────────────────────────────────────────────────────────────────
# Tab 1 — rewriting a client turn (branching)
# ─────────────────────────────────────────────────────────────────────────

def _history_to_chat(session: Session) -> list:
    msgs = [{"role": "assistant", "content": INTRO}]
    for user_msg, assistant_msg in session.history:
        msgs.append({"role": "user", "content": user_msg})
        if assistant_msg:
            msgs.append({"role": "assistant", "content": assistant_msg})
    return msgs


def _branch_controls(session: Session | None):
    """Refresh the version switcher from session state."""
    if session is None:
        return gr.update(choices=[], value=None)
    brs = list_branches(session)
    choices = [(("● " if b["active"] else "○ ") + b["label"], b["id"])
               for b in brs]
    active = next((b["id"] for b in brs if b["active"]), None)
    return gr.update(choices=choices, value=active)


def _therapy_view(session: Session, strategy: str, phase: str, technique: str,
                  steer_status: str | None = None):
    bar = _session_bar_html(phase, technique, session.turn_count, strategy,
                            steer_status)
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges(),
                                      skip_scaffold=True)
    return bar, _render_canvas(nodes, edges, edit_mode=False,
                           hide_legend_labels=THERAPY_HIDDEN_LABELS,
                           height=THERAPY_CANVAS_H)


def _message_index_to_turn(chat_history: list, index) -> int:
    """Chat-message position -> client turn number.

    Counted rather than computed from the position, because the transcript is
    not a clean alternation: it opens with the therapist's INTRO and a turn
    whose reply failed contributes no assistant message.
    """
    if isinstance(index, (tuple, list)):
        index = index[0]
    upto = chat_history[:int(index) + 1]
    return sum(1 for m in upto
               if (m.get("role") if isinstance(m, dict) else None) == "user")


def _message_text(message) -> str:
    """Text of a chat message, whichever shape Gradio hands back."""
    if isinstance(message, dict):
        message = message.get("content", "")
    if isinstance(message, dict):
        message = message.get("text", "")
    if isinstance(message, (list, tuple)):
        message = "".join(_message_text(part) for part in message)
    return str(message or "")


def _unchanged_view(session: Session | None, strategy: str, status: str):
    """The current state, rendered without touching it — the branch_outputs shape."""
    branch_dd = _branch_controls(session)
    if session is None:
        return ([], session, "",
                _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS,
                               height=THERAPY_CANVAS_H),
                branch_dd, status)
    snap = session.graph.snapshot()
    bar, graph_html = _therapy_view(session, strategy,
                                    snap.get("session_phase", "Rapport"),
                                    snap.get("active_technique", "Rapport Building"))
    return (_history_to_chat(session), session, bar, graph_html, branch_dd, status)


def _on_edit_message(session: Session, chat_history: list, strategy: str,
                     edit_data: gr.EditData):
    """Inline pencil-edit of a client message = rewrite that turn.

    Dismissing the inline editor still delivers the original text, so an
    unchanged message must short-circuit: replaying the turn would spend a whole
    generate call behind a spinner to arrive back where it started. Returning the
    current state also closes the editor, since the chatbot is re-rendered from
    the authoritative history.

    Deliberately NOT a generator. Yielding an interim "rewriting…" frame would be
    nicer for a real edit, but it makes Gradio mark the event `generator: True`,
    and the chatbot then shows a streaming/pending state that `show_progress` does
    not control — putting an indicator on screen for a plain cancel.

    The chatbot is also deliberately NOT among this event's outputs. Gradio draws a
    pending bubble under the last message whenever the Chatbot is an output of a
    running event, no matter how fast it returns or what show_progress says — which
    is the spinner that appeared under the therapist's reply on cancel. The chat is
    refreshed indirectly instead: a real rewrite writes a new `edit_token`, whose
    .change re-renders it, while a cancel returns gr.skip() and nothing fires.
    """
    new_text = _message_text(edit_data.value)
    # Gradio reports what the message held before the edit; prefer it over
    # indexing the history, which can misalign (the transcript opens with the
    # therapist's INTRO and a failed turn contributes no assistant message).
    previous = _message_text(getattr(edit_data, "previous_value", None))
    if not previous:
        index = edit_data.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        try:
            previous = _message_text(chat_history[int(index)])
        except (IndexError, TypeError, ValueError):
            previous = ""
    if new_text.strip() == previous.strip():
        # gr.skip() leaves each component untouched, so the edit_token keeps its
        # value and its .change never fires — the chat is not re-rendered at all.
        return (session, gr.skip(), gr.skip(), gr.skip(),
                "No change — the turn was left as it was.", gr.skip())
    turn_index = _message_index_to_turn(chat_history, edit_data.index)
    _chat, session, bar, graph_html, branch_dd, status = _do_rewrite(
        session, turn_index, new_text, strategy)
    # A fresh token fires edit_token.change, which is what refreshes the chat.
    return (session, bar, graph_html, branch_dd, status, uuid.uuid4().hex)


def _refresh_chat(session: Session | None):
    """Re-render the transcript. Driven by edit_token so the chatbot is only ever
    an output of an event that genuinely has something new to show."""
    return _history_to_chat(session) if session else []


def _do_rewrite(session: Session, turn_index, new_message: str,
                strategy: str = "none"):
    """Replay a past client turn with different words, keeping both versions."""
    if session is None or turn_index in (None, "") or not (new_message or "").strip():
        return _unchanged_view(session, strategy,
                               "Edit a message of yours to try it a different way.")
    if hasattr(session.generator, "set_strategy"):
        session.generator.set_strategy(strategy)
    try:
        result = edit_turn(session, int(turn_index), new_message)
    except ValueError as exc:
        branch_dd = _branch_controls(session)
        snap = session.graph.snapshot()
        bar, graph_html = _therapy_view(
            session, strategy, snap.get("session_phase", "Rapport"),
            snap.get("active_technique", "Rapport Building"))
        return (_history_to_chat(session), session, bar, graph_html,
                branch_dd, f"Cannot rewrite: {exc}")
    bar, graph_html = _therapy_view(session, strategy, result["phase"],
                                    result["technique"], result.get("steer_status"))
    branch_dd = _branch_controls(session)
    return (_history_to_chat(session), session, bar, graph_html, branch_dd,
            f"Replayed turn {int(turn_index)} — the previous wording is kept, "
            f"switch between them above.")


def _do_switch_branch(session: Session, branch_id, strategy: str = "none"):
    if session is None or not branch_id:
        branch_dd = _branch_controls(session)
        return ([] if session is None else _history_to_chat(session), session,
                "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS,
                                       height=THERAPY_CANVAS_H), branch_dd, "No version selected.")
    try:
        switch_branch(session, branch_id)
    except ValueError as exc:
        branch_dd = _branch_controls(session)
        return (_history_to_chat(session), session, "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS,
                                       height=THERAPY_CANVAS_H),
                branch_dd, f"{exc}")
    snap = session.graph.snapshot()
    bar, graph_html = _therapy_view(
        session, strategy, snap.get("session_phase", "Rapport"),
        snap.get("active_technique", "Rapport Building"))
    branch_dd = _branch_controls(session)
    return (_history_to_chat(session), session, bar, graph_html, branch_dd,
            "Switched — the conversation continues from this version.")


# ─────────────────────────────────────────────────────────────────────────
# Tab 2 — repairing a loaded graph, then handing it back to the session
# ─────────────────────────────────────────────────────────────────────────


def _node_text(n: GraphNode) -> str:
    for k in ("description", "content", "statement", "taskDescription", "text"):
        v = n.props.get(k)
        if isinstance(v, str) and v:
            return v
    return ""








def _sync_canvas_edits(handle, payload: str):
    """Fold edits made in the canvas back into the loaded graph.

    The canvas is the editor now; this is what makes its buttons persist instead
    of merely redrawing. Utterances ride along in `hiddenNodes`, so provenance
    survives an edit-then-apply round trip.
    """
    if not handle or handle not in _loaded_graphs or not payload:
        return gr.update(), gr.update()
    try:
        data = json.loads(payload)
    except Exception:
        return gr.update(), "Could not read the canvas edit."
    nodes = [
        GraphNode(node_id=str(n.get("id")), label=str(n.get("label")),
                  props=dict(n.get("props") or {}),
                  status=str(n.get("status") or "found"),
                  evidence=list(n.get("evidence") or []))
        for n in (data.get("nodes") or []) if n.get("id")
    ]
    edges = [
        GraphEdge(subject_id=str(e.get("from")), predicate=str(e.get("predicate")),
                  object_id=str(e.get("to")), props=dict(e.get("props") or {}),
                  status=str(e.get("status") or "found"),
                  evidence=list(e.get("evidence") or []))
        for e in (data.get("edges") or []) if e.get("from") and e.get("to")
    ]
    label = _loaded_graphs[handle][2]
    _loaded_graphs[handle] = (nodes, edges, label)
    return (_summary_text(nodes, edges, label),
            "Canvas edits saved — *Apply to therapy session* to use them.")


def _apply_to_session(handle, session: Session):
    """Hand the corrected graph back to the live session and keep talking.

    The next reply is built from graph.cbt_context(), so corrections take effect
    on the very next turn — that is the whole point of the round-trip.
    """
    if session is None:
        return "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS,
                                       height=THERAPY_CANVAS_H), "No therapy session yet — start one in Tab 1."
    if not handle or handle not in _loaded_graphs:
        return "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS,
                                       height=THERAPY_CANVAS_H), "Load a graph first."
    gnodes, gedges, _ = _loaded_graphs[handle]
    # Not replace_all: the loaded graph is found-only, so a literal swap would
    # delete the 'not yet discovered' placeholder scaffold this panel draws.
    apply_graph(session.graph, gnodes, gedges)
    snap = session.graph.snapshot()
    bar, graph_html = _therapy_view(
        session, "none", snap.get("session_phase", "Rapport"),
        snap.get("active_technique", "Rapport Building"))
    n_found = sum(1 for n in gnodes if n.status == "found")
    return (bar, graph_html,
            f"Applied {n_found} nodes / {len(gedges)} edges to the therapy "
            f"session — go back to Tab 1 and keep talking.")


# ─────────────────────────────────────────────────────────────────────────

# Hides the Trash/"clear" IconButton Gradio bakes into the chatbot toolbar.
# `buttons=["copy"]` puts the copy control on each message instead, so the
# component-level wrapper holds nothing else worth keeping here.
# Injected as a <style> block rather than Blocks(css=...) because Gradio 6 moved
# that parameter to launch(), which never runs under gr.mount_gradio_app.
_UI_CSS = """<style>
:root {
  --app-bg: #f3f6fa;
  --panel: #ffffff;
  --panel-soft: #fbfdff;
  --border: #dbe3ee;
  --border-strong: #c8d4e2;
  --text: #1f2a37;
  --muted: #64748b;
  --subtle: #94a3b8;
  --blue: #2f7dd1;
  --blue-soft: #e8f1fd;
  --green: #24a47a;
  --green-soft: #e4f7f0;
  --amber-soft: #fff4d6;
  --danger-soft: #fdebec;
  --danger: #b4232a;
  --chip-phase-text: #174a7c;
  --chip-caution-text: #805600;
  --chip-muted-bg: #eef2f7;
  --tab-active-bg: rgba(255,255,255,0.72);
  /* Tab labels carry their own colours so both themes clear the 4.5:1
     readable-text threshold; --muted/--blue sat just under it. */
  --tab-text: #5d6b7d;   --tab-text-active: #2a71bf;
  --drop-shadow: rgba(31,42,55,0.06);
  --inset-shadow: rgba(31,42,55,0.04);
}
/* Dark counterpart. Gradio marks dark mode with a .dark class on <html>/<body>;
   this sheet forces its colours with !important, so without an override the app
   stayed light no matter what the toggle said. */
html.dark, body.dark, .dark .gradio-container {
  --app-bg: #0f1216;
  --panel: #181c22;
  --panel-soft: #1d222a;
  --border: #2a313b;
  --border-strong: #3a434f;
  --text: #e5e9ef;
  --muted: #94a3b8;
  --subtle: #6b7787;
  --blue: #5da2e8;
  --blue-soft: #17293c;
  --green: #34d399;
  --green-soft: #13312a;
  --amber-soft: #372c0d;
  --danger-soft: #3a1c1e;
  --danger: #f98a90;
  --chip-phase-text: #a8cff0;
  --chip-caution-text: #f0c674;
  --chip-muted-bg: #232a33;
  --tab-active-bg: rgba(255,255,255,0.06);
  --tab-text: #b6c2d1;   --tab-text-active: #7dbcff;
  --drop-shadow: rgba(0,0,0,0.45);
  --inset-shadow: rgba(0,0,0,0.30);
}
body, .gradio-container {
  background: var(--app-bg) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.gradio-container {
  max-width: none !important;
  padding: 10px 22px 16px !important;
}
.tabs {
  background: transparent !important;
}
.tab-nav {
  border-bottom: 1px solid var(--border) !important;
  gap: 8px !important;
  margin-bottom: 14px !important;
}
.tab-nav button {
  border-radius: 8px 8px 0 0 !important;
  color: var(--tab-text) !important;
  font-weight: 650 !important;
  padding: 10px 14px !important;
}
.tab-nav button.selected,
button[role="tab"].selected {
  color: var(--tab-text-active) !important;
  border-bottom-color: var(--tab-text-active) !important;
  background: var(--tab-active-bg) !important;
}
button[role="tab"] {
  color: var(--muted) !important;
  font-weight: 650 !important;
}
.workspace-row {
  gap: 16px !important;
  align-items: stretch !important;
}
.control-panel, .graph-column {
  min-width: 0 !important;
}
.control-panel {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 14px !important;
  box-shadow: 0 12px 28px var(--drop-shadow);
}
.graph-column {
  background: transparent !important;
}
.session-strip {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 11px;
  margin-bottom: 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel-soft);
}
.session-chip {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11px;
  line-height: 1.2;
  font-weight: 650;
  white-space: nowrap;
}
.session-chip-phase { background: var(--blue-soft); color: var(--chip-phase-text); }
.session-chip-technique {
  border: 1px solid var(--border-strong);
  background: var(--panel);
  color: var(--muted);
}
.session-chip-warn { background: var(--danger-soft); color: var(--danger); }
.session-chip-caution { background: var(--amber-soft); color: var(--chip-caution-text); }
.session-chip-muted { background: var(--chip-muted-bg); color: var(--muted); }
.session-turn {
  margin-left: auto;
  color: var(--subtle);
  font-size: 11px;
  font-weight: 650;
}
#therapy_chat {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--panel) !important;
  overflow: hidden !important;
}
/* Hides Gradio's undisableable chatbot-level Trash/"clear" control. The
   per-message copy+edit group is ALSO an .icon-button-wrapper (Gradio ships
   `.message-buttons-right .icon-button-wrapper`), so the blanket hide took the
   edit pencil down with it — restore anything nested under .message-buttons.
   `flex` is the value Gradio's own .icon-button-wrapper rule uses. */
#therapy_chat .icon-button-wrapper { display: none !important; }
#therapy_chat .message-buttons .icon-button-wrapper { display: flex !important; }
.message, .message-row {
  border-radius: 8px !important;
}
.bubble-wrap .message {
  box-shadow: none !important;
}
textarea, input, select {
  border-color: var(--border-strong) !important;
  border-radius: 8px !important;
}
textarea:focus, input:focus, select:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(47,125,209,0.12) !important;
}
button.primary, .primary > button {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
}
/* Gradio's base button has no border-width, so setting border-color alone left
   the secondary buttons outline-less. Declare the full shorthand. */
button.secondary,
.secondary > button,
button[class*="secondary"] {
  border: 1px solid var(--border-strong) !important;
  background: var(--panel) !important;
  color: var(--text) !important;
  box-shadow: 0 1px 1px var(--inset-shadow) !important;
}
button.secondary:hover,
.secondary > button:hover,
button[class*="secondary"]:hover {
  border-color: var(--blue) !important;
  background: var(--blue-soft) !important;
  color: var(--chip-phase-text) !important;
}

/* Composer: the textbox sits in a Gradio form wrapper whose padding made Send
   sit a few px off and stretch with the row. Pin both to one height and stop
   the button growing. */
.composer { align-items: center !important; gap: 8px !important; }
.composer textarea, .composer input[type="text"] {
  min-height: 42px !important;
  padding-top: 11px !important;
  padding-bottom: 11px !important;
}
.composer button {
  min-height: 42px !important;
  height: 42px !important;
  flex: 0 0 auto !important;
  min-width: 104px !important;
  max-width: 132px !important;
  font-weight: 650 !important;
}
/* Paired action buttons share the row evenly. */
.action-pair button { min-height: 38px !important; }
.section-title {
  margin: 2px 0 8px;
  color: var(--text);
  font-size: 13px;
  font-weight: 750;
}
.status-md, .branch-md {
  color: var(--muted);
  font-size: 12px;
}
.compact-tabs .tab-nav {
  margin-bottom: 10px !important;
}
.compact-tabs .tab-nav button {
  padding: 8px 10px !important;
  font-size: 12px !important;
}
#graph_sync, #edit_token { display: none !important; }
iframe {
  box-shadow: 0 12px 28px var(--drop-shadow);
}
@media (max-width: 900px) {
  .gradio-container { padding: 14px !important; }
  .workspace-row { gap: 12px !important; }
  .control-panel { padding: 12px !important; }
}
</style>"""

# Installed once on page load: relays the canvas iframe's postMessage into the
# hidden #graph_sync textbox and fires an input event, which is what Gradio
# listens for. The timestamp guarantees the value differs every time, so .change
# still fires when the same edit is repeated.
_SYNC_JS = """
() => {
  if (window.__cbtSyncInstalled) return;
  window.__cbtSyncInstalled = true;
  window.addEventListener('message', (e) => {
    const d = e.data;
    if (!d || d.kind !== 'cbt_graph_sync') return;
    const el = document.querySelector('#graph_sync textarea, #graph_sync input');
    if (!el) return;
    el.value = JSON.stringify({nodes: d.nodes, edges: d.edges, t: Date.now()});
    el.dispatchEvent(new Event('input', {bubbles: true}));
  });

  // Tell every canvas iframe which theme Gradio is actually in. They are separate
  // documents, so prefers-color-scheme is all they can see by themselves — and it
  // is wrong whenever Gradio's toggle (or ?__theme=) disagrees with the OS.
  const isDark = () => document.documentElement.classList.contains('dark') ||
                       document.body.classList.contains('dark');
  const broadcast = () => {
    const dark = isDark();
    document.querySelectorAll('iframe').forEach((f) => {
      try { f.contentWindow.postMessage({kind: 'cbt_theme', dark}, '*'); } catch (err) {}
    });
  };
  // Re-send on toggle, and on any re-render that swaps an iframe in.
  new MutationObserver(broadcast).observe(document.documentElement,
    {attributes: true, attributeFilter: ['class']});
  new MutationObserver(broadcast).observe(document.body, {childList: true, subtree: true});
  broadcast();
  setTimeout(broadcast, 300);
}
"""

with gr.Blocks(title="CBT V4_flat — Therapy + Query", fill_height=True) as demo:
    gr.HTML(_UI_CSS, visible=True, padding=False)
    demo.load(None, None, None, js=_SYNC_JS)
    session_state = gr.State(None)
    pending_msg = gr.State("")

    with gr.Tabs(elem_classes=["main-tabs"]):
        # ── Tab 1: Therapy ───────────────────────────────────────────
        with gr.Tab("Therapy (Part 1)"):
            with gr.Row(equal_height=False, elem_classes=["workspace-row"]):
                # Left column: chat
                with gr.Column(scale=2, elem_classes=["control-panel"]):
                    session_bar = gr.HTML(value="")
                    chatbot = gr.Chatbot(
                        height=THERAPY_CHAT_H,
                        show_label=False,
                        # Gradio's default toolbar is ["share", "copy_all"], and
                        # "share" opens a Hugging Face Spaces Discussions panel —
                        # useless here and startling. Ask for the per-message
                        # copy button only.
                        buttons=["copy"],
                        # Puts a pencil next to that copy button on client
                        # messages; editing one fires .edit() below.
                        editable="user",
                        # Gradio always renders a Trash/"clear" IconButton with no
                        # option to disable it; hidden via CSS on this id below.
                        elem_id="therapy_chat",
                    )
                    with gr.Row(equal_height=True, elem_classes=["composer"]):
                        msg_box = gr.Textbox(
                            placeholder="Share what's on your mind…",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    # Two equal columns, each button directly under the control it
                    # acts on: steering / New session · versions / Switch. The
                    # previous full-width dropdown-then-full-width-button stack was
                    # both lopsided and three rows taller. `info=` lines are dropped
                    # — each one cost a row, and the hints now live in the labels.
                    with gr.Row(equal_height=True):
                        strategy_dd = gr.Dropdown(
                            choices=["none", "Question", "Affirmation and Reassurance",
                                     "Self-disclosure", "Reflection of feelings",
                                     "Info+Suggest"],
                            value="none", label="Steering strategy", scale=1,
                        )
                        branch_dd = gr.Dropdown(
                            choices=[], label="Versions (edit a message to add one)",
                            interactive=True, scale=1,
                        )
                    with gr.Row(equal_height=True, elem_classes=["action-pair"]):
                        reset_btn = gr.Button("New session", scale=1)
                        switch_btn = gr.Button("Switch version", scale=1)
                    branch_status = gr.Markdown("")

                    with gr.Accordion("Demo script", open=False):
                        gr.Examples(
                            examples=[[m] for m in DEMO_SCRIPT],
                            inputs=[msg_box],
                            label="",
                            examples_per_page=6,
                        )

                # Right column: live graph
                with gr.Column(scale=3, elem_classes=["graph-column"]):
                    graph_panel = gr.HTML()

            therapy_outputs = [chatbot, session_state, session_bar, graph_panel]
            branch_outputs = therapy_outputs + [branch_dd, branch_status]
            # Hidden relay: the inline-edit event must not list `chatbot` as an
            # output (Gradio would draw a pending bubble under the last message on
            # every trigger, cancel included), so it writes a token here and the
            # token's .change is what re-renders the chat. Hidden via CSS rather
            # than visible=False so the component is really rendered.
            edit_token = gr.Textbox(elem_id="edit_token", value="",
                                    label="", show_label=False)
            edit_outputs = [session_state, session_bar, graph_panel,
                            branch_dd, branch_status, edit_token]

            send_btn.click(
                _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
            ).then(
                _bot_respond, [pending_msg, chatbot, session_state, strategy_dd], therapy_outputs
            ).then(
                _branch_controls, [session_state], [branch_dd]
            )
            msg_box.submit(
                _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
            ).then(
                _bot_respond, [pending_msg, chatbot, session_state, strategy_dd], therapy_outputs
            ).then(
                _branch_controls, [session_state], [branch_dd]
            )
            reset_btn.click(_reset_therapy, [], therapy_outputs).then(
                _branch_controls, [session_state], [branch_dd]
            )
            demo.load(_reset_therapy, [], therapy_outputs).then(
                _branch_controls, [session_state], [branch_dd]
            )

            edit_token.change(_refresh_chat, [session_state], [chatbot])
            chatbot.edit(
                _on_edit_message, [session_state, chatbot, strategy_dd],
                edit_outputs,
                # Gradio fires `edit` when the inline editor closes — including
                # on cancel — and paints a progress overlay for any round-trip,
                # however fast. That put a spinner on screen for pressing ✗ with
                # nothing changed. The handler reports its own progress in the
                # status line, so the overlay is pure noise here.
                show_progress="hidden",
            )
            switch_btn.click(
                _do_switch_branch, [session_state, branch_dd, strategy_dd],
                branch_outputs,
            )

        # ── Tab 2: Query ─────────────────────────────────────────────
        with gr.Tab("Query (Part 2)"):
            handle_state = gr.State(None)

            with gr.Row(equal_height=False, elem_classes=["workspace-row"]):
                # Left column: load + summary + query chat
                with gr.Column(scale=2, elem_classes=["control-panel"]):
                    gr.Markdown('<div class="section-title">Load a graph</div>')
                    with gr.Tabs(elem_classes=["compact-tabs"]):
                        with gr.Tab("Live session"):
                            live_btn = gr.Button("Load current therapy session")
                        with gr.Tab("Upload JSON"):
                            json_file = gr.File(
                                label="V4_flat Stage 5 export",
                                file_types=[".json"],
                            )
                            json_btn = gr.Button("Load JSON")
                        with gr.Tab("Neo4j"):
                            neo_uri = gr.Textbox(
                                label="URI", placeholder="bolt://localhost:7687"
                            )
                            neo_user = gr.Textbox(label="User", value="neo4j")
                            neo_pw = gr.Textbox(label="Password", type="password")
                            neo_btn = gr.Button("Connect & load")
                    summary_md = gr.Markdown("_Load a graph to start._", elem_classes=["status-md"])

                    gr.Markdown('<div class="section-title">Ask</div>')
                    query_chat = gr.Chatbot(
                        height=300,
                        show_label=False,
                        buttons=["copy"],   # drop the HF "share" panel here too
                    )
                    with gr.Row():
                        question_box = gr.Textbox(
                            placeholder="e.g. What automatic thoughts came up?",
                            show_label=False,
                            scale=5,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)

                # Right column: editable graph
                with gr.Column(scale=3, elem_classes=["graph-column"]):
                    query_graph_panel = gr.HTML(
                        value=_render_canvas([], [], edit_mode=True)
                    )

                    # Edits are made in the canvas above; this only commits them.
                    apply_btn = gr.Button("Apply to therapy session",
                                          variant="primary")
                    repair_status = gr.Markdown("")
                    # Carries the edited graph from the iframe back to Python.
                    # Hidden via CSS rather than visible=False, because an
                    # invisible component is not rendered and the browser-side
                    # listener would have nothing to write into.
                    graph_sync = gr.Textbox(elem_id="graph_sync", value="",
                                            label="", show_label=False)

            load_outputs = [handle_state, summary_md, query_graph_panel, query_chat, question_box]

            live_btn.click(_load_live, [session_state], load_outputs)
            json_btn.click(_load_json, [json_file], load_outputs)
            neo_btn.click(_load_neo4j, [neo_uri, neo_user, neo_pw], load_outputs)

            graph_sync.change(_sync_canvas_edits, [handle_state, graph_sync],
                              [summary_md, repair_status])
            apply_btn.click(_apply_to_session, [handle_state, session_state],
                            [session_bar, graph_panel, repair_status])

            ask_btn.click(
                _query_ask, [handle_state, question_box, query_chat],
                [query_chat, question_box],
            )
            question_box.submit(
                _query_ask, [handle_state, question_box, query_chat],
                [query_chat, question_box],
            )
