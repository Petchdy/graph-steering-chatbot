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
                      switch_branch, _client_message_at)

# ─────────────────────────────────────────────────────────────────────────
# Color / style constants
# ─────────────────────────────────────────────────────────────────────────

NODE_COLORS: dict[str, tuple[str, str, str]] = {
    # label: (fill, stroke, text)
    "Client":              ("#E5E7EB", "#D1D5DB", "#1F2937"),
    "Session":             ("#E5E7EB", "#D1D5DB", "#1F2937"),
    "Problem":             ("#F87171", "#EF4444", "#FFFFFF"),
    "Goal":                ("#34D399", "#10B981", "#1F2937"),
    "Intervention":        ("#A78BFA", "#8B5CF6", "#FFFFFF"),
    "Homework":            ("#FBBF24", "#F59E0B", "#1F2937"),
    "CoreBelief":          ("#9D174D", "#831843", "#FFFFFF"),
    "IntermediateBelief":  ("#BE185D", "#9D174D", "#FFFFFF"),
    "Situation":           ("#FDE047", "#FACC15", "#1F2937"),
    "AutomaticThought":    ("#6EE7B7", "#34D399", "#1F2937"),
    "Reaction":            ("#FCA5A5", "#F87171", "#1F2937"),
    "AdaptiveResponse":    ("#D1FAE5", "#6EE7B7", "#065F46"),
    "Utterance":           ("#D1D5DB", "#9CA3AF", "#1F2937"),
}
_MISSING_COLORS = ("#F5F5F5", "#AAAAAA", "#AAAAAA")

_COLOR = {k: v[1] for k, v in NODE_COLORS.items()}    # stroke
_BADGE_BG = {k: v[0] for k, v in NODE_COLORS.items()}  # fill
_BADGE_COLOR = {k: v[2] for k, v in NODE_COLORS.items()}  # text
_COLOR["missing"] = _MISSING_COLORS[1]

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
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px; background: #fafafa; }
.shell { display: flex; flex-direction: column; height: 610px; background: #fafafa; overflow: hidden; }
.graph-header { display: flex; align-items: center; justify-content: space-between;
  padding: 8px 14px; border-bottom: 0.5px solid #e5e7eb; background: #fff; flex-shrink: 0; }
.graph-title { font-size: 12px; font-weight: 500; color: #555; }
.graph-actions { display: flex; gap: 6px; align-items: center; }
.btn-sm { font-size: 11px; padding: 4px 10px; border-radius: 6px;
  border: 0.5px solid #d1d5db; background: transparent; cursor: pointer; color: #555; }
.btn-sm:hover { background: #f3f4f6; }
.btn-sm.primary { background: #D85A30; color: #fff; border-color: #D85A30; }
.btn-sm.primary:hover { background: #993C1D; }
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: #1D9E75;
  display: inline-block; margin-right: 5px; vertical-align: middle; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.workspace { display: flex; flex: 1; overflow: hidden; position: relative; }
.graph-panel { flex: 1; display: flex; flex-direction: column; position: relative; background: #fff; }
canvas { position: absolute; top: 0; left: 0; cursor: pointer; }
.legend { padding: 7px 14px; border-top: 0.5px solid #e5e7eb;
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  flex-shrink: 0; background: #fff; margin-top: auto; }
.leg { display: flex; align-items: center; gap: 4px; font-size: 10px; color: #777; }
.ld { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.detail-panel { width: 240px; flex-shrink: 0; display: flex; flex-direction: column;
  border-left: 0.5px solid #e5e7eb; background: #fff; overflow-y: auto; }
.dp-header { padding: 10px 14px 8px; border-bottom: 0.5px solid #e5e7eb;
  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.dp-title { font-size: 12px; font-weight: 500; color: #222; }
.dp-close { font-size: 16px; color: #aaa; cursor: pointer; border: none; background: none; padding: 0; }
.dp-close:hover { color: #222; }
.dp-empty { padding: 24px 14px; text-align: center; color: #aaa; font-size: 12px; line-height: 1.6; }
.dp-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; flex: 1; }
.dp-label-badge { display: inline-block; font-size: 10px; font-weight: 500;
  padding: 2px 8px; border-radius: 20px; margin-bottom: 4px; }
.dp-field { display: flex; flex-direction: column; gap: 3px; }
.dp-field label { font-size: 10px; color: #999; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.05em; }
.dp-field input, .dp-field select, .dp-field textarea {
  font-size: 12px; padding: 5px 8px; border-radius: 6px;
  border: 0.5px solid #d1d5db; background: #f9fafb; color: #222;
  width: 100%; font-family: inherit; resize: none; }
.dp-field textarea { min-height: 52px; }
.dp-field input:focus, .dp-field select:focus, .dp-field textarea:focus
  { outline: none; border-color: #D85A30; }
.dp-actions { padding: 10px 14px; border-top: 0.5px solid #e5e7eb;
  display: flex; gap: 6px; flex-shrink: 0; }
.dp-actions button { flex: 1; font-size: 11px; padding: 6px; border-radius: 6px;
  border: 0.5px solid #d1d5db; background: transparent; cursor: pointer; color: #555; }
.dp-actions button.save { background: #D85A30; color: #fff; border-color: #D85A30; }
.dp-actions button.del { color: #E24B4A; border-color: #E24B4A; }
.dp-actions button:hover { filter: brightness(0.92); }
.create-modal { position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.3); display: flex; align-items: center;
  justify-content: center; z-index: 100; }
.modal-box { background: #fff; border-radius: 10px; border: 0.5px solid #d1d5db;
  padding: 16px; width: 220px; display: flex; flex-direction: column; gap: 10px; }
.modal-title { font-size: 13px; font-weight: 500; color: #222; }
.modal-field { display: flex; flex-direction: column; gap: 4px; }
.modal-field label { font-size: 10px; color: #999; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.05em; }
.modal-field select, .modal-field input, .modal-field textarea {
  font-size: 12px; padding: 5px 8px; border-radius: 6px;
  border: 0.5px solid #d1d5db; background: #f9fafb; color: #222;
  width: 100%; font-family: inherit; }
.modal-field textarea { min-height: 48px; resize: none; }
.modal-actions { display: flex; gap: 6px; }
.modal-actions button { flex: 1; font-size: 11px; padding: 6px; border-radius: 6px;
  border: 0.5px solid #d1d5db; background: transparent; cursor: pointer; color: #555; }
.modal-actions button.confirm { background: #D85A30; color: #fff; border-color: #D85A30; }
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
        <div class="leg"><div style="width:16px;height:1.5px;background:#1D9E75;"></div>Found</div>
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
const COLOR = __COLOR__;
const BADGE_BG = __BADGE_BG__;
const BADGE_COLOR = __BADGE_CLR__;
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
const SUBROW_GAP = 64;
const MIN_SEP = 68;

function applyLayout(W, H) {
  const MARGIN = 48;
  const RIGHT_W = 160;

  // Virtual canvas scales with node count so dense graphs breathe
  const SCALE = Math.max(1, Math.sqrt(nodes.length / 18));
  const VW = W * SCALE, VH = H * SCALE;
  const vM = MARGIN * SCALE;
  const vMAIN_W = VW - RIGHT_W * SCALE - vM * 2;

  const layerGroups = {}, rightGroups = {};
  for (const n of nodes) {
    if (RIGHT_SIDE.has(n.label)) {
      (rightGroups[n.label] = rightGroups[n.label] || []).push(n);
    } else {
      const l = LAYERS[n.label] !== undefined ? LAYERS[n.label] : 6;
      (layerGroups[l] = layerGroups[l] || []).push(n);
    }
  }

  const mainLayers = Object.keys(layerGroups).map(Number).sort(function(a,b){return a-b;});
  const totalLayers = mainLayers.length;
  const nodeBaseY = {};

  // Hierarchical slot assignment — dense layers wrap into sub-rows
  mainLayers.forEach(function(l, layerIndex) {
    const row = layerGroups[l];
    const subRows = Math.ceil(row.length / MAX_PER_ROW);
    const layerH = totalLayers > 1 ? (VH - vM * 2) / (totalLayers - 1) : VH / 2;
    const yBase = vM + layerIndex * layerH;
    for (let i = 0; i < row.length; i++) {
      const sr = Math.floor(i / MAX_PER_ROW);
      const idxInSr = i % MAX_PER_ROW;
      const cntInSr = Math.min(MAX_PER_ROW, row.length - sr * MAX_PER_ROW);
      const slotW = vMAIN_W / cntInSr;
      row[i].x = vM + slotW * idxInSr + slotW / 2;
      row[i].y = yBase + (sr - (subRows - 1) / 2) * SUBROW_GAP * SCALE;
      nodeBaseY[row[i].id] = row[i].y;
    }
  });

  // Right-side nodes (Intervention, Homework)
  const rightLabels = Object.keys(rightGroups);
  rightLabels.forEach(function(label, li) {
    const group = rightGroups[label];
    const slotH = rightLabels.length > 0 ? (VH - vM * 2) / rightLabels.length : VH - vM * 2;
    const slotStart = vM + li * slotH;
    const itemH = group.length > 1 ? slotH / group.length : slotH;
    group.forEach(function(n, i) {
      n.x = VW - RIGHT_W * SCALE / 2;
      n.y = slotStart + itemH * i + itemH / 2;
    });
  });

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
        n.x = Math.max(VW - RIGHT_W * SCALE - 10, Math.min(VW - 30, n.x + force[n.id].x * step));
        n.y = Math.max(30, Math.min(VH - 30, n.y + force[n.id].y * step));
      } else {
        const yBase = nodeBaseY[n.id] !== undefined ? nodeBaseY[n.id] : VH / 2;
        n.x = Math.max(vM + 20, Math.min(vM + vMAIN_W - 20, n.x + force[n.id].x * step));
        n.y = Math.max(yBase - 25 * SCALE, Math.min(yBase + 25 * SCALE, n.y + force[n.id].y * step));
      }
    }
  }

  // MIN_SEP hard-separation relaxation pass
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx, dy) || 1;
      if (d < MIN_SEP) {
        const push = (MIN_SEP - d) / 2, ux = dx / d, uy = dy / d;
        a.x -= ux * push; a.y -= uy * push * 0.5;
        b.x += ux * push; b.y += uy * push * 0.5;
      }
    }
  }
}

// ── Canvas setup ──────────────────────────────────────────────────────────
const gp = document.getElementById('gp');
const cv = document.getElementById('gc');
const ctx = cv.getContext('2d');
let dpr = window.devicePixelRatio || 1;
let lastNodeCount = 0;
let W = 1, H = 1;

// ── Viewport state ────────────────────────────────────────────────────────
let view = { scale: 1, tx: 0, ty: 0 };
let panning = null;

function toWorld(sx, sy) {
  return { x: (sx - view.tx) / view.scale, y: (sy - view.ty) / view.scale };
}

function zoomToFit(pad) {
  pad = pad !== undefined ? pad : 40;
  if (!nodes.length) return;
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for (const n of nodes) {
    minX=Math.min(minX,n.x); minY=Math.min(minY,n.y);
    maxX=Math.max(maxX,n.x); maxY=Math.max(maxY,n.y);
  }
  const gw=(maxX-minX)||1, gh=(maxY-minY)||1;
  view.scale = Math.max(0.2, Math.min(3, Math.min((W-pad*2)/gw, (H-pad*2)/gh)));
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
  if (nodes.length > 0 && w > 50 && h > 50 && nodes.length !== lastNodeCount) {
    applyLayout(w, h);
    lastNodeCount = nodes.length;
    zoomToFit();
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

document.getElementById('btnFit').addEventListener('click', function() { zoomToFit(); });
document.getElementById('btnZoomIn').addEventListener('click', function() {
  const before = toWorld(W/2, H/2);
  view.scale = Math.min(3, view.scale * 1.2);
  view.tx = W/2 - before.x * view.scale; view.ty = H/2 - before.y * view.scale; draw();
});
document.getElementById('btnZoomOut').addEventListener('click', function() {
  const before = toWorld(W/2, H/2);
  view.scale = Math.max(0.2, view.scale / 1.2);
  view.tx = W/2 - before.x * view.scale; view.ty = H/2 - before.y * view.scale; draw();
});
document.getElementById('btnReset').addEventListener('click', function() {
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
    const col = sel ? '#D85A30' : (isFound ? '#1D9E75' : '#bbb');
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
    ctx.fillStyle = sel ? '#D85A30' : (isFound ? '#0F6E56' : '#aaa');
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
    const col    = isMissing ? COLOR['missing'] : (COLOR[n.label] || '#aaa');
    const bgCol  = isMissing ? '#F5F5F5' : (BADGE_BG[n.label] || '#eee');
    const isRect = RECT_LABELS.has(n.label);
    // §UI-1.1 §2.1: use BADGE_COLOR for both text lines, not stroke colour
    const nodeTextCol = isMissing ? '#9aa0a6' : (BADGE_COLOR[n.label] || '#1F2937');

    if (sel || efrom) { ctx.shadowColor = efrom ? '#378ADD' : '#D85A30'; ctx.shadowBlur = 10; }
    ctx.fillStyle = bgCol; ctx.strokeStyle = sel ? '#D85A30' : col; ctx.lineWidth = sel ? 2.2 : 1.5;
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
  const bb = BADGE_BG[n.label]||'#eee', bc = BADGE_COLOR[n.label]||'#333';
  const sBg = n.status==='found'?'#E1F5EE':'#f0f0f0', sFg = n.status==='found'?'#0F6E56':'#888';
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
  const sBg = e.status==='found'?'#E1F5EE':'#f0f0f0', sFg = e.status==='found'?'#0F6E56':'#888';
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

function saveNode(id) {
  const n = nodes.find(function(x){return x.id===id;});
  if (!n) return;
  n.label = document.getElementById('dpClass').value;
  n.status = document.getElementById('dpStatus').value;
  try { n.props = JSON.parse(document.getElementById('dpProps').value||'{}'); } catch(err) {}
  draw(); syncGraph(); showNodePanel(n);
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
  clearSelection(); draw(); syncGraph();
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
  nodes.push({id:cls.toLowerCase().slice(0,3)+'_'+Date.now(), label:cls,
    x:60+Math.random()*(W-120), y:60+Math.random()*(H-120),
    status:'found', props:propKey?{[propKey]:txt}:{}, evidence:[]});
  closeModal(); updateTitle(); draw(); syncGraph();
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
            f'<div class="leg"><div class="ld" style="background:{fill};'
            f'border:1px solid {stroke};{radius}"></div>{label}</div>'
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
    classes_json = json.dumps(_NODE_CLASSES)
    predicates_json = json.dumps(_PREDICATES)

    filled = (
        _CANVAS_TEMPLATE
        .replace("__NODES__", nodes_json)
        .replace("__EDGES__", edges_json)
        .replace("__HIDDEN_NODES__", hidden_json)
        .replace("__LEGEND__", _legend_html(hide_legend_labels,
                                            row_breaks=not hide_legend_labels))
        .replace("__EDIT_MODE__", edit_str)
        .replace("__COLOR__", color_json)
        .replace("__BADGE_BG__", badge_bg_json)
        .replace("__BADGE_CLR__", badge_clr_json)
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
            steer = (f'<span style="font-size:11px;padding:3px 10px;border-radius:20px;'
                     f'background:#FDECEC;color:#B4232A;">steer: {html.escape(strategy)}</span>')
        elif steer_status == "fallback":
            # The steering service errored/returned nothing for THIS turn and
            # SteeredRemoteGenerator silently fell back to the plain Ollama reply.
            steer = (f'<span style="font-size:11px;padding:3px 10px;border-radius:20px;'
                     f'background:#FFF3CD;color:#8A6100;" '
                     f'title="Steering service unavailable this turn — used the plain reply instead.">'
                     f'steer: {html.escape(strategy)} (fallback)</span>')
        else:
            # steer_status is None — the active generator has no steering concept at all
            # (GENERATOR != steered), so the dropdown selection was never even attempted, not
            # just unavailable for one turn. Distinct from "fallback" so this doesn't read as a
            # transient hiccup — it means steering isn't wired up for this session at all.
            steer = (f'<span style="font-size:11px;padding:3px 10px;border-radius:20px;'
                     f'background:#EEE;color:#888;" '
                     f'title="This session\'s generator has no steering support — set GENERATOR=steered '
                     f'and restart the chatbot for the dropdown to take effect.">'
                     f'steer: {html.escape(strategy)} (inactive)</span>')
    return (
        '<div style="display:flex;align-items:center;gap:8px;padding:8px 4px;">'
        f'<span style="font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;'
        f'background:#E6F1FB;color:#185FA5;">{html.escape(phase)}</span>'
        f'<span style="font-size:11px;padding:3px 10px;border-radius:20px;'
        f'border:0.5px solid #d1d5db;color:#666;">{html.escape(technique)}</span>'
        f'{steer}'
        f'<span style="font-size:11px;color:#aaa;margin-left:auto;">Turn {turn_count}</span>'
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
                           hide_legend_labels=THERAPY_HIDDEN_LABELS)
    return history, session, bar_html, graph_html


def _reset_therapy():
    session = _new_session()
    history = [{"role": "assistant", "content": INTRO}]
    bar_html = _session_bar_html("Rapport", "Rapport Building", 0)
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges(),
                                      skip_scaffold=True)
    graph_html = _render_canvas(nodes, edges, edit_mode=False,
                           hide_legend_labels=THERAPY_HIDDEN_LABELS)
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
                           hide_legend_labels=THERAPY_HIDDEN_LABELS)


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


def _on_edit_message(session: Session, chat_history: list, strategy: str,
                     edit_data: gr.EditData):
    """Inline pencil-edit of a client message = rewrite that turn."""
    turn_index = _message_index_to_turn(chat_history, edit_data.index)
    new_text = edit_data.value
    if isinstance(new_text, dict):
        new_text = new_text.get("content", "")
    return _do_rewrite(session, turn_index, str(new_text or ""), strategy)


def _do_rewrite(session: Session, turn_index, new_message: str,
                strategy: str = "none"):
    """Replay a past client turn with different words, keeping both versions."""
    if session is None or turn_index in (None, "") or not (new_message or "").strip():
        branch_dd = _branch_controls(session)
        snap = session.graph.snapshot() if session else {}
        bar, graph_html = (_therapy_view(session, strategy,
                                         snap.get("session_phase", "Rapport"),
                                         snap.get("active_technique", "Rapport Building"))
                           if session else ("", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS)))
        return (_history_to_chat(session) if session else [], session, bar,
                graph_html, branch_dd,
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
                "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS), branch_dd, "No version selected.")
    try:
        switch_branch(session, branch_id)
    except ValueError as exc:
        branch_dd = _branch_controls(session)
        return (_history_to_chat(session), session, "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS),
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
        return "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS), "No therapy session yet — start one in Tab 1."
    if not handle or handle not in _loaded_graphs:
        return "", _render_canvas([], [], hide_legend_labels=THERAPY_HIDDEN_LABELS), "Load a graph first."
    gnodes, gedges, _ = _loaded_graphs[handle]
    session.graph.replace_all(gnodes, gedges)
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
#therapy_chat .icon-button-wrapper { display: none !important; }
#graph_sync { display: none !important; }
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
}
"""

with gr.Blocks(title="CBT V4_flat — Therapy + Query", fill_height=True) as demo:
    gr.HTML(_UI_CSS, visible=True, padding=False)
    demo.load(None, None, None, js=_SYNC_JS)
    session_state = gr.State(None)
    pending_msg = gr.State("")

    with gr.Tabs():
        # ── Tab 1: Therapy ───────────────────────────────────────────
        with gr.Tab("Therapy (Part 1)"):
            with gr.Row(equal_height=False):
                # Left column: chat
                with gr.Column(scale=2):
                    session_bar = gr.HTML(value="")
                    chatbot = gr.Chatbot(
                        height=420,
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
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Share what's on your mind…",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    strategy_dd = gr.Dropdown(
                        choices=["none", "Question", "Affirmation and Reassurance",
                                 "Self-disclosure", "Reflection of feelings", "Info+Suggest"],
                        value="none", label="Steering strategy",
                        info="Manually steer the therapist's next reply (needs GENERATOR=steered).",
                    )
                    reset_btn = gr.Button("New session")

                    with gr.Accordion("Demo script", open=False):
                        gr.Examples(
                            examples=[[m] for m in DEMO_SCRIPT],
                            inputs=[msg_box],
                            label="",
                            examples_per_page=6,
                        )

                    with gr.Row():
                        branch_dd = gr.Dropdown(
                            choices=[], label="Versions of this turn",
                            interactive=True, scale=4,
                            info="Edit any message of yours (pencil icon) to make "
                                 "another version. ● = the one you're in.",
                        )
                        switch_btn = gr.Button("Switch", scale=1)
                    branch_status = gr.Markdown("")

                # Right column: live graph
                with gr.Column(scale=3):
                    graph_panel = gr.HTML()

            therapy_outputs = [chatbot, session_state, session_bar, graph_panel]
            branch_outputs = therapy_outputs + [branch_dd, branch_status]

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

            chatbot.edit(
                _on_edit_message, [session_state, chatbot, strategy_dd],
                branch_outputs,
            )
            switch_btn.click(
                _do_switch_branch, [session_state, branch_dd, strategy_dd],
                branch_outputs,
            )

        # ── Tab 2: Query ─────────────────────────────────────────────
        with gr.Tab("Query (Part 2)"):
            handle_state = gr.State(None)

            with gr.Row(equal_height=False):
                # Left column: load + summary + query chat
                with gr.Column(scale=2):
                    gr.Markdown("### Load a graph")
                    with gr.Tabs():
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
                    summary_md = gr.Markdown("_Load a graph to start._")

                    gr.Markdown("### Ask")
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
                with gr.Column(scale=3):
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
