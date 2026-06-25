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
from .therapy import Session, turn

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

INTRO = (
    "Hello, and welcome. I'm glad you're here today. "
    "This is a safe space to talk about whatever is on your mind. "
    "What's been weighing on you lately, or what would you most like to explore today?"
)

# ─────────────────────────────────────────────────────────────────────────
# Data conversion helper
# ─────────────────────────────────────────────────────────────────────────

def _build_canvas_data(
    graph_nodes: list[GraphNode],
    graph_edges: list[GraphEdge],
    skip_utterance: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Convert GraphNode/GraphEdge lists to canvas-friendly dicts.

    Filters out Utterance nodes (too noisy) and edges where BOTH endpoints
    are missing (placeholder-only noise at startup).
    """
    filtered_nodes = [
        n for n in graph_nodes
        if not (skip_utterance and n.label == "Utterance")
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
        # Skip edges where both endpoints are missing (startup noise)
        if (node_status.get(e.subject_id) == "missing" and
                node_status.get(e.object_id) == "missing"):
            continue
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
        <div class="leg"><div class="ld" style="background:#E5E7EB;border:1px solid #D1D5DB;"></div>Client</div>
        <div class="leg"><div class="ld" style="background:#E5E7EB;border:1px solid #D1D5DB;"></div>Session</div>
        <div class="leg"><div class="ld" style="background:#F87171;border:1px solid #EF4444;border-radius:2px;"></div>Problem</div>
        <div class="leg"><div class="ld" style="background:#34D399;border:1px solid #10B981;border-radius:2px;"></div>Goal</div>
        <div class="leg"><div class="ld" style="background:#A78BFA;border:1px solid #8B5CF6;"></div>Intervention</div>
        <div class="leg"><div class="ld" style="background:#FBBF24;border:1px solid #F59E0B;"></div>Homework</div>
        <div style="width:100%;height:0;"></div>
        <div class="leg"><div class="ld" style="background:#9D174D;border:1px solid #831843;"></div>CoreBelief</div>
        <div class="leg"><div class="ld" style="background:#BE185D;border:1px solid #9D174D;"></div>IntermBelief</div>
        <div class="leg"><div class="ld" style="background:#FDE047;border:1px solid #FACC15;"></div>Situation</div>
        <div class="leg"><div class="ld" style="background:#6EE7B7;border:1px solid #34D399;"></div>AutoThought</div>
        <div class="leg"><div class="ld" style="background:#FCA5A5;border:1px solid #F87171;"></div>Reaction</div>
        <div class="leg"><div class="ld" style="background:#D1FAE5;border:1px solid #6EE7B7;"></div>AdaptResponse</div>
        <div style="width:100%;height:0;"></div>
        <div class="leg"><div class="ld" style="background:#D1D5DB;border:1px solid #9CA3AF;"></div>Utterance</div>
        <div class="leg"><div style="width:16px;height:1.5px;background:#1D9E75;"></div>Found</div>
        <div class="leg"><div style="width:16px;height:1.5px;background:repeating-linear-gradient(90deg,#bbb 0,#bbb 3px,transparent 3px,transparent 6px);"></div>Placeholder</div>
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

// ── Layout ──────────────────────────────────────────────────────────────
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

function applyLayout(W, H) {
  const MARGIN = 48;
  const RIGHT_W = 160;
  const MAIN_W = W - RIGHT_W - MARGIN * 2;

  // Bucket nodes into layer groups and right-side groups
  const layerGroups = {};
  const rightGroups = {};
  for (const n of nodes) {
    if (RIGHT_SIDE.has(n.label)) {
      if (!rightGroups[n.label]) rightGroups[n.label] = [];
      rightGroups[n.label].push(n);
    } else {
      const l = LAYERS[n.label] !== undefined ? LAYERS[n.label] : 6;
      if (!layerGroups[l]) layerGroups[l] = [];
      layerGroups[l].push(n);
    }
  }

  const mainLayers = Object.keys(layerGroups).map(Number).sort(function(a,b){return a-b;});
  const totalLayers = mainLayers.length;
  const nodeBaseY = {};

  // Step 1 — hierarchical slot assignment
  mainLayers.forEach(function(l, layerIndex) {
    const row = layerGroups[l];
    const count = row.length;
    const slotW = MAIN_W / count;
    const layerH = totalLayers > 1 ? (H - MARGIN * 2) / (totalLayers - 1) : H / 2;
    const yPos = MARGIN + layerIndex * layerH;
    row.forEach(function(n, i) {
      n.x = MARGIN + slotW * i + slotW / 2;
      n.y = yPos;
      nodeBaseY[n.id] = yPos;
    });
  });

  // RIGHT_SIDE: group by label, distribute vertically
  const rightLabelOrder = Object.keys(rightGroups);
  const rightLabelCount = rightLabelOrder.length;
  rightLabelOrder.forEach(function(label, labelIdx) {
    const group = rightGroups[label];
    const groupSlotH = rightLabelCount > 0 ? (H - MARGIN * 2) / rightLabelCount : H - MARGIN * 2;
    const slotStart = MARGIN + labelIdx * groupSlotH;
    const itemH = group.length > 1 ? groupSlotH / group.length : groupSlotH;
    group.forEach(function(n, i) {
      n.x = W - RIGHT_W / 2;
      n.y = slotStart + itemH * i + itemH / 2;
    });
  });

  // Step 2 — spring-force refinement (80 iterations, annealed)
  for (let iter = 0; iter < 80; iter++) {
    const step = 0.4 * Math.pow(0.97, iter);
    const force = {};
    for (const n of nodes) force[n.id] = {x: 0, y: 0};

    // Repulsion (all pairs)
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.max(Math.hypot(dx, dy), 1);
        const rep = Math.min(8000 / (dist * dist), 60);
        const ux = dx / dist, uy = dy / dist;
        force[a.id].x -= ux * rep;
        force[a.id].y -= uy * rep * 0.15;
        force[b.id].x += ux * rep;
        force[b.id].y += uy * rep * 0.15;
      }
    }

    // Attraction along edges
    const nodeMap = {};
    for (const n of nodes) nodeMap[n.id] = n;
    for (const e of edges) {
      const a = nodeMap[e.from], b = nodeMap[e.to];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.max(Math.hypot(dx, dy), 1);
      const att = (dist - 130) * 0.03;
      const ux = dx / dist, uy = dy / dist;
      force[a.id].x += ux * att;
      force[a.id].y += uy * att;
      force[b.id].x -= ux * att;
      force[b.id].y -= uy * att;
    }

    // Apply forces with layer-aware boundary clamping
    for (const n of nodes) {
      if (RIGHT_SIDE.has(n.label)) {
        n.x = Math.max(W - RIGHT_W - 10, Math.min(W - 30, n.x + force[n.id].x * step));
        n.y = Math.max(30, Math.min(H - 30, n.y + force[n.id].y * step));
      } else {
        const yBase = nodeBaseY[n.id] !== undefined ? nodeBaseY[n.id] : H / 2;
        n.x = Math.max(MARGIN + 20, Math.min(MARGIN + MAIN_W - 20, n.x + force[n.id].x * step));
        n.y = Math.max(yBase - 25, Math.min(yBase + 25, n.y + force[n.id].y * step));
      }
    }
  }
}

// ── Canvas setup ─────────────────────────────────────────────────────────
const gp = document.getElementById('gp');
const cv = document.getElementById('gc');
const ctx = cv.getContext('2d');
let dpr = window.devicePixelRatio || 1;
let lastNodeCount = 0;

function resize() {
  const rect = gp.getBoundingClientRect();
  const legendEl = gp.querySelector('.legend');
  const legendH = legendEl ? legendEl.offsetHeight : 0;
  const w = Math.max(rect.width, 10);
  const h = Math.max(rect.height - legendH, 10);
  cv.style.width = w + 'px';
  cv.style.height = h + 'px';
  cv.style.top = '0px';
  cv.width = w * dpr;
  cv.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (nodes.length > 0 && w > 50 && h > 50 && nodes.length !== lastNodeCount) {
    applyLayout(w, h);
    lastNodeCount = nodes.length;
  }
  draw();
}

// ── Edit-mode toolbar ─────────────────────────────────────────────────────
const gActions = document.getElementById('gActions');
if (EDIT_MODE) {
  gActions.innerHTML =
    '<button class="btn-sm" id="btnNode">+ Node</button>' +
    '<button class="btn-sm" id="btnEdge">+ Edge</button>' +
    '<button class="btn-sm primary" id="btnSave">Save JSON</button>';
  document.getElementById('btnNode').addEventListener('click', showCreateNode);
  document.getElementById('btnEdge').addEventListener('click', startEdgeMode);
  document.getElementById('btnSave').addEventListener('click', saveJSON);
}

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

function nodeAt(x, y) {
  for (const n of nodes) {
    if (RECT_LABELS.has(n.label)) {
      if (x >= n.x-RADIUS_RECT_W && x <= n.x+RADIUS_RECT_W && y >= n.y-RADIUS_RECT_H && y <= n.y+RADIUS_RECT_H) return n;
    }
  }
  for (const n of nodes) {
    if (!RECT_LABELS.has(n.label)) {
      if (Math.hypot(n.x - x, n.y - y) < RADIUS_CIRCLE) return n;
    }
  }
  return null;
}

function edgeAt(x, y) {
  const nmap = {};
  for (const n of nodes) nmap[n.id] = n;
  for (const e of edges) {
    const a = nmap[e.from], b = nmap[e.to];
    if (!a || !b) continue;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    if (Math.hypot(mx - x, my - y) < 16) return e;
  }
  return null;
}

function draw() {
  const W = cv.width / dpr, H = cv.height / dpr;
  ctx.clearRect(0, 0, W, H);
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
    ctx.strokeStyle = col;
    ctx.lineWidth = sel ? 2.2 : 1.4;
    if (!isFound) ctx.setLineDash([5, 3]);
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.max(Math.hypot(dx, dy), 1);
    const ux = dx / dist, uy = dy / dist;
    const startR = RECT_LABELS.has(a.label) ? RADIUS_RECT_H : RADIUS_CIRCLE;
    const endR   = RECT_LABELS.has(b.label) ? RADIUS_RECT_H : RADIUS_CIRCLE;
    const sx = a.x + ux * startR, sy = a.y + uy * startR;
    const ex = b.x - ux * (endR + ARROW_CLEARANCE), ey = b.y - uy * (endR + ARROW_CLEARANCE);
    // Bezier control points (lateral offset perpendicular to edge)
    const cx1 = sx + uy * CURVE, cy1 = sy - ux * CURVE;
    const cx2 = ex + uy * CURVE, cy2 = ey - ux * CURVE;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.bezierCurveTo(cx1, cy1, cx2, cy2, ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);
    // Arrow head using final bezier tangent
    const ang = Math.atan2(ey - cy2, ex - cx2);
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - 9 * Math.cos(ang - 0.4), ey - 9 * Math.sin(ang - 0.4));
    ctx.lineTo(ex - 9 * Math.cos(ang + 0.4), ey - 9 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    // Edge label at bezier midpoint offset laterally
    const mx = (sx + ex) / 2 + uy * CURVE * 0.5;
    const my = (sy + ey) / 2 - ux * CURVE * 0.5;
    ctx.font = '9px system-ui,sans-serif';
    ctx.fillStyle = sel ? '#D85A30' : (isFound ? '#0F6E56' : '#aaa');
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(e.predicate, mx, my - 7);
    ctx.restore();
  }

  // Draw nodes
  for (const n of nodes) {
    const sel = selected && selected.type === 'node' && selected.id === n.id;
    const efrom = edgeMode && edgeFrom === n.id;
    ctx.save();
    const isMissing = n.status === 'missing';
    const col = isMissing ? COLOR['missing'] : (COLOR[n.label] || '#aaa');
    const bgCol = isMissing ? '#F5F5F5' : (BADGE_BG[n.label] || '#eee');
    const isRect = RECT_LABELS.has(n.label);

    if (sel || efrom) {
      ctx.shadowColor = efrom ? '#378ADD' : '#D85A30';
      ctx.shadowBlur = 10;
    }

    ctx.fillStyle = bgCol;
    ctx.strokeStyle = sel ? '#D85A30' : col;
    ctx.lineWidth = sel ? 2.2 : 1.5;
    if (isMissing) ctx.setLineDash([4, 3]);

    if (isRect) {
      roundRect(ctx, n.x - RADIUS_RECT_W, n.y - RADIUS_RECT_H, RADIUS_RECT_W * 2, RADIUS_RECT_H * 2, 6);
    } else {
      ctx.beginPath(); ctx.arc(n.x, n.y, RADIUS_CIRCLE, 0, Math.PI * 2);
    }
    ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowBlur = 0;

    // Label text
    const textCol = isMissing ? '#aaa' : (BADGE_COLOR[n.label] || '#333');
    ctx.fillStyle = textCol;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = '9px system-ui,sans-serif';
    const shortLabel = n.label.length > 13 ? n.label.slice(0, 11) + '…' : n.label;
    ctx.fillText(shortLabel, n.x, n.y - 7);
    // Main prop snippet
    const rawProp = Object.values(n.props || {})[0];
    const mainProp = rawProp !== undefined && rawProp !== null ? String(rawProp) : (isMissing ? 'missing' : '');
    const shortProp = mainProp.slice(0, 13) + (mainProp.length > 13 ? '…' : '');
    ctx.font = '8px system-ui,sans-serif';
    ctx.fillStyle = isMissing ? '#ccc' : col;
    ctx.fillText(shortProp, n.x, n.y + 6);

    ctx.restore();
  }
}

// ── Interaction ───────────────────────────────────────────────────────────
cv.addEventListener('mousedown', function(e) {
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const n = nodeAt(mx, my);
  if (edgeMode) {
    if (n) {
      if (!edgeFrom) { edgeFrom = n.id; draw(); }
      else if (edgeFrom !== n.id) {
        showCreateEdge(edgeFrom, n.id);
        edgeFrom = null; edgeMode = false;
      }
    }
    return;
  }
  if (n) { drag = n; dragOff = {x: mx - n.x, y: my - n.y}; selectItem('node', n.id); return; }
  const ed = edgeAt(mx, my);
  if (ed) { selectItem('edge', ed.id); return; }
  clearSelection();
});
cv.addEventListener('mousemove', function(e) {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x = e.clientX - r.left - dragOff.x;
  drag.y = e.clientY - r.top - dragOff.y;
  draw();
});
cv.addEventListener('mouseup', function() { drag = null; });

function selectItem(type, id) {
  selected = {type, id};
  draw();
  if (type === 'node') showNodePanel(nodes.find(n => n.id === id));
  else showEdgePanel(edges.find(e => e.id === id));
}

function clearSelection() {
  selected = null;
  showEmpty();
  draw();
}

function showEmpty() {
  document.getElementById('dpEmpty').style.display = '';
  const dc = document.getElementById('dpContent');
  dc.style.display = 'none';
}

function renderDP(body, actions) {
  document.getElementById('dpEmpty').style.display = 'none';
  const dc = document.getElementById('dpContent');
  dc.style.display = 'flex';
  dc.style.flexDirection = 'column';
  document.getElementById('dpBody').innerHTML = body;
  const da = document.getElementById('dpActions');
  if (actions && EDIT_MODE) {
    da.style.display = 'flex';
    da.innerHTML = actions;
  } else {
    da.style.display = 'none';
    da.innerHTML = '';
  }
}

function showNodePanel(n) {
  if (!n) return;
  const bb = BADGE_BG[n.label] || '#eee';
  const bc = BADGE_COLOR[n.label] || '#333';
  const statusBg = n.status === 'found' ? '#E1F5EE' : '#f0f0f0';
  const statusFg = n.status === 'found' ? '#0F6E56' : '#888';
  const badge = '<span class="dp-label-badge" style="background:' + bb + ';color:' + bc + ';">' + n.label + '</span>';
  const stag = '<span style="font-size:10px;padding:2px 7px;border-radius:10px;background:' + statusBg + ';color:' + statusFg + ';">' + n.status + '</span>';
  let fields = '<div class="dp-field">' + badge + ' ' + stag + '</div>';

  if (EDIT_MODE) {
    fields += '<div class="dp-field"><label>Node ID</label><input id="dpId" value="' + esc(n.id) + '" /></div>';
    fields += '<div class="dp-field"><label>Class</label><select id="dpClass">' +
      NODE_CLASSES.map(function(c) { return '<option' + (c === n.label ? ' selected' : '') + '>' + c + '</option>'; }).join('') +
      '</select></div>';
    fields += '<div class="dp-field"><label>Status</label><select id="dpStatus">' +
      '<option' + (n.status === 'found' ? ' selected' : '') + '>found</option>' +
      '<option' + (n.status === 'missing' ? ' selected' : '') + '>missing</option>' +
      '</select></div>';
    const propStr = Object.keys(n.props || {}).length > 0 ? JSON.stringify(n.props, null, 1) : '';
    fields += '<div class="dp-field"><label>Properties (JSON)</label>' +
      '<textarea id="dpProps" placeholder="{&quot;content&quot;:&quot;...&quot;}">' + esc(propStr) + '</textarea></div>';
  } else {
    // Read-only
    fields += '<div class="dp-field"><label>Node ID</label><input value="' + esc(n.id) + '" readonly /></div>';
    if (Object.keys(n.props || {}).length > 0) {
      fields += '<div class="dp-field"><label>Properties</label>' +
        '<textarea readonly>' + esc(JSON.stringify(n.props, null, 1)) + '</textarea></div>';
    }
  }
  const evidTurns = (n.evidence || []).join(', ') || '—';
  fields += '<div class="dp-field"><label>Evidence turns</label><input value="' + esc(evidTurns) + '" readonly /></div>';

  renderDP(fields,
    '<button class="save" onclick="saveNode(&quot;' + esc(n.id) + '&quot;)">Save</button>' +
    '<button class="del" onclick="deleteNode(&quot;' + esc(n.id) + '&quot;)">Delete</button>'
  );
}

function showEdgePanel(e) {
  if (!e) return;
  const statusBg = e.status === 'found' ? '#E1F5EE' : '#f0f0f0';
  const statusFg = e.status === 'found' ? '#0F6E56' : '#888';
  const stag = '<span style="font-size:10px;padding:2px 7px;border-radius:10px;background:' + statusBg + ';color:' + statusFg + ';">' + e.status + '</span>';
  let fields = '<div class="dp-field"><span style="font-size:11px;font-weight:500;color:#222;">Edge</span> ' + stag + '</div>';

  if (EDIT_MODE) {
    fields += '<div class="dp-field"><label>Predicate</label><select id="dpPred">' +
      PREDICATES.map(function(p) { return '<option' + (p === e.predicate ? ' selected' : '') + '>' + p + '</option>'; }).join('') +
      '</select></div>';
    fields += '<div class="dp-field"><label>From</label><select id="dpFrom">' +
      nodes.map(function(n) { return '<option value="' + n.id + '"' + (n.id === e.from ? ' selected' : '') + '>' + n.label + ' (' + n.id + ')</option>'; }).join('') +
      '</select></div>';
    fields += '<div class="dp-field"><label>To</label><select id="dpTo">' +
      nodes.map(function(n) { return '<option value="' + n.id + '"' + (n.id === e.to ? ' selected' : '') + '>' + n.label + ' (' + n.id + ')</option>'; }).join('') +
      '</select></div>';
    fields += '<div class="dp-field"><label>Status</label><select id="dpEdgeStatus">' +
      '<option' + (e.status === 'found' ? ' selected' : '') + '>found</option>' +
      '<option' + (e.status === 'placeholder' ? ' selected' : '') + '>placeholder</option>' +
      '</select></div>';
  } else {
    fields += '<div class="dp-field"><label>Predicate</label><input value="' + esc(e.predicate) + '" readonly /></div>';
    fields += '<div class="dp-field"><label>From → To</label><input value="' + esc(e.from + ' → ' + e.to) + '" readonly /></div>';
  }
  const evid = (e.evidence || []).join(', ') || '—';
  fields += '<div class="dp-field"><label>Evidence turns</label><input value="' + esc(evid) + '" readonly /></div>';

  renderDP(fields,
    '<button class="save" onclick="saveEdge(&quot;' + esc(e.id) + '&quot;)">Save</button>' +
    '<button class="del" onclick="deleteEdge(&quot;' + esc(e.id) + '&quot;)">Delete</button>'
  );
}

function saveNode(id) {
  const n = nodes.find(function(x) { return x.id === id; });
  if (!n) return;
  n.label = document.getElementById('dpClass').value;
  n.status = document.getElementById('dpStatus').value;
  try { n.props = JSON.parse(document.getElementById('dpProps').value || '{}'); } catch(err) {}
  draw(); showNodePanel(n);
}
function saveEdge(id) {
  const e = edges.find(function(x) { return x.id === id; });
  if (!e) return;
  e.predicate = document.getElementById('dpPred').value;
  e.from = document.getElementById('dpFrom').value;
  e.to = document.getElementById('dpTo').value;
  e.status = document.getElementById('dpEdgeStatus').value;
  draw(); showEdgePanel(e);
}
function deleteNode(id) {
  nodes = nodes.filter(function(n) { return n.id !== id; });
  edges = edges.filter(function(e) { return e.from !== id && e.to !== id; });
  clearSelection(); draw();
}
function deleteEdge(id) {
  edges = edges.filter(function(e) { return e.id !== id; });
  clearSelection(); draw();
}

// ── Edge creation ─────────────────────────────────────────────────────────
let edgeModeTimeout = null;
function startEdgeMode() {
  edgeMode = true; edgeFrom = null;
  const t = document.getElementById('gTitle');
  t.innerHTML = 'Click source node, then target node…';
  if (edgeModeTimeout) clearTimeout(edgeModeTimeout);
  edgeModeTimeout = setTimeout(function() {
    edgeMode = false; edgeFrom = null;
    updateTitle();
  }, 10000);
}

function showCreateEdge(fromId, toId) {
  if (edgeModeTimeout) clearTimeout(edgeModeTimeout);
  updateTitle();
  const m = document.getElementById('createModal');
  const b = document.getElementById('modalBox');
  b.innerHTML =
    '<div class="modal-title">Add edge</div>' +
    '<div class="modal-field"><label>From</label><input value="' + esc(fromId) + '" readonly/></div>' +
    '<div class="modal-field"><label>To</label><input value="' + esc(toId) + '" readonly/></div>' +
    '<div class="modal-field"><label>Predicate</label><select id="mPred">' +
    PREDICATES.map(function(p) { return '<option>' + p + '</option>'; }).join('') +
    '</select></div>' +
    '<div class="modal-actions">' +
    '<button onclick="closeModal()">Cancel</button>' +
    '<button class="confirm" onclick="confirmEdge(&quot;' + esc(fromId) + '&quot;,&quot;' + esc(toId) + '&quot;)">Add</button>' +
    '</div>';
  m.style.display = 'flex';
}

function confirmEdge(from, to) {
  const pred = document.getElementById('mPred').value;
  const id = 'e' + Date.now();
  edges.push({id: id, from: from, to: to, predicate: pred, status: 'found', evidence: [], props: {}});
  closeModal(); updateTitle(); draw();
}

// ── Node creation ─────────────────────────────────────────────────────────
function showCreateNode() {
  const m = document.getElementById('createModal');
  const b = document.getElementById('modalBox');
  b.innerHTML =
    '<div class="modal-title">Add node</div>' +
    '<div class="modal-field"><label>Class</label><select id="mClass">' +
    NODE_CLASSES.map(function(c) { return '<option>' + c + '</option>'; }).join('') +
    '</select></div>' +
    '<div class="modal-field"><label>Main text / content</label>' +
    '<textarea id="mText" placeholder="e.g. I will never succeed"></textarea></div>' +
    '<div class="modal-actions">' +
    '<button onclick="closeModal()">Cancel</button>' +
    '<button class="confirm" onclick="confirmNode()">Add</button>' +
    '</div>';
  m.style.display = 'flex';
}

function confirmNode() {
  const cls = document.getElementById('mClass').value;
  const txt = document.getElementById('mText').value;
  const propKeys = {
    Problem: 'description', Goal: 'statement', CoreBelief: 'content',
    IntermediateBelief: 'content', Situation: 'description',
    AutomaticThought: 'content', Reaction: 'content',
    AdaptiveResponse: 'content', Intervention: 'description',
    Homework: 'taskDescription', Client: '', Session: ''
  };
  const propKey = propKeys[cls] || 'content';
  const id = cls.toLowerCase().slice(0, 3) + '_' + Date.now();
  const W = cv.width / dpr, H = cv.height / dpr;
  nodes.push({
    id: id, label: cls,
    x: 60 + Math.random() * (W - 120),
    y: 60 + Math.random() * (H - 120),
    status: 'found',
    props: propKey ? {[propKey]: txt} : {},
    evidence: []
  });
  closeModal(); updateTitle(); draw();
}

// ── Save JSON ─────────────────────────────────────────────────────────────
function saveJSON() {
  const data = JSON.stringify({nodes: nodes, edges: edges}, null, 2);
  const blob = new Blob([data], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cbt_graph.json'; a.click();
}

function closeModal() {
  document.getElementById('createModal').style.display = 'none';
}

function updateTitle() {
  const t = document.getElementById('gTitle');
  t.innerHTML = '<span class="live-dot"></span>Knowledge graph · ' +
    nodes.length + ' nodes · ' + edges.length + ' edges';
}

function esc(s) {
  if (s === undefined || s === null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Expose globals for inline onclick ─────────────────────────────────────
window.saveNode = saveNode; window.saveEdge = saveEdge;
window.deleteNode = deleteNode; window.deleteEdge = deleteEdge;
window.confirmEdge = confirmEdge; window.confirmNode = confirmNode;
window.closeModal = closeModal; window.clearSelection = clearSelection;

new ResizeObserver(resize).observe(gp);
resize();
updateTitle();
})();
</script>
</body>
</html>'''


def _render_canvas(
    canvas_nodes: list[dict],
    canvas_edges: list[dict],
    edit_mode: bool = False,
    height: int = 610,
) -> str:
    """Render the canvas graph as an iframe srcdoc string."""
    nodes_json = json.dumps(canvas_nodes)
    edges_json = json.dumps(canvas_edges)
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

def _session_bar_html(phase: str, technique: str, turn_count: int) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:8px;padding:8px 4px;">'
        f'<span style="font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;'
        f'background:#E6F1FB;color:#185FA5;">{html.escape(phase)}</span>'
        f'<span style="font-size:11px;padding:3px 10px;border-radius:20px;'
        f'border:0.5px solid #d1d5db;color:#666;">{html.escape(technique)}</span>'
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


def _bot_respond(message: str, history: list, session: Session):
    if session is None:
        session = _new_session()
    result = turn(session, message)
    history = history + [{"role": "assistant", "content": result["reply"]}]
    phase = result["phase"]
    technique = result["technique"]
    bar_html = _session_bar_html(phase, technique, session.turn_count)
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges())
    graph_html = _render_canvas(nodes, edges, edit_mode=False)
    return history, session, bar_html, graph_html


def _reset_therapy():
    session = _new_session()
    history = [{"role": "assistant", "content": INTRO}]
    bar_html = _session_bar_html("Rapport", "Rapport Building", 0)
    nodes, edges = _build_canvas_data(session.graph.nodes(), session.graph.edges())
    graph_html = _render_canvas(nodes, edges, edit_mode=False)
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
    return _render_canvas(cn, ce, edit_mode=True)


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

with gr.Blocks(title="CBT V4_flat — Therapy + Query", fill_height=True) as demo:
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
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Share what's on your mind…",
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    reset_btn = gr.Button("New session")

                # Right column: live graph
                with gr.Column(scale=3):
                    graph_panel = gr.HTML()

            therapy_outputs = [chatbot, session_state, session_bar, graph_panel]

            send_btn.click(
                _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
            ).then(
                _bot_respond, [pending_msg, chatbot, session_state], therapy_outputs
            )
            msg_box.submit(
                _add_user, [msg_box, chatbot], [chatbot, msg_box, pending_msg]
            ).then(
                _bot_respond, [pending_msg, chatbot, session_state], therapy_outputs
            )
            reset_btn.click(_reset_therapy, [], therapy_outputs)
            demo.load(_reset_therapy, [], therapy_outputs)

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

            load_outputs = [handle_state, summary_md, query_graph_panel, query_chat, question_box]

            live_btn.click(_load_live, [session_state], load_outputs)
            json_btn.click(_load_json, [json_file], load_outputs)
            neo_btn.click(_load_neo4j, [neo_uri, neo_user, neo_pw], load_outputs)

            ask_btn.click(
                _query_ask, [handle_state, question_box, query_chat],
                [query_chat, question_box],
            )
            question_box.submit(
                _query_ask, [handle_state, question_box, query_chat],
                [query_chat, question_box],
            )
