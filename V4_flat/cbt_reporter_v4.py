"""
CBT Knowledge Graph — Pipeline Report Generator (V4_flat).

Generates a concise Markdown report of the full pipeline execution:
  Stage 1: Extraction summary table + node list grouped by class
  Stage 2: Merge decisions table (merge-only rows)
  Stage 3: Edge table with reason column

V4_flat note: the "Discriminator" column shows the discriminator that V4 would have
emitted as a subclass label (domain / subtype / channel) — under v4_flat the
same value is what Stage 5 writes into properties.<name>.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from cbt_ontology_v4_flat import Turn, Node, Edge
from cbt_stage2_merge_v4 import MergeResult

def _escape_md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# Stage 1 — summary table + node list by class
# ---------------------------------------------------------------------------

def _section_stage1(raw_nodes: dict[str, list[Node]]) -> str:
    lines: list[str] = ["## Stage 1: Extracted Entities\n"]

    # Summary counts
    lines.append("| Class | Raw nodes |")
    lines.append("|-------|-----------|")
    total = 0
    for label, nodes in raw_nodes.items():
        if nodes:
            lines.append(f"| {label} | {len(nodes)} |")
            total += len(nodes)
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    # Node list grouped by class
    for label, nodes in raw_nodes.items():
        if not nodes:
            continue
        lines.append(f"### {label}\n")
        has_summary = any(n.props.get("summary") for n in nodes)
        if has_summary:
            lines.append("| ID | Text | Summary | Discriminator | Turns |")
            lines.append("|----|------|---------|-----------|-------|")
        else:
            lines.append("| ID | Text | Discriminator | Turns |")
            lines.append("|----|------|-----------|-------|")
        for n in nodes:
            turns = ", ".join(str(t) for t in sorted(n.evidence))
            if has_summary:
                summary_text = _escape_md(n.props.get("summary") or "—")
                lines.append(
                    f"| {n.id} | {_escape_md(n.text)} | {summary_text} | {n.group_key or '—'} | {turns} |"
                )
            else:
                lines.append(
                    f"| {n.id} | {_escape_md(n.text)} | {n.group_key or '—'} | {turns} |"
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2 — merge decisions
# ---------------------------------------------------------------------------

def _section_stage2(merge_result: MergeResult, raw_nodes: dict[str, list[Node]]) -> str:
    lines: list[str] = ["## Stage 2: Merge Decisions\n"]

    # NOTE: "raw" here is post-Stage-1.2 input to Stage 2. The diff between raw
    # and survivors is the SUM of: (a) Stage 1.5 drops (happen BEFORE merge), and
    # (b) actual Stage 2 merges. The counts below are accurate; the "merged" tag
    # is only used when audit covers the diff. Otherwise reported as "retired".
    for label, survivor_nodes in merge_result.survivors.items():
        raw_count = len(raw_nodes.get(label, []))
        survivor_count = len(survivor_nodes)
        retired_count = raw_count - survivor_count
        if retired_count > 0:
            audit_count = sum(1 for m in merge_result.audit if m.get('survivor_text')
                              and any(n.text == m.get('survivor_text')
                                      for n in survivor_nodes))
            merged_at_2 = sum(1 for m in merge_result.audit
                              if m.get('survivor') in {n.id for n in survivor_nodes})
            dropped_at_15 = retired_count - merged_at_2
            tail = f" ({merged_at_2} merged at Stage 2"
            if dropped_at_15 > 0:
                tail += f", {dropped_at_15} dropped at Stage 1.5"
            tail += ")"
            lines.append(f"**{label}**: {raw_count} raw → {survivor_count} survivors{tail}")
        else:
            lines.append(f"**{label}**: {raw_count} raw → {survivor_count} survivors")

    # Merge audit (folds only)
    if merge_result.audit:
        lines.append("")
        lines.append("### Merges (folds)")
        lines.append("| Survivor Node | Retired Node | Mechanism (Cos) |")
        lines.append("|---------------|--------------|-----------------|")
        for m in merge_result.audit:
            lines.append(
                f"| {_escape_md(m.get('survivor_text', str(m.get('survivor'))))} "
                f"| {_escape_md(m.get('retired_text', str(m.get('retired'))))} "
                f"| {_escape_md(m.get('via', ''))} ({m.get('cos', '')}) |"
            )
        lines.append("")
    else:
        lines.append("\n*(no merges performed)*\n")

    # All candidate-pair decisions (above GATE). Lets the reader see what was
    # judged "different" and the cosine landscape — important now that Stage 1.1
    # adds more potential duplicates.
    pair_decisions = getattr(merge_result, "pair_decisions", None) or []
    if pair_decisions:
        lines.append("### All candidate pairs (cos ≥ GATE)")
        lines.append("| Partition | Node A | Node B | Cos | Verdict |")
        lines.append("|-----------|--------|--------|-----|---------|")
        for d in sorted(pair_decisions, key=lambda r: (r.get("partition", ""), -r.get("cos", 0))):
            lines.append(
                f"| {_escape_md(d.get('partition', ''))} "
                f"| {_escape_md(d.get('a_text', ''))} "
                f"| {_escape_md(d.get('b_text', ''))} "
                f"| {d.get('cos', '')} "
                f"| {_escape_md(d.get('verdict', ''))} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 3 — per-anchor audit
# ---------------------------------------------------------------------------

def _section_stage3_audit(stage3_audit: list[dict] | None) -> str:
    lines: list[str] = ["## Stage 3 audit (per anchor)\n"]
    if not stage3_audit:
        lines.append("*(no audit recorded)*\n")
        return "\n".join(lines)
    lines.append("Per-subject record: candidates offered to the LLM vs edges it proposed. "
                 "A `0 / N` row means the anchor had N candidates but the LLM proposed nothing.\n")
    lines.append("| Subject | Predicate | Candidates | Proposed | Note |")
    lines.append("|---------|-----------|-----------:|---------:|------|")
    for row in stage3_audit:
        subj = f"[{row.get('subject_label')}] {_escape_md(row.get('subject_text', ''))[:60]}"
        preds = row.get("predicates", {})
        note = row.get("note", "")
        if not preds:
            lines.append(f"| {subj} | (none) | 0 | 0 | {_escape_md(note)} |")
            continue
        for pred, info in preds.items():
            if isinstance(info, dict):
                lines.append(f"| {subj} | **{pred}** | {info.get('candidates', 0)} | "
                             f"{info.get('proposed', 0)} | {_escape_md(note)} |")
            else:
                # parse-fail case: info is a count, no proposed
                lines.append(f"| {subj} | **{pred}** | {info} | 0 | {_escape_md(note)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 4 — drops
# ---------------------------------------------------------------------------

def _section_stage4(vreport) -> str:
    lines: list[str] = ["## Stage 4: validate + verify\n"]
    if vreport is None:
        lines.append("*(no Stage 4 report)*\n")
        return "\n".join(lines)
    lines.append(
        f"- **Kept:** {vreport.kept}\n"
        f"- **Dropped at 4a (deterministic):** {len(vreport.dropped_4a)}\n"
        f"- **Repaired at 4a:** {len(vreport.repaired)}\n"
        f"- **Dropped at 4b (LLM verify):** {len(vreport.dropped_4b)}\n"
    )

    def _row(d):
        e = d.get("edge", {})
        return (f"| {_escape_md(e.get('predicate', ''))} "
                f"| {_escape_md(e.get('subject', ''))} "
                f"| {_escape_md(e.get('object', ''))} "
                f"| {_escape_md(d.get('rule') or d.get('reason') or '')} |")

    if vreport.dropped_4a:
        lines.append("### Dropped at 4a")
        lines.append("| Predicate | Subject | Object | Rule |")
        lines.append("|-----------|---------|--------|------|")
        for d in vreport.dropped_4a:
            lines.append(_row(d))
        lines.append("")

    if vreport.repaired:
        lines.append("### Repaired at 4a")
        lines.append("| From | To |")
        lines.append("|------|----|")
        for d in vreport.repaired:
            f = d.get("from", {}); t = d.get("to", {})
            lines.append(
                f"| {_escape_md(f.get('predicate'))} {_escape_md(f.get('subject', ''))} → {_escape_md(f.get('object', ''))} "
                f"| {_escape_md(t.get('predicate'))} {_escape_md(t.get('subject', ''))} → {_escape_md(t.get('object', ''))} |"
            )
        lines.append("")

    if vreport.dropped_4b:
        lines.append("### Dropped at 4b")
        lines.append("| Predicate | Subject | Object | Reason |")
        lines.append("|-----------|---------|--------|--------|")
        for d in vreport.dropped_4b:
            lines.append(_row(d))
        lines.append("")

    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Stage 3 — edge table
# ---------------------------------------------------------------------------

def _section_stage3(edges: list[Edge], all_nodes: dict[str, list[Node]]) -> str:
    lines: list[str] = ["## Stage 3 + Repair: Extracted Edges\n"]

    if not edges:
        lines.append("*(no edges extracted)*\n")
        return "\n".join(lines)

    node_by_id: dict[int | str, Node] = {n.id: n for ns in all_nodes.values() for n in ns}

    regular = [e for e in edges if not e.reason.startswith("[repair]")]
    repaired = [e for e in edges if e.reason.startswith("[repair]")]

    lines.append(f"**{len(regular)} extracted** + **{len(repaired)} repaired** = {len(edges)} total\n")
    lines.append("| Subject | Predicate | Object | Evidence | Reason |")
    lines.append("|---------|-----------|--------|----------|--------|")

    for e in edges:
        subj = node_by_id.get(e.subject_id)
        obj_ = node_by_id.get(e.object_id)
        subj_text = _escape_md(subj.text) if subj else f"id={e.subject_id}"
        obj_text  = _escape_md(obj_.text)  if obj_  else f"id={e.object_id}"
        ev = ", ".join(str(t) for t in sorted(e.evidence)) if e.evidence else "—"
        reason = _escape_md(e.reason or "—")
        lines.append(f"| {subj_text} | **{e.predicate}** | {obj_text} | {ev} | {reason} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failures — Stage 1 silent failures
# ---------------------------------------------------------------------------

def _section_failures(
    turns: list[Turn],
    raw_nodes: dict[str, list[Node]],
) -> str:
    lines: list[str] = ["## Extraction Failures\n"]
    any_failure = False

    # Stage 1: turns where zero entities were extracted
    covered_turns: set[int] = set()
    for nodes in raw_nodes.values():
        for n in nodes:
            covered_turns |= n.evidence

    silent = [t for t in turns if t.turn_index not in covered_turns]
    if silent:
        any_failure = True
        lines.append(f"### Stage 1 — Silent turns ({len(silent)} turns produced no entities)\n")
        lines.append("| Turn | Speaker | Text |")
        lines.append("|------|---------|------|")
        for t in silent:
            snippet = _escape_md(t.text[:100] + ("…" if len(t.text) > 100 else ""))
            lines.append(f"| {t.turn_index} | {t.speaker} | {snippet} |")
        lines.append("")

    if not any_failure:
        lines.append("*(no failures — all turns produced at least one entity)*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def generate_markdown_report(
    turns: list[Turn],
    raw_nodes: dict[str, list[Node]],
    merge_result: MergeResult,
    edges: list[Edge],
    output_path: str | Path = "pipeline_report.md",
    stage3_audit: list[dict] | None = None,
    vreport=None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_raw = sum(len(ns) for ns in raw_nodes.values())
    total_survivors = sum(len(nodes) for nodes in merge_result.survivors.values())

    header = (
        f"# CBT Pipeline Report (V4_flat)\n\n"
        f"**Generated:** {timestamp}  \n"
        f"**Turns:** {len(turns)}  |  "
        f"**Raw nodes:** {total_raw}  |  "
        f"**After merge:** {total_survivors}  |  "
        f"**Edges:** {len(edges)}\n\n---\n"
    )

    merged_nodes = merge_result.survivors

    sections = [
        header,
        _section_stage1(raw_nodes),
        "---\n",
        _section_stage2(merge_result, raw_nodes),
        "---\n",
        _section_stage3(edges, merged_nodes),
        "---\n",
        _section_stage3_audit(stage3_audit),
        "---\n",
        _section_stage4(vreport),
        "---\n",
        _section_failures(turns, raw_nodes),
    ]
    report = "\n".join(sections)

    output_path.write_text(report, encoding="utf-8")
    return output_path
