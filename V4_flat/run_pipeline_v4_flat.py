"""
CBT KG — V4_flat pipeline runner.

End-to-end orchestration. Same stages as V4; differs only at the output edge:
the four formerly-subclassed families (Problem / CoreBelief / IntermediateBelief /
Reaction) emit a single abstract label, with the discriminator promoted to a
property (domain / subtype / channel) by Stage 2.5.

Stages:
  Stage 1   all-classes per-turn extraction
  Stage 1.5 validate (drop fails) + subclass partition key (still set on group_key)
  Stage 2   partition + dedupe (still partitions on group_key — no cross-domain merges)
  Stage 2.5 properties + group_key→props promotion
  Stage 3   subject-anchored chain extraction + reinforces + deterministic structure
  Stage 4   4a validate -> 4a-repair -> 4b verify-all
  Stage 5   reusable JSON (versioned (N)) + optional Neo4j (flat labels + property indexes)

Usage:
  python run_pipeline_v4_flat.py <transcript.json> [out_dir] [--neo4j] [--workers 2] [--ctx 8192]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_ollama import ChatOllama, OllamaEmbeddings

from cbt_ontology_v4_flat import load_transcript
from cbt_stage1_extract_v4 import run_stage1
from cbt_stage1_1_session_extract import run_stage1_1, SESSION_LEVEL_CLASSES
from cbt_stage1_2_atomize import run_stage1_2
from cbt_stage1_5_validate import run_stage1_5
from cbt_stage2_merge_v4 import run_stage2, GATE, AUTO
from cbt_stage2_5_properties_v4 import run_stage2_5
from cbt_stage3_chains_v4 import run_stage3, _structure_edges  # noqa: F401
from cbt_stage4_validate_v4 import run_stage4
from cbt_stage5_persist_v4 import build_json, save_json, write_neo4j, CLIENT_ID, SESSION_ID
from cbt_reporter_v4 import generate_markdown_report

LLM_MODEL = "qwen3.5-nothink"
EMBED_MODEL = "qwen3-embedding:8b"
OLLAMA_TIMEOUT = 900  # 15-minute timeout per LLM call

class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, ctx_limit: int):
        self.ctx_limit = ctx_limit

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            gen = response.generations[0][0]
            if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                meta = gen.message.response_metadata
                p_eval = meta.get("prompt_eval_count", 0)
                eval_c = meta.get("eval_count", 0)
                total = p_eval + eval_c
                if total > 0:
                    warning = " ⚠️ [NEAR/EXCEEDED LIMIT!]" if total >= self.ctx_limit * 0.95 else ""
                    print(f"[LLM Token Usage] Prompt: {p_eval} | Completion: {eval_c} | Total: {total}/{self.ctx_limit}{warning}")
        except Exception:
            pass


def _stage_banner(stage: str, intent: str, inputs: dict[str, int | str] | None = None) -> float:
    """Print a 'about to run' banner before a stage. Returns t0 for timing.

    `inputs` is a per-class or per-quantity breakdown of what we are feeding the
    stage (e.g., raw node counts going into Stage 1.5). Empty rows are omitted."""
    print()
    print("─" * 70)
    print(f"▶ {stage}")
    print(f"  intent : {intent}")
    if inputs:
        rows = [(k, v) for k, v in inputs.items() if v not in (0, "0", None, "", [])]
        if rows:
            print(f"  input  :")
            width = max(len(str(k)) for k, _ in rows)
            for k, v in rows:
                print(f"    {str(k):<{width}}  {v}")
    print("─" * 70)
    return time.time()


def _stage_summary(stage: str, t0: float, **counts) -> None:
    """Print an 'after stage' summary. `counts` mixes scalars (printed as
    'key: value') and dicts (printed as a per-class breakdown table). Pass
    `_drop_empty=False` to keep zero rows in dict breakdowns (default drops them)."""
    elapsed = time.time() - t0
    drop_empty = counts.pop("_drop_empty", True)
    print(f"✓ {stage} done  ({elapsed:.1f}s)")
    for key, value in counts.items():
        if isinstance(value, dict):
            if not value:
                continue
            rows = list(value.items())
            if drop_empty:
                rows = [(k, v) for k, v in rows if v not in (0, None, "", [])]
            if not rows:
                continue
            print(f"  {key}:")
            width = max(len(str(k)) for k, _ in rows)
            for k, v in rows:
                print(f"    {str(k):<{width}}  {v}")
        else:
            print(f"  {key}: {value}")


def _count_by_class(by_label: dict[str, list]) -> dict[str, int]:
    return {lbl: len(ns) for lbl, ns in by_label.items()}


def _count_edges_by_predicate(edges: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in edges:
        out[e.predicate] = out.get(e.predicate, 0) + 1
    return dict(sorted(out.items()))


def _brief_report(transcript_path: str, turns, ctx_limit: int, max_workers: int):
    """Print a brief summary of the pipeline configuration before starting."""
    speakers = {}
    for t in turns:
        speakers[t.speaker] = speakers.get(t.speaker, 0) + 1
    avg_len = sum(len(t.text) for t in turns) / max(len(turns), 1)

    print("=" * 60)
    print("  CBT KG V4_flat Pipeline — Pre-run Report")
    print("=" * 60)
    print(f"  Transcript : {os.path.basename(transcript_path)}")
    print(f"  Turns       : {len(turns)}")
    for spk, cnt in sorted(speakers.items()):
        print(f"    {spk:>10} : {cnt} turns")
    print(f"  Avg length  : {avg_len:.0f} chars/turn")
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  Embed model : {EMBED_MODEL}")
    print(f"  Context     : {ctx_limit} tokens")
    print(f"  Timeout     : {OLLAMA_TIMEOUT}s ({OLLAMA_TIMEOUT // 60}min) per call")
    print(f"  Workers     : {max_workers}")
    print("=" * 60)
    print()


def run(transcript_path: str, out_dir: str = ".", use_neo4j: bool = False,
        session_type: str = "therapy", ctx_limit: int = 8192,
        max_workers: int = 1, limit: int = 0, merge_debug: bool = False) -> str:
    callbacks = [TokenUsageCallbackHandler(ctx_limit)]
    llm = ChatOllama(model=LLM_MODEL, temperature=0, request_timeout=OLLAMA_TIMEOUT,
                     callbacks=callbacks, num_ctx=ctx_limit, reasoning=False)
    embedder = OllamaEmbeddings(model=EMBED_MODEL)

    turns = load_transcript(transcript_path)
    if limit > 0:
        turns = turns[:limit]
    name = os.path.basename(transcript_path)

    _brief_report(transcript_path, turns, ctx_limit, max_workers)

    t_pipeline = time.time()

    # ─── Stage 1 ─────────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 1: extract entities (one LLM call per turn)",
        "Per-turn extraction across all classes with class definitions in-prompt.",
        {"turns": len(turns), "workers": max_workers})
    raw = run_stage1(turns, llm, max_workers=max_workers)
    _stage_summary("Stage 1", t,
                   total_raw=sum(len(v) for v in raw.values()),
                   per_class=_count_by_class(raw))

    # ─── Stage 1.1 ───────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 1.1: session-level extraction",
        f"Skim the whole transcript with adjacent-class priors and emit additional "
        f"Nodes for session-spanning classes ({', '.join(SESSION_LEVEL_CLASSES)}). "
        "Chunked if rendered transcript > CHAR_BUDGET. Additive — Stage 1 outputs preserved.",
        _count_by_class(raw))
    raw = run_stage1_1(raw, turns, llm)
    _stage_summary("Stage 1.1", t,
                   total_after=sum(len(v) for v in raw.values()),
                   per_class_after=_count_by_class(raw))

    # ─── Stage 1.2 ───────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 1.2: atomize + normalize",
        "Split multi-concept AutomaticThought / CoreBelief / IntermediateBelief into "
        "atomic propositions (≤ MAX_SPLITS=4). Normalize Intervention text into a "
        "one-sentence description. Pass-through everything else.",
        _count_by_class(raw))
    raw = run_stage1_2(raw, turns, llm)
    _stage_summary("Stage 1.2", t,
                   total_after=sum(len(v) for v in raw.values()),
                   per_class_after=_count_by_class(raw))

    # ─── Stage 1.5 ───────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 1.5: validate (keep/drop)",
        "Drop nodes whose text does not fit the assigned class definition. "
        "Discriminators (domain / subtype / channel) are now assigned in Stage 2.5.",
        _count_by_class(raw))
    kept, dropped = run_stage1_5(raw, turns, llm)
    _stage_summary("Stage 1.5", t,
                   dropped=len(dropped),
                   per_class_after=_count_by_class(kept))

    # ─── Stage 2 ─────────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 2: merge (class-only partition + candidate gate + LLM judge)",
        f"Partition by class label only. Embed, generate candidate pairs at cos >= {GATE:.2f}, "
        f"auto-confirm at cos >= {AUTO:.2f}, batched LLM 'same?' judge on the rest, "
        f"connected components fold into lowest-id survivor. {'(--merge-debug ON)' if merge_debug else ''}",
        _count_by_class(kept))
    merged = run_stage2(kept, embedder, llm, debug=merge_debug)
    _stage_summary("Stage 2", t,
                   merges=len(merged.audit),
                   retired=len(merged.retired_ids),
                   per_class_survivors=_count_by_class(merged.survivors))

    # ─── Stage 2.5 ───────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 2.5: assign properties (LLM batch + deterministic lexicons)",
        "First classify discriminators (Problem.domain, CoreBelief.domain, "
        "IntermediateBelief.subtype, Reaction.channel); then fill distortionType, "
        "modality, kind, technique, taskType, isOptional, category (self only), "
        "valence (emotional only), temporality.",
        _count_by_class(merged.survivors))
    run_stage2_5(merged.survivors, turns, llm)
    # snapshot how many of each property were populated
    prop_pop = {
        "AutomaticThought.distortionType": sum(1 for n in merged.survivors.get("AutomaticThought", []) if "distortionType" in n.props),
        "AutomaticThought.modality":       sum(1 for n in merged.survivors.get("AutomaticThought", []) if "modality" in n.props),
        "Situation.kind":                  sum(1 for n in merged.survivors.get("Situation", []) if "kind" in n.props),
        "Situation.temporality":           sum(1 for n in merged.survivors.get("Situation", []) if "temporality" in n.props),
        "Intervention.technique":          sum(1 for n in merged.survivors.get("Intervention", []) if "technique" in n.props),
        "Homework.taskType":               sum(1 for n in merged.survivors.get("Homework", []) if "taskType" in n.props),
        "Homework.isOptional":             sum(1 for n in merged.survivors.get("Homework", []) if "isOptional" in n.props),
        "CoreBelief.category (self only)": sum(1 for n in merged.survivors.get("CoreBelief", []) if "category" in n.props),
        "Reaction.valence (emo only)":     sum(1 for n in merged.survivors.get("Reaction", []) if "valence" in n.props),
        "Problem.domain":                  sum(1 for n in merged.survivors.get("Problem", []) if "domain" in n.props),
        "CoreBelief.domain":               sum(1 for n in merged.survivors.get("CoreBelief", []) if "domain" in n.props),
        "IntermediateBelief.subtype":      sum(1 for n in merged.survivors.get("IntermediateBelief", []) if "subtype" in n.props),
        "Reaction.channel":                sum(1 for n in merged.survivors.get("Reaction", []) if "channel" in n.props),
    }
    _stage_summary("Stage 2.5", t, properties_populated=prop_pop)

    # ─── Stage 3 ─────────────────────────────────────────────────────────────
    n_subjects = sum(len(merged.survivors.get(lbl, [])) for lbl in (
        "Situation", "AutomaticThought", "CoreBelief", "IntermediateBelief",
        "Reaction", "Problem", "Goal", "Homework", "Intervention"))
    t = _stage_banner(
        "Stage 3: extract edges (subject-anchored + reinforces + structural)",
        "Pass A: one LLM call per subject node (chain + hinge). "
        "Pass B: wide-window Reaction × CoreBelief reinforces. "
        "Pass C: deterministic Session→Problem/Intervention/Homework + hasSession.",
        {"anchored subjects": n_subjects})
    edges, stage3_audit = run_stage3(merged.survivors, turns, CLIENT_ID, SESSION_ID, llm)
    _stage_summary("Stage 3", t,
                   total_candidate_edges=len(edges),
                   per_predicate=_count_edges_by_predicate(edges))

    # ─── Stage 4 ─────────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 4: validate (4a) + repair + verify (4b)",
        "4a drops dangling/disjoint/signature-fail/evidence-outside edges. "
        "4a-repair rescues wrong-class endpoints when exactly one alternative exists. "
        "4b LLM verify-all on surviving LLM edges (structural skips 4b).",
        {"candidate edges": len(edges)})
    final_edges, vreport = run_stage4(edges, merged.survivors, turns, llm)
    _stage_summary("Stage 4", t,
                   kept=vreport.kept,
                   dropped_4a=len(vreport.dropped_4a),
                   repaired=len(vreport.repaired),
                   dropped_4b=len(vreport.dropped_4b),
                   final_per_predicate=_count_edges_by_predicate(final_edges))

    elapsed = time.time() - t_pipeline
    print()
    print("=" * 70)
    print(f"  Extraction complete — total {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)

    # ─── Stage 5 ─────────────────────────────────────────────────────────────
    t = _stage_banner(
        "Stage 5: persist (Markdown report + flat JSON + optional Neo4j)",
        "Write run report; build flat JSON (abstract label + discriminator in props); "
        f"{'wipe + write Neo4j with property indexes' if use_neo4j else 'skip Neo4j (--neo4j not set)'}.",
        {"surviving nodes": sum(len(v) for v in merged.survivors.values()),
         "final edges":     len(final_edges)})

    report_path = os.path.join(out_dir, name.replace(".json", "") + "_report.md")
    generate_markdown_report(turns, raw, merged, final_edges,
                             output_path=report_path,
                             stage3_audit=stage3_audit, vreport=vreport)

    graph = build_json(merged.survivors, final_edges, turns, name, session_type)
    base = os.path.join(out_dir, name.replace(".json", "") + "_KG_v4flat.json")
    path = save_json(graph, base)

    if use_neo4j:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j"),
                  os.environ.get("NEO4J_PASSWORD", "")))
        write_neo4j(merged.survivors, final_edges, turns, driver, session_type)
        driver.close()

    _stage_summary("Stage 5", t,
                   markdown_report=report_path,
                   json_output=path,
                   neo4j="wrote" if use_neo4j else "skipped")

    return path


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    
    ap = argparse.ArgumentParser(description="CBT KG — V4_flat pipeline runner.")
    ap.add_argument("transcript", help="Path to transcript JSON")
    ap.add_argument("out_dir", nargs="?", default=".", help="Output directory")
    ap.add_argument("--neo4j", action="store_true", help="Push results to Neo4j")
    ap.add_argument("--ctx", type=int, default=8192, help="LLM context window in tokens")
    ap.add_argument("--workers", type=int, default=1, help="Max workers for parallel extract")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of turns for testing")
    ap.add_argument("--merge-debug", action="store_true",
                    help="Print per-partition cosines and verdicts during Stage 2 merge")
    args = ap.parse_args()

    run(args.transcript, args.out_dir, args.neo4j, ctx_limit=args.ctx,
        max_workers=args.workers, limit=args.limit, merge_debug=args.merge_debug)
