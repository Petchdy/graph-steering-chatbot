"""
CBT KG — Stage 5 (V4_flat): persist.

Flat ontology: every content node carries the abstract family label only
(:Problem, :CoreBelief, ...). The subclass discriminator lives in a property
(domain / subtype / channel), assigned by Stage 2.5's LLM classifier.

Two outputs:
  1. Reusable JSON: meta / nodes[id,label,parent=null,properties,evidence] /
     edges[type,from,to,evidence]. Re-importable into Neo4j without rerunning
     the pipeline.
  2. Neo4j (optional, if a driver is provided): flat TBox (no leaf subclasses) +
     ABox nodes with single label + property index for the discriminator +
     Utterance provenance + edges.

File versioning: never overwrite. base `..._KG.json` -> `..._KG(1).json` ->
`..._KG(2).json` ... newest highest number.

Node id convention: <ID_PREFIX[family]>_<n>, matching the V4 gold (cb_1, at_3, ...).
"""

from __future__ import annotations

import json
import os
import sys

from cbt_ontology_v4_flat import (Turn, Node, Edge, ONTOLOGY_VERSION, leaf_label, parent_of,
                                  ID_PREFIX, TEXT_PROP, REL_TYPE, SUBCLASS_RULES, CLASS_HIERARCHY,
                                  GROUP_KEY_PROP)

CLIENT_ID = -1          # reserved internal ids for scaffold nodes
SESSION_ID = -2


# ---------------------------------------------------------------------------
# id mapping
# ---------------------------------------------------------------------------

def _build_id_map(survivors: dict[str, list[Node]]) -> dict[int, str]:
    """internal int id -> '<prefix>_<n>' (stable, per-prefix counter, sorted by id)."""
    counters: dict[str, int] = {}
    id_map: dict[int, str] = {CLIENT_ID: "client_1", SESSION_ID: "session_1"}
    # deterministic order: by family prefix then internal id
    for label in sorted(survivors):
        prefix = ID_PREFIX.get(label, label.lower())
        for n in sorted(survivors[label], key=lambda x: x.id):
            counters[prefix] = counters.get(prefix, 0) + 1
            id_map[n.id] = f"{prefix}_{counters[prefix]}"
    return id_map


# ---------------------------------------------------------------------------
# JSON build
# ---------------------------------------------------------------------------

def _node_properties(n: Node) -> dict:
    """Properties dict per family, matching the flat ontology shape.
    Strips `sourceText` (Stage 1.2 audit-only field) from every output."""
    if n.label == "Intervention":
        # Intervention: emit description + technique + optional techniqueLabel.
        # description comes from Stage 1.2 normalization (stored on n.text);
        # fall back to n.text if Stage 1.2 was skipped.
        out: dict = {"description": n.props.get("description") or n.text}
        if "technique" in n.props:
            out["technique"] = n.props["technique"]
        if n.props.get("techniqueLabel"):
            out["techniqueLabel"] = n.props["techniqueLabel"]
        return out
    text_key = TEXT_PROP.get(n.label, "content")
    out = {text_key: n.text}
    for k, v in n.props.items():
        if k == "sourceText":
            continue                            # audit-only — never exported
        out[k] = v
    return out


def build_json(survivors: dict[str, list[Node]], edges: list[Edge], turns: list[Turn],
               transcript_name: str, session_type: str = "therapy") -> dict:
    id_map = _build_id_map(survivors)

    nodes_json = [
        {"id": "client_1", "label": "Client", "parent": None, "properties": {}, "evidence": []},
        {"id": "session_1", "label": "Session", "parent": None,
         "properties": {"sessionType": session_type}, "evidence": []},
    ]
    for label in sorted(survivors):
        for n in sorted(survivors[label], key=lambda x: x.id):
            nodes_json.append({
                "id": id_map[n.id],
                "label": n.label,         # flat: always abstract family
                "parent": None,           # flat: no leaf/parent split
                "properties": _node_properties(n),
                "evidence": sorted(n.evidence),
            })

    edges_json = []
    for e in edges:
        if e.subject_id not in id_map or e.object_id not in id_map:
            continue
        d = {"type": e.predicate, "from": id_map[e.subject_id], "to": id_map[e.object_id],
             "evidence": sorted(e.evidence)}
        if e.properties.get("reportedIntensity"):
            d["reportedIntensity"] = e.properties["reportedIntensity"]
        if e.repaired:
            d["note"] = "repaired (4a)"
        elif e.reason:
            d["note"] = e.reason
        edges_json.append(d)

    counts: dict[str, int] = {}
    for nj in nodes_json:
        counts[nj["label"]] = counts.get(nj["label"], 0) + 1

    # Generate TBox for the JSON output
    tbox_nodes = []
    tbox_edges = []
    
    # Generate from CLASS_HIERARCHY
    for cls, parent in CLASS_HIERARCHY.items():
        tbox_nodes.append({"id": f"tbox_{cls}", "label": "TBox", "name": cls})
        if parent:
            tbox_edges.append({
                "type": "SUB_CLASS_OF", 
                "from": f"tbox_{cls}", 
                "to": f"tbox_{parent}"
            })

    return {
        "meta": {
            "schema_version": ONTOLOGY_VERSION,
            "transcript": transcript_name,
            "session_type": session_type,
            "n_turns": len(turns),
            "speaker_enum": ["therapist", "client"],
            "generated_by": "cbt v4 pipeline",
        },
        "tbox_nodes": tbox_nodes,
        "tbox_edges": tbox_edges,
        "nodes": nodes_json,
        "edges": edges_json,
        "summary_counts": {"nodes_total": len(nodes_json), "by_label": counts,
                           "edges_total": len(edges_json)},
    }


# ---------------------------------------------------------------------------
# versioned write
# ---------------------------------------------------------------------------

def versioned_path(base_path: str) -> str:
    """base.json -> base.json | base(1).json | base(2).json ... (next free)."""
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    i = 1
    while os.path.exists(f"{root}({i}){ext}"):
        i += 1
    return f"{root}({i}){ext}"


def save_json(graph: dict, base_path: str) -> str:
    path = versioned_path(base_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"[stage5] wrote {path}", file=sys.stderr)
    return path


# ---------------------------------------------------------------------------
# Neo4j (optional)
# ---------------------------------------------------------------------------

def write_neo4j(survivors: dict[str, list[Node]], edges: list[Edge], turns: list[Turn],
                driver, session_type: str = "therapy") -> None:
    """Wipe + write TBox (class hierarchy) and ABox (leaf+parent labels, Utterance
    provenance, edges). `driver` is a neo4j GraphDatabase driver."""
    id_map = _build_id_map(survivors)

    # TBox: every abstract family + its leaf subclasses, with SUB_CLASS_OF
    tbox_pairs = list(CLASS_HIERARCHY.items())

    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for cls, parent in tbox_pairs:
            s.run("MERGE (c:TBox {name:$n})", n=cls)
        for cls, parent in tbox_pairs:
            if parent:
                s.run("MATCH (a:TBox {name:$c}),(b:TBox {name:$p}) "
                      "MERGE (a)-[:SUB_CLASS_OF]->(b)", c=cls, p=parent)

        # Flat-ontology indexes: with the discriminator now a property, scans on
        # (:CoreBelief {domain:'self'}) need a property index to match the speed
        # of label scans like (:SelfCoreBelief) did in V4.
        for label, prop in GROUP_KEY_PROP.items():
            s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})")

        # scaffold
        s.run("MERGE (c:Client:ABox {id:'client_1'})")
        s.run("MERGE (se:Session:ABox {id:'session_1', sessionType:$t})", t=session_type)

        # Utterances
        for t in turns:
            s.run("MERGE (u:Utterance:ABox {id:$id}) "
                  "SET u.turnIndex=$ti, u.speaker=$sp, u.text=$tx",
                  id=f"utt_{t.turn_index}", ti=t.turn_index, sp=t.speaker, tx=t.text)
            s.run("MATCH (u:Utterance {id:$id}),(se:Session {id:'session_1'}) "
                  "MERGE (u)-[:IN_SESSION]->(se)", id=f"utt_{t.turn_index}")

        # ABox content nodes — flat: single abstract label, discriminator in props.
        for label, nodes in survivors.items():
            for n in nodes:
                primary = n.label                    # always abstract
                props = _node_props_for_neo(n)
                s.run(f"MERGE (x:`{primary}`:ABox {{id:$id}}) SET x += $props, "
                      f"x.primaryLabel=$primary",
                      id=id_map[n.id], props=props, primary=primary)
                s.run("MATCH (x:ABox {id:$id}),(c:TBox {name:$primary}) MERGE (x)-[:IS_A]->(c)",
                      id=id_map[n.id], primary=primary)
                for ti in sorted(n.evidence):
                    s.run("MATCH (x:ABox {id:$id}),(u:Utterance {id:$u}) "
                          "MERGE (x)-[:EVIDENCED_BY]->(u)",
                          id=id_map[n.id], u=f"utt_{ti}")

        # edges
        for e in edges:
            if e.subject_id not in id_map or e.object_id not in id_map:
                continue
            rel = REL_TYPE.get(e.predicate)
            if not rel:
                continue
            s.run(f"MATCH (a:ABox {{id:$s}}),(b:ABox {{id:$o}}) "
                  f"MERGE (a)-[r:`{rel}`]->(b) "
                  f"SET r.evidence=$ev" + (", r.reportedIntensity=$ri" if
                  e.properties.get("reportedIntensity") else ""),
                  s=id_map[e.subject_id], o=id_map[e.object_id],
                  ev=sorted(e.evidence), ri=e.properties.get("reportedIntensity"))
    print("[stage5] wrote Neo4j graph", file=sys.stderr)


def _node_props_for_neo(n: Node) -> dict:
    p = _node_properties(n)
    # Neo4j stores scalars; keep as-is (all our props are str/bool)
    return p


if __name__ == "__main__":
    print("Stage 5 is a library; import build_json/save_json/write_neo4j.")
