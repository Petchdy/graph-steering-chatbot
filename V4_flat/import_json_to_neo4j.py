import json
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from cbt_ontology_v4_flat import SUBCLASS_RULES, ID_PREFIX, GROUP_KEY_PROP

def main():
    # Load environment variables
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD")
    if not uri or not pw:
        sys.exit("NEO4J_URI and NEO4J_PASSWORD must be set in .env")

    # Load JSON Graph (v4_flat output)
    json_path = sys.argv[1] if len(sys.argv) > 1 else "demo1_transcript_KG_v4flat.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load Transcript for Utterance text
    transcript_path = "../transcripts/" + data["meta"]["transcript"]
    with open(transcript_path, "r", encoding="utf-8") as f:
        turns = json.load(f)

    from cbt_ontology_v4_flat import SUBCLASS_RULES, ID_PREFIX, CLASS_HIERARCHY

    driver = GraphDatabase.driver(uri, auth=(user, pw))

    # TBox mapping
    tbox_pairs = list(CLASS_HIERARCHY.items())

    print("Connecting to Neo4j to wipe and write...")
    with driver.session() as s:
        # Wipe DB
        s.run("MATCH (n) DETACH DELETE n")

        # Flat-ontology indexes (parity with Stage 5)
        for label, prop in GROUP_KEY_PROP.items():
            s.run(f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.{prop})")

        # TBox
        for cls, parent in tbox_pairs:
            s.run("MERGE (c:TBox {name:$n})", n=cls)
        for cls, parent in tbox_pairs:
            if parent:
                s.run("MATCH (a:TBox {name:$c}),(b:TBox {name:$p}) MERGE (a)-[:SUB_CLASS_OF]->(b)", c=cls, p=parent)

        # Scaffold
        s.run("MERGE (c:Client:ABox {id:'client_1'})")
        s.run("MERGE (se:Session:ABox {id:'session_1', sessionType:$t})", t=data["meta"].get("session_type", "therapy"))

        # Utterances
        for i, t in enumerate(turns):
            s.run("MERGE (u:Utterance:ABox {id:$id}) SET u.turnIndex=$ti, u.speaker=$sp, u.text=$tx",
                  id=f"utt_{i}", ti=i, sp=t.get("speaker", "unknown"), tx=t.get("text", ""))
            s.run("MATCH (u:Utterance {id:$id}),(se:Session {id:'session_1'}) MERGE (u)-[:IN_SESSION]->(se)", id=f"utt_{i}")

        # ABox Nodes
        for n in data["nodes"]:
            if n["id"] in ("client_1", "session_1"): 
                continue
            leaf = n["label"]
            props = n.get("properties", {})
            s.run(f"MERGE (x:`{leaf}`:ABox {{id:$id}}) SET x += $props, x.primaryLabel=$leaf",
                  id=n["id"], props=props, leaf=leaf)
            s.run("MATCH (x:ABox {id:$id}),(c:TBox {name:$leaf}) MERGE (x)-[:IS_A]->(c)",
                  id=n["id"], leaf=leaf)
            for ti in n.get("evidence", []):
                s.run("MATCH (x:ABox {id:$id}),(u:Utterance {id:$u}) MERGE (x)-[:EVIDENCED_BY]->(u)",
                      id=n["id"], u=f"utt_{ti}")

        # Edges
        for e in data["edges"]:
            rel = e["type"]
            props = {k: v for k, v in e.items() if k not in ("type", "from", "to", "evidence")}
            s.run(f"MATCH (a:ABox {{id:$from}}), (b:ABox {{id:$to}}) MERGE (a)-[r:`{rel}`]->(b) SET r += $props",
                  **{"from": e["from"], "to": e["to"], "props": props})
            if "evidence" in e and e["evidence"]:
                s.run(f"MATCH (a:ABox {{id:$from}})-[r:`{rel}`]->(b:ABox {{id:$to}}) "
                      f"SET r.evidenceTurns = $turns",
                      **{"from": e["from"], "to": e["to"], "turns": e["evidence"]})

    driver.close()
    print("Graph successfully pushed to Neo4j!")

if __name__ == "__main__":
    main()
