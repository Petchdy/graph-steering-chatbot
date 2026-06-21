# V4 → V4_flat migration

## Scope

`V4_flat/` is the flat-ontology sibling of `V4/`. The only ontology change is in 4 abstract families:

| Family               | V4 (subclass labels)                                          | V4_flat (single label + property)                              |
|----------------------|---------------------------------------------------------------|----------------------------------------------------------------|
| `Problem`            | 7 subclasses (`AcademicProblem` ... `OtherProblem`)           | `:Problem` + `properties.domain ∈ {academic, work, social, family, financial, health, other}` |
| `CoreBelief`         | 3 subclasses (`Self/World/OthersCoreBelief`)                  | `:CoreBelief` + `properties.domain ∈ {self, world, others}`; `category` only when `domain == "self"` |
| `IntermediateBelief` | 3 subclasses (`Attitude`, `Rule`, `Assumption`)               | `:IntermediateBelief` + `properties.subtype`                   |
| `Reaction`           | 3 subclasses (`Emotional/Behavioral/PhysiologicalReaction`)   | `:Reaction` + `properties.channel`; `valence` only when `channel == "emotional"` |

Everything else (Situation/AutomaticThought/Goal/Intervention/Homework/AdaptiveResponse, all edges, all signatures, all enums on data properties, Situation.kind / temporality design, distortionType enum, technique enum, Homework.taskType) is **identical** between v4 and v4_flat.

## Pipeline-level changes

| Stage | Behavioral change? | What changed |
|-------|--------------------|--------------|
| 1 — extract       | No  | Already emits abstract `label` + `group_key` guess. Only the import path changed. |
| 1.5 — validate    | No  | Still classifies into the 4 subclasses; result lives in `group_key`. |
| 2 — merge         | No  | Still partitions on `(label, group_key)` — no cross-domain merges. |
| 2.5 — properties  | **Small** | Added `assign_group_key_to_props()` which copies `group_key` into `props.domain | subtype | channel` before the LLM property-fill passes. |
| 3 — chains        | No  | Edge anchors and signatures were already abstract-label-keyed. |
| 4 — validate      | No  | Same. |
| 5 — persist       | **Main change** | JSON emits `label = n.label` (abstract), `parent = null`. Neo4j writes a single label (e.g. `:Problem:ABox`) and adds property indexes for the discriminator on each of the 4 families. |

## Where the discriminator lives

| Format | V4                                          | V4_flat                                                    |
|--------|---------------------------------------------|------------------------------------------------------------|
| JSON   | `{"label": "SelfCoreBelief", "parent": "CoreBelief"}` | `{"label": "CoreBelief", "parent": null, "properties": {"domain": "self"}}` |
| Neo4j  | `(:CoreBelief:SelfCoreBelief:ABox)`         | `(:CoreBelief:ABox {domain: "self"})`                       |

## Why property indexes

The flat shape replaces label scans like `(:SelfCoreBelief)` with property scans like `(:CoreBelief {domain: "self"})`. Without an index, the property scan walks the full `:CoreBelief` set. `cbt_stage5_persist_v4.py:write_neo4j()` and `import_json_to_neo4j.py` both run:

```cypher
CREATE INDEX IF NOT EXISTS FOR (n:`<family>`) ON (n.<prop>)
```

for `(Problem, domain)`, `(CoreBelief, domain)`, `(IntermediateBelief, subtype)`, `(Reaction, channel)`.

## Evaluation against the V4 gold standard

The V4 gold file uses leaf subclass labels (e.g. `SelfCoreBelief`). To score a v4_flat run against it, the comparator needs to treat `(label="CoreBelief", properties.domain="self")` as equivalent to `label="SelfCoreBelief"`. That adapter is **not yet written** — it's a follow-up if you want a direct numeric comparison.

## Files in this folder

- `cbt_ontology_v4_flat.py` — modified ontology constants. `CLASS_HIERARCHY` no longer contains leaf subclasses; `leaf_label()` / `parent_of()` are no-ops; `GROUP_KEY_PROP` map added.
- `cbt_stage1_extract_v4.py`, `cbt_stage1_5_validate.py`, `cbt_stage2_merge_v4.py`, `cbt_stage3_chains_v4.py`, `cbt_stage4_validate_v4.py` — copy of V4, import path renamed.
- `cbt_stage2_5_properties_v4.py` — adds `assign_group_key_to_props()` step.
- `cbt_stage5_persist_v4.py` — flat JSON emission + property index creation.
- `cbt_reporter_v4.py` — relabels "Group Key" column → "Discriminator" for clarity.
- `run_pipeline_v4_flat.py` — orchestrator; output filename suffix is `_KG_v4flat.json`.
- `import_json_to_neo4j.py` — imports the flat JSON, creates the same property indexes.
- `Modelfile` — unchanged.

## Smoke test

```
cd V4_flat
python -c "import cbt_ontology_v4_flat as o; print(o.ONTOLOGY_VERSION); print('AcademicProblem' not in o.CLASS_HIERARCHY); print(o.leaf_label('Problem','social')=='Problem'); print(o.GROUP_KEY_PROP['Reaction']=='channel')"
# expected: ontology_v4_flat  True  True  True

python run_pipeline_v4_flat.py ../transcripts/demo1_transcript_translated_gemini.json
# inspect the produced *_KG_v4flat.json — every node label is one of the 11 abstract
# families; every formerly-subclassed node carries its discriminator in properties.
```
