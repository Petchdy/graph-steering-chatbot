"""CBT intake schema. Drop a real clinical ontology in here without touching anything else."""

from interfaces import OntologyField

_CLINICAL_FIELDS = [
    OntologyField(key="presenting_problem", description="The main issue or concern the client describes.", priority=1),
    OntologyField(key="emotion",            description="The client's current or described emotional state.", priority=2),
    OntologyField(key="negative_thought",   description="A specific negative or irrational thought the client expresses.", priority=3),
    OntologyField(key="cognitive_pattern",  description="The cognitive distortion pattern (e.g. catastrophizing, labeling, overgeneralization).", priority=4),
    OntologyField(key="trigger_situation",  description="The external situation or event that triggered the distress.", priority=5),
    OntologyField(key="physical_symptoms",  description="Any physical symptoms mentioned (tension, sleep issues, fatigue, etc.).", priority=6),
    OntologyField(key="past_coping",        description="What the client has tried before to cope with the problem.", priority=7),
    OntologyField(key="reframe_attempt",    description="Any reframing or alternative perspective the client generates themselves.", priority=8),
    OntologyField(key="coping_strategies",  description="Strategies discussed or agreed upon during the session.", priority=9),
]

_SESSION_STATE_FIELDS = [
    OntologyField(key="session_phase",    description="Current session phase: Rapport / Exploration / Technique / Consolidation.", priority=0),
    OntologyField(key="active_technique", description="The CBT technique currently being applied.", priority=0),
]


class CBTSchema:
    """9 clinical fields (extracted from client speech) + 2 session-state fields
    (set by LLM JSON output, never surfaced as 'missing' to ask about)."""

    def fields(self) -> list[OntologyField]:
        return list(_SESSION_STATE_FIELDS + _CLINICAL_FIELDS)

    def render(self) -> str:
        """Renders only clinical fields — session-state fields are not for extraction."""
        return "\n".join(f"- {f.key}: {f.description}" for f in _CLINICAL_FIELDS)

    def render_ontology(self) -> str:
        """Renders rich CBT node classes with property schemas, for node extraction.
        Skips structural nodes (Session, Client, Utterance) — those are not extracted
        from client speech."""
        skip = {"Session", "Client", "Utterance"}
        lines = []
        for cls in CBT_NODE_CLASSES:
            label = cls["label"]
            if label in skip:
                continue
            props = cls.get("props", {})
            prop_str = "; ".join(f"{k}={v}" for k, v in props.items()) if props else "(no props)"
            lines.append(f"- {label}: {prop_str}")
        return "\n".join(lines)

    def node_classes(self) -> list[dict]:
        return CBT_NODE_CLASSES

    def edge_map(self) -> list[tuple[str, str, str]]:
        return CBT_EDGE_MAP

    def subject_edges(self) -> dict[str, list[tuple[str, str]]]:
        return CBT_SUBJECT_EDGES


# Ontology node classes — mirrors cbt_kg_ontology_v4_flat.txt.
# "multi" = True means multiple instances of this class can exist in one session.
# "props" documents the property schema from the ontology file (not enforced by
# the graph itself — extraction layer is responsible for populating valid values).
_CONTENT_NODE_LABELS = [
    "Problem", "Goal", "Intervention", "Homework",
    "CoreBelief", "IntermediateBelief", "Situation",
    "AutomaticThought", "Reaction", "AdaptiveResponse",
]

CBT_NODE_CLASSES: list[dict] = [
    # ── Provenance ──────────────────────────────────────────────────────────
    {"label": "Utterance", "multi": True, "props": {
        "text": "string",
        "speaker": "enum: therapist | client",
        "turnIndex": "integer (unique within Session)",
        "timestamp": "string (optional)",
    }},
    # ── Client anchor ───────────────────────────────────────────────────────
    {"label": "Client", "multi": False, "props": {}},
    # ── SessionStructure ────────────────────────────────────────────────────
    {"label": "Session", "multi": False, "props": {
        "sessionNumber": "integer (optional)",
        "sessionType":   "enum: evaluation | therapy",
        "date":          "string (optional)",
        "duration":      "string (optional)",
    }},
    {"label": "Problem", "multi": True, "props": {
        "description": "string",
        "domain": "enum: academic | work | social | family | financial | health | other",
    }},
    {"label": "Goal", "multi": True, "props": {
        "statement": "string",
    }},
    {"label": "Intervention", "multi": True, "props": {
        "description":     "string",
        "technique":       "enum: CACTUS 12 (efficiencyEvaluation | pieChartTechnique | ...)",
        "techniqueLabel":  "string (optional; when technique=other)",
    }},
    {"label": "Homework", "multi": True, "props": {
        "taskDescription": "string",
        "taskType":        "enum: thoughtRecord | behavioralExperiment | activityScheduling | "
                           "copingCard | skillsPractice | reading | other",
        "isOptional":      "boolean",
    }},
    # ── CognitiveModel ──────────────────────────────────────────────────────
    {"label": "CoreBelief", "multi": True, "props": {
        "content":  "string",
        "domain":   "enum: self | world | others",
        "category": "enum: helpless | unlovable | worthless (only when domain=self)",
    }},
    {"label": "IntermediateBelief", "multi": True, "props": {
        "content": "string",
        "subtype": "enum: attitude | rule | assumption",
    }},
    {"label": "Situation", "multi": True, "props": {
        "description": "string",
        "kind":        "enum: externalSituation | thoughtStream | image | "
                       "emotion | behavior | physiological",
        "temporality": "enum: past | present | anticipated (optional)",
    }},
    {"label": "AutomaticThought", "multi": True, "props": {
        "content":        "string",
        "modality":       "enum: verbal | image",
        "distortionType": "enum: PatternReframe 10 (allOrNothing | catastrophizing | ...)",
    }},
    {"label": "Reaction", "multi": True, "props": {
        "content": "string",
        "channel": "enum: emotional | behavioral | physiological",
        "valence": "enum: positive | negative (only when channel=emotional)",
    }},
    {"label": "AdaptiveResponse", "multi": True, "props": {
        "content": "string",
    }},
]

# Edge map — (subject_label, predicate, object_label). Mirrors v4_flat ontology.
CBT_EDGE_MAP: list[tuple[str, str, str]] = [
    # ── COGNITIVE CHAIN ─────────────────────────────────────────────────────
    ("CoreBelief",         "givesRiseTo",             "IntermediateBelief"),
    ("IntermediateBelief", "influencesPerceptionOf",  "Situation"),
    ("Situation",          "triggers",                "AutomaticThought"),
    ("AutomaticThought",   "leadsTo",                 "Reaction"),  # edge prop: reportedIntensity
    ("AutomaticThought",   "stemsFrom",               "CoreBelief"),
    ("Reaction",           "reinforces",              "CoreBelief"),
    ("Reaction",           "becomesSituation",        "Situation"),
    ("AutomaticThought",   "hasAdaptiveResponse",     "AdaptiveResponse"),
    # ── STRUCTURE — client anchor ───────────────────────────────────────────
    ("Client",             "hasSession",              "Session"),
    # ── STRUCTURE — session → content ───────────────────────────────────────
    ("Session",            "hasProblem",              "Problem"),
    ("Session",            "hasIntervention",         "Intervention"),
    ("Session",            "hasHomework",             "Homework"),
    # ── STRUCTURE — problem / goal ──────────────────────────────────────────
    ("Problem",            "manifestsAs",             "Situation"),
    ("Goal",               "targetsProblem",          "Problem"),
    ("Homework",           "targets",                 "Problem"),
    ("Homework",           "targets",                 "AutomaticThought"),
    ("Homework",           "targets",                 "IntermediateBelief"),
    ("Homework",           "targets",                 "CoreBelief"),
    # ── CROSS-LAYER HINGE ───────────────────────────────────────────────────
    ("AutomaticThought",   "associatedWith",          "Problem"),
    ("Intervention",       "appliedTo",               "AutomaticThought"),
    ("Intervention",       "appliedTo",               "IntermediateBelief"),
    ("Intervention",       "appliedTo",               "CoreBelief"),
    ("Intervention",       "appliedTo",               "Problem"),
    ("Intervention",       "produces",                "AdaptiveResponse"),
    # ── PROVENANCE ──────────────────────────────────────────────────────────
    *[(label, "evidencedBy", "Utterance") for label in _CONTENT_NODE_LABELS],
    ("Utterance",          "inSession",               "Session"),
]

# Per-class: which predicates this class can emit (subject side).
CBT_SUBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in CBT_EDGE_MAP:
    CBT_SUBJECT_EDGES.setdefault(_subj, []).append((_pred, _obj))

# Per-class: which predicates can point AT this class (object side).
CBT_OBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in CBT_EDGE_MAP:
    CBT_OBJECT_EDGES.setdefault(_obj, []).append((_pred, _subj))
