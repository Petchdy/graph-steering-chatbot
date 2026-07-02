"""
CBT Knowledge Graph — V4_flat ontology (single source of truth).

Ported verbatim from V4_flat/cbt_ontology_v4_flat.py. Every CBT concept used
anywhere in this codebase — node classes, properties, property enums, edge
predicates, class definitions — comes from this file. Anything not in here is
not part of the model.

Flat variant: Problem / CoreBelief / IntermediateBelief / Reaction stay as
single classes; the former subclass discriminator is promoted to a property
enum (domain / subtype / channel) carried in `props`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ONTOLOGY_VERSION = "ontology_v4_flat"

# ===========================================================================
# Runtime dataclasses (used by extract pipeline + Stage 5 export shape)
# ===========================================================================

SPEAKER_NORMALISE = {"client": "client", "patient": "client", "therapist": "therapist"}


@dataclass
class Turn:
    turn_index: int
    speaker: str             # "therapist" | "client"
    text: str


@dataclass
class Node:
    id: int
    label: str               # ABSTRACT/extract label
    text: str
    group_key: str | None = None
    evidence: set[int] = field(default_factory=set)
    context: dict[int, str] = field(default_factory=dict)
    props: dict = field(default_factory=dict)


@dataclass
class Edge:
    predicate: str
    subject_id: int
    object_id: int
    evidence: set[int] = field(default_factory=set)
    properties: dict = field(default_factory=dict)
    reason: str = ""
    repaired: bool = False


# ===========================================================================
# Stage 1 extraction contract
# ===========================================================================

EXTRACT_CLASSES: list[str] = [
    "Problem", "Goal", "Intervention", "Homework",
    "CoreBelief", "IntermediateBelief", "Situation",
    "AutomaticThought", "Reaction", "AdaptiveResponse",
]
# Client / Session / Utterance are scaffold, not extracted.

CLASS_DEFINITIONS: dict[str, str] = {
    "Problem": ("A GENERAL, ongoing area of difficulty — the kind of thing that "
                "would be a heading on a session agenda (\"trouble making friends\", "
                "\"procrastination\"). A recurring theme, NOT a single moment, NOT a "
                "mood. Test: would it appear as a session heading? → Problem."),
    "Goal": ("A desired outcome — what the client wants instead of the problem. "
             "Forward-facing and positive."),
    "Intervention": ("A therapeutic technique the therapist applies. Spans many "
                     "turns. Guided discovery / Socratic questioning is the general "
                     "style — do NOT extract it as a technique."),
    "Homework": ("A between-session task the therapist assigns the client to do "
                 "before the next session."),
    "CoreBelief": (
        "An ABSOLUTE, global belief about the self, the world, or other people, stated "
        "with no condition — e.g. 'I am worthless', 'I am unlovable', 'people cannot be "
        "trusted'. It is NOT a rule and NOT a condition. If the text contains "
        "'I must / I should / I have to' or 'if ... then ...', it is an IntermediateBelief, "
        "not a CoreBelief. If it is tied to one specific moment, it is an AutomaticThought."
    ),
    "IntermediateBelief": (
        "A conditional or instrumental belief sitting between core beliefs and automatic "
        "thoughts: a RULE ('I must always do my best'), an ASSUMPTION ('if I ask for help, "
        "people think I'm incompetent'), or an ATTITUDE ('it's terrible to fail'). It is "
        "marked by 'must / should / have to' or 'if ... then ...'. It is NOT an absolute "
        "identity claim like 'I am worthless' (that is a CoreBelief)."
    ),
    "Situation": ("A SINGLE, concrete, time-bound moment that triggered a specific "
                  "automatic thought. The first link of a Situation→AutomaticThought→"
                  "Reaction chain. Test: can you attach ONE specific automatic "
                  "thought to this exact moment? → Situation."),
    "AutomaticThought": (
        "A spontaneous, situation-specific thought tied to ONE moment — e.g. 'he didn't "
        "text back, he's angry at me'. It is NOT a general rule ('I must ...' = "
        "IntermediateBelief) and NOT an absolute identity claim ('I am worthless' = "
        "CoreBelief). Keep emotions OUT of the content (a feeling is a Reaction)."
    ),
    "Reaction": ("The client's response to an automatic thought — a feeling, an "
                 "action/avoidance, or a body sensation."),
    "AdaptiveResponse": ("A BALANCED, realistic response developed to counter an "
                         "automatic thought (the product of reframing). Balanced, "
                         "NOT merely positive. Only when genuinely reached — a "
                         "therapist suggestion the client does not internalise is not one."),
}

SPEAKER_PRIOR = {
    "client":    ["Situation", "AutomaticThought", "Reaction", "CoreBelief",
                  "IntermediateBelief", "Problem", "Goal", "AdaptiveResponse"],
    "therapist": ["Intervention", "Homework", "Goal", "AdaptiveResponse"],
}

CLASS_HIERARCHY: dict[str, str | None] = {
    "OWL_Thing": None,
    "Provenance": "OWL_Thing",
    "Utterance": "Provenance",
    "SessionStructure": "OWL_Thing",
    "Session": "SessionStructure",
    "Client": "OWL_Thing",
    "Problem": "SessionStructure",
    "Goal": "SessionStructure",
    "Intervention": "SessionStructure",
    "Homework": "SessionStructure",
    "CognitiveModel": "OWL_Thing",
    "CoreBelief": "CognitiveModel",
    "IntermediateBelief": "CognitiveModel",
    "Situation": "CognitiveModel",
    "AutomaticThought": "CognitiveModel",
    "Reaction": "CognitiveModel",
    "AdaptiveResponse": "CognitiveModel",
}

# ===========================================================================
# All node classes (13) — used to materialize placeholders + UI rendering.
# ===========================================================================

NODE_CLASSES: list[dict] = [
    {"label": "Utterance", "multi": True, "props": {
        "text": "string",
        "speaker": "enum: therapist | client",
        "turnIndex": "integer",
        "timestamp": "string (optional)",
    }},
    {"label": "Client", "multi": False, "props": {}},
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
        "technique":       "enum: CACTUS-12",
        "techniqueLabel":  "string (optional, when technique=other)",
    }},
    {"label": "Homework", "multi": True, "props": {
        "taskDescription": "string",
        "taskType":        "enum: thoughtRecord | behavioralExperiment | activityScheduling | "
                           "copingCard | skillsPractice | reading | other",
        "isOptional":      "boolean",
    }},
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
        "distortionType": "enum: PatternReframe-10",
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

CONTENT_LABELS = [c["label"] for c in NODE_CLASSES
                  if c["label"] not in ("Client", "Session", "Utterance")]

# ===========================================================================
# Subclass partition keys (kept for compatibility / Stage 1.5 prompts)
# ===========================================================================

SUBCLASS_RULES: dict[str, dict[str, str]] = {
    "Problem": {"academic": "AcademicProblem", "work": "WorkProblem",
                "social": "SocialProblem", "family": "FamilyProblem",
                "financial": "FinancialProblem", "health": "HealthProblem",
                "other": "OtherProblem"},
    "CoreBelief": {"self": "SelfCoreBelief", "world": "WorldCoreBelief",
                   "others": "OthersCoreBelief"},
    "IntermediateBelief": {"attitude": "Attitude", "rule": "Rule",
                           "assumption": "Assumption"},
    "Reaction": {"emotional": "EmotionalReaction", "behavioral": "BehavioralReaction",
                 "physiological": "PhysiologicalReaction"},
}
SUBCLASSED = frozenset(SUBCLASS_RULES.keys())

SUBCLASS_GLOSS: dict[str, dict[str, str]] = {
    "Problem": {
        "academic": "school, study, exams, grades",
        "work": "job, career, workplace",
        "social": "friends, peers, dating, social situations",
        "family": "parents, siblings, partner, children",
        "financial": "money, debt, finances",
        "health": "physical or mental health (incl. depression/anxiety)",
        "other": "cross-domain, unclear, or fits no other category",
    },
    "CoreBelief": {
        "self": "a belief about oneself (incl. lovability/worth, even if it mentions "
                "relationships — \"no one could love me\" is about the SELF)",
        "world": "a belief about the world (\"the world is dangerous\")",
        "others": "a belief about other people in general (\"people are cruel\")",
    },
    "IntermediateBelief": {
        "attitude": "an evaluation (\"It's terrible to fail\", \"Asking for help is weak\")",
        "rule": "a demand or standard (\"I must always do my best\")",
        "assumption": "a conditional if-then (\"If I ask for help, people think I'm incompetent\")",
    },
    "Reaction": {
        "emotional": "a feeling (anxious, ashamed, relieved, sad, lonely)",
        "behavioral": "an action or avoidance (stayed in bed, avoided the call, cried)",
        "physiological": "a body sensation (heart racing, chest tight, couldn't breathe)",
    },
}

GROUP_KEY_PROP: dict[str, str] = {
    "Problem":            "domain",
    "CoreBelief":         "domain",
    "IntermediateBelief": "subtype",
    "Reaction":           "channel",
}

PROBLEM_DOMAINS     = ("academic", "work", "social", "family", "financial", "health", "other")
CORE_BELIEF_DOMAINS = ("self", "world", "others")
IB_SUBTYPES         = ("attitude", "rule", "assumption")
REACTION_CHANNELS   = ("emotional", "behavioral", "physiological")


# ===========================================================================
# Stage 2.5 property enums + glosses
# ===========================================================================

SITUATION_KINDS: dict[str, str] = {
    "externalSituation": "a real external event/circumstance (incl. a recalled past event)",
    "thoughtStream": "a vague worry-spiral; the extracted thought is its conclusion",
    "image": "a mental picture that arose unbidden",
    "emotion": "an emotion that itself triggered the thought (first link in the chain)",
    "behavior": "an action that triggered the thought (first link in the chain)",
    "physiological": "a body sensation that triggered the thought (first link in the chain)",
}

DISTORTION_TYPES: dict[str, str] = {
    "allOrNothing": "black-and-white, nothing in between (\"I'm a total failure\")",
    "catastrophizing": "assuming the worst outcome (\"my life is ruined\")",
    "discountingPositive": "positives don't count (\"I only passed because it was easy\")",
    "fortuneTelling": "predicting a negative outcome (\"I know I'll fail\")",
    "labeling": "judging character from one action (\"I'm stupid\")",
    "mentalFiltering": "only the negatives matter",
    "mindReading": "assuming others' negative thoughts (\"she thinks I'm boring\")",
    "overgeneralization": "broad conclusion from one event (\"I always mess up\")",
    "personalization": "excessive self-blame (\"the team failed because of me\")",
    "shouldStatements": "rigid demands (\"I should be able to handle this\")",
    "none": "the thought is realistic / not distorted",
}

TECHNIQUES: dict[str, str] = {
    "efficiencyEvaluation": "is this thought actually useful in real life?",
    "pieChartTechnique": "break down contributing factors visually (excessive self-blame)",
    "alternativePerspective": "how would someone else see this situation?",
    "decatastrophizing": "examine the real probability of the feared outcome",
    "prosAndConsAnalysis": "list advantages and disadvantages of a thought",
    "evidenceBasedQuestioning": "what evidence supports / goes against this thought?",
    "realityTesting": "does this thought actually match reality?",
    "continuumTechnique": "place a judgment on a 0-100 scale, not all-or-nothing",
    "changingRulesToWishes": "turn \"I must/should\" into \"I'd prefer/wish\"",
    "behaviorExperiment": "plan a real-world test of a belief or prediction",
    "problemSolvingSkillsTraining": "teach step-by-step problem-solving",
    "systematicExposure": "graded, repeated approach to a feared situation",
    "other": "anything outside the 12 (record the name in techniqueLabel)",
}

SELF_CB_CATEGORIES: dict[str, str] = {
    "helpless": "incapable, ineffective, powerless, a failure",
    "unlovable": "unwanted, rejected, unworthy of relationships",
    "worthless": "bad, immoral, deserving of bad things",
}

HOMEWORK_TASKTYPES: dict[str, str] = {
    "thoughtRecord": "record situations/thoughts/feelings (e.g. a thought diary)",
    "behavioralExperiment": "test a prediction or belief through a real-world action",
    "activityScheduling": "schedule and carry out planned activities",
    "copingCard": "make/use a card with a coping statement",
    "skillsPractice": "practise a specific skill (relaxation, assertiveness, etc.)",
    "reading": "read assigned material (bibliotherapy)",
    "other": "any task outside the categories above",
}

# ===========================================================================
# Edge registry
# ===========================================================================

ANCHOR_FAMILIES: dict[str, list[tuple[str, str, str]]] = {
    "Situation": [
        ("triggers", "AutomaticThought",
         "this situation sparked that automatic thought"),
        ("triggers", "Reaction",
         "this situation produced that reaction DIRECTLY with no thought in between — "
         "use ONLY for reflexive/pre-cognitive responses (startle, immediate disgust, grief); "
         "if a thought came first, the reaction belongs to the thought via leadsTo instead"),
    ],
    "AutomaticThought": [
        ("leadsTo", "Reaction",
         "the reaction that followed this thought (add reportedIntensity ONLY if the "
         "client states a strength like \"very anxious\"/\"8/10\")"),
        ("stemsFrom", "CoreBelief",
         "this thought reflects/derives from that core belief (downward-arrow / ladder) — "
         "evaluate INDEPENDENTLY per thought; thoughts from the same Situation can stem from different CoreBeliefs"),
        ("stemsFrom", "IntermediateBelief",
         "use when the thought's content is driven by the rule/demand itself "
         "(\"I might slip up and break my rule\") rather than an identity claim "
         "(\"I am worthless\") — the rule-strength is the proximate driver"),
        ("associatedWith", "Problem",
         "FALLBACK ONLY: use when no Situation can be identified to route this thought — "
         "attempt the full Situation→AutomaticThought chain first"),
        ("hasAdaptiveResponse", "AdaptiveResponse",
         "the balanced response that answers this thought"),
    ],
    "CoreBelief": [
        ("givesRiseTo", "IntermediateBelief",
         "this core belief underlies that rule/attitude/assumption"),
    ],
    "IntermediateBelief": [
        ("influencesPerceptionOf", "Situation",
         "this belief shapes how the client perceived that situation"),
    ],
    "Reaction": [
        ("becomesSituation", "Situation",
         "this reaction itself became the trigger for a new thought (cascade) — only "
         "if the transcript clearly shows it"),
        ("leadsTo", "Reaction",
         "this reaction led to that next reaction in the SAME episode, as an explicit "
         "ordered sequence (e.g. scared → goosebumps → fled) — only when the transcript "
         "shows the order; if the reactions are independent responses to the same thought, "
         "leave them parallel (do not link)"),
    ],
    "Problem": [
        ("manifestsAs", "Situation",
         "this general problem shows up as that specific moment"),
    ],
    "Goal": [
        ("targetsProblem", "Problem",
         "this goal addresses that problem (Goal → Problem direction)"),
    ],
    "Homework": [
        ("targets", "Problem", "the homework works on this problem"),
        ("targets", "AutomaticThought", "the homework (e.g. thought record) works on this thought"),
        ("targets", "IntermediateBelief", "the homework (e.g. experiment) tests this rule/assumption"),
        ("targets", "CoreBelief", "the homework works directly on this core belief"),
    ],
    "Intervention": [
        ("appliedTo", "AutomaticThought", "this technique examined that thought"),
        ("appliedTo", "IntermediateBelief", "this technique examined that rule/assumption"),
        ("appliedTo", "CoreBelief", "this technique directly challenged that core belief"),
        ("appliedTo", "Problem", "this technique worked on that problem"),
        ("produces", "AdaptiveResponse",
         "the transcript shows this technique GENERATING that balanced response"),
    ],
}

REINFORCES = ("reinforces", "Reaction", "CoreBelief")
# Stage-3 wide-window LLM pass only. Never extracted per-turn (Tier A).
# Absence of a reinforces edge is clinically informative; do not fabricate.

ALLOWED_SIGNATURES: frozenset[tuple[str, str, str]] = frozenset(
    [(p, subj, o) for subj, fams in ANCHOR_FAMILIES.items() for (p, o, _h) in fams]
    + [REINFORCES]
)

PREDICATE_OBJECTS: dict[str, set[str]] = {}
for _p, _s, _o in ALLOWED_SIGNATURES:
    PREDICATE_OBJECTS.setdefault(_p, set()).add(_o)

DETERMINISTIC_PREDICATES = frozenset({
    "hasSession", "hasProblem", "hasIntervention", "hasHomework",
    "evidencedBy", "inSession",
})

DISJOINT_RULES: list[tuple[str, str, str]] = [
    ("AutomaticThought", "*", "AutomaticThought"),
    ("CoreBelief", "influencesPerceptionOf", "Situation"),
    ("Intervention", "appliedTo", "Goal"),
]

# Full edge map = ANCHOR_FAMILIES + REINFORCES + structure + provenance + targets
# Used to build placeholder edges and to drive the GraphReader / Cytoscape.
def _build_edge_map() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for subj, fams in ANCHOR_FAMILIES.items():
        for (pred, obj, _h) in fams:
            out.append((subj, pred, obj))
    # REINFORCES is (predicate, subject_label, object_label) — V4_flat order.
    _pred, _subj, _obj = REINFORCES
    out.append((_subj, _pred, _obj))
    # Structure
    out.extend([
        ("Client",   "hasSession",      "Session"),
        ("Session",  "hasProblem",      "Problem"),
        ("Session",  "hasIntervention", "Intervention"),
        ("Session",  "hasHomework",     "Homework"),
    ])
    # Provenance
    for label in CONTENT_LABELS:
        out.append((label, "evidencedBy", "Utterance"))
    out.append(("Utterance", "inSession", "Session"))
    return out


EDGE_MAP: list[tuple[str, str, str]] = _build_edge_map()

SUBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in EDGE_MAP:
    SUBJECT_EDGES.setdefault(_subj, []).append((_pred, _obj))

OBJECT_EDGES: dict[str, list[tuple[str, str]]] = {}
for _subj, _pred, _obj in EDGE_MAP:
    OBJECT_EDGES.setdefault(_obj, []).append((_pred, _subj))


REL_TYPE: dict[str, str] = {
    "hasSession": "HAS_SESSION", "hasProblem": "HAS_PROBLEM",
    "hasIntervention": "HAS_INTERVENTION", "hasHomework": "HAS_HOMEWORK",
    "manifestsAs": "MANIFESTS_AS", "triggers": "TRIGGERS", "leadsTo": "LEADS_TO",
    "stemsFrom": "STEMS_FROM", "givesRiseTo": "GIVES_RISE_TO",
    "influencesPerceptionOf": "INFLUENCES_PERCEPTION_OF",
    "becomesSituation": "BECOMES_SITUATION", "reinforces": "REINFORCES",
    "hasAdaptiveResponse": "HAS_ADAPTIVE_RESPONSE", "associatedWith": "ASSOCIATED_WITH",
    "targetsProblem": "TARGETS_PROBLEM", "targets": "TARGETS",
    "appliedTo": "APPLIED_TO", "produces": "PRODUCES",
    "evidencedBy": "EVIDENCED_BY", "inSession": "IN_SESSION",
}

# Reverse of REL_TYPE (Neo4j reader uses it).
PREDICATE_FROM_REL: dict[str, str] = {v: k for k, v in REL_TYPE.items()}


# ===========================================================================
# JSON export maps
# ===========================================================================

ID_PREFIX: dict[str, str] = {
    "Client": "client", "Session": "session", "Problem": "prob", "Goal": "goal",
    "Intervention": "intv", "Homework": "hw", "CoreBelief": "cb",
    "IntermediateBelief": "ib", "Situation": "sit", "AutomaticThought": "at",
    "Reaction": "react", "AdaptiveResponse": "adapt", "Utterance": "utt",
}

TEXT_PROP: dict[str, str] = {
    "Problem": "description", "Goal": "statement", "Intervention": "description",
    "Homework": "taskDescription", "CoreBelief": "content",
    "IntermediateBelief": "content", "Situation": "description",
    "AutomaticThought": "content", "Reaction": "content", "AdaptiveResponse": "content",
}


# ===========================================================================
# Deterministic lexicons (English + Thai — used by Stage 2.5 deterministic passes)
# ===========================================================================

THAI_TIME_MARKERS: dict[str, str] = {
    "เมื่อวาน": "past", "เมื่อกี้": "past", "ที่แล้ว": "past", "ก่อนหน้านี้": "past",
    "ตอนเด็ก": "past", "สมัยก่อน": "past", "เคย": "past", "ที่ผ่านมา": "past",
    "ตอนนี้": "present", "ขณะนี้": "present", "ช่วงนี้": "present", "ทุกวันนี้": "present",
    "พรุ่งนี้": "anticipated", "เดือนหน้า": "anticipated", "ปีหน้า": "anticipated",
    "กำลังจะ": "anticipated", "ในอนาคต": "anticipated", "ต่อไป": "anticipated",
}
THAI_EMOTION_NEG = ["เศร้า", "เหงา", "เสียใจ", "กลัว", "กังวล", "เครียด", "โกรธ",
                    "ผิดหวัง", "อาย", "ท้อ", "หดหู่", "สิ้นหวัง", "เจ็บปวด", "ว้าวุ่น"]
THAI_EMOTION_POS = ["ดีใจ", "สบายใจ", "โล่งใจ", "มีความสุข", "ภูมิใจ", "อุ่นใจ",
                    "มั่นใจ", "ผ่อนคลาย"]
EN_TIME_MARKERS: dict[str, str] = {
    "yesterday": "past", "last week": "past", "as a child": "past", "used to": "past",
    "right now": "present", "these days": "present", "currently": "present",
    "tomorrow": "anticipated", "next week": "anticipated", "next month": "anticipated",
    "going to": "anticipated", "in the future": "anticipated",
}
EN_EMOTION_NEG = ["sad", "lonely", "anxious", "afraid", "scared", "worried", "angry",
                  "ashamed", "hopeless", "depressed", "devastated", "heartbroken", "guilty"]
EN_EMOTION_POS = ["happy", "relieved", "proud", "calm", "confident", "glad", "content"]


def temporality_from_text(text: str) -> str | None:
    low = text.lower()
    for marker, value in THAI_TIME_MARKERS.items():
        if marker in text:
            return value
    for marker, value in EN_TIME_MARKERS.items():
        if marker in low:
            return value
    return None


def emotion_valence_from_text(text: str) -> str | None:
    low = text.lower()
    if any(w in text for w in THAI_EMOTION_NEG) or any(w in low for w in EN_EMOTION_NEG):
        return "negative"
    if any(w in text for w in THAI_EMOTION_POS) or any(w in low for w in EN_EMOTION_POS):
        return "positive"
    return None


def apply_gating_constraints(label: str, props: dict) -> dict:
    """Enforce §3.3 property-population constraints in place and return props.

    Rules:
    1. CoreBelief.category → null unless domain=="self"
    2. Reaction.valence → null unless channel=="emotional"
    3. Intervention.techniqueLabel → absent unless technique=="other"
    4. AutomaticThought.modality defaults to "verbal"
    5. Homework.isOptional defaults to False
    (Situation.temporality gating is handled by temporality_from_text;
     AutomaticThought.distortionType may be "none" — no code enforcement needed.)
    """
    if label == "CoreBelief":
        if props.get("domain") != "self":
            props.pop("category", None)
    elif label == "Reaction":
        if props.get("channel") != "emotional":
            props.pop("valence", None)
    elif label == "Intervention":
        if props.get("technique") != "other":
            props.pop("techniqueLabel", None)
    elif label == "AutomaticThought":
        props.setdefault("modality", "verbal")
    elif label == "Homework":
        props.setdefault("isOptional", False)
    return props


# ===========================================================================
# Schema adapter — implements the Schema Protocol from interfaces.py
# ===========================================================================

class CBTSchema:
    """Schema adapter exposing the V4_flat ontology to the rest of the codebase."""

    def node_classes(self) -> list[dict]:
        return NODE_CLASSES

    def edge_map(self) -> list[tuple[str, str, str]]:
        return EDGE_MAP

    def subject_edges(self) -> dict[str, list[tuple[str, str]]]:
        return SUBJECT_EDGES

    def anchor_families(self) -> dict[str, list[tuple[str, str, str]]]:
        return ANCHOR_FAMILIES

    def class_definitions(self) -> dict[str, str]:
        return CLASS_DEFINITIONS

    def render_ontology(self) -> str:
        """Compact rendering for prompts (skip scaffolding classes)."""
        skip = {"Session", "Client", "Utterance"}
        lines = []
        for cls in NODE_CLASSES:
            if cls["label"] in skip:
                continue
            props = cls.get("props", {})
            prop_str = "; ".join(f"{k}={v}" for k, v in props.items()) if props else "(no props)"
            lines.append(f"- {cls['label']}: {prop_str}")
        return "\n".join(lines)
