"""
CBT Knowledge Graph — V4_flat contract + shared types (single source of truth).

Flat variant of V4: Problem / CoreBelief / IntermediateBelief / Reaction stay
as single classes; the former subclass discriminator is promoted to a property
enum (domain / subtype / channel) carried in `Node.props`.

Same as V4: all edges, signatures, Situation.kind / temporality, distortionType,
technique enum, Homework.taskType, AdaptiveResponse, Goal targetsProblem Problem,
all deferred items.

Flat vs subclass implementation notes:
  - JSON `label` is always the abstract family (no SelfCoreBelief / Attitude / ...).
  - JSON `parent` is None for content nodes.
  - Stage 2 still partitions on `group_key` to prevent cross-domain merges.
  - Stage 2.5 copies group_key into `props.domain | subtype | channel` so the flat
    output shape emerges automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

ONTOLOGY_VERSION = "ontology_v4_flat"

# ===========================================================================
# Runtime dataclasses (shared by all stages)
# ===========================================================================

SPEAKER_NORMALISE = {"client": "client", "patient": "client", "therapist": "therapist"}


@dataclass
class Turn:
    turn_index: int          # 0-based position in the transcript array
    speaker: str             # "therapist" | "client"
    text: str


@dataclass
class Node:
    id: int
    label: str               # ABSTRACT/extract label at Stage 1; stays abstract.
    text: str                # client/therapist words (content/description/etc.)
    group_key: str | None = None      # partition key (subclass for subclassed families)
    evidence: set[int] = field(default_factory=set)   # supporting turnIndex values
    context: dict[int, str] = field(default_factory=dict)  # {turn_index: window blob}
    props: dict = field(default_factory=dict)         # distortionType, kind, valence, ...


@dataclass
class Edge:
    predicate: str
    subject_id: int
    object_id: int
    evidence: set[int] = field(default_factory=set)
    properties: dict = field(default_factory=dict)    # e.g. {"reportedIntensity": "8/10"}
    reason: str = ""
    repaired: bool = False                            # set by Stage 4a-repair


def load_transcript(path: str) -> list[Turn]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    turns: list[Turn] = []
    for i, item in enumerate(raw):
        spk = SPEAKER_NORMALISE.get(item["speaker"].strip().lower(), item["speaker"])
        turns.append(Turn(turn_index=i, speaker=spk, text=item["text"].strip()))
    return turns


def turn_texts(turns: list[Turn], indices: set[int] | list[int],
               window: int = 1) -> list[tuple[int, str, str]]:
    """Deduped, ordered (idx, speaker_tag, text) for the given turns ± window."""
    by_idx = {t.turn_index: t for t in turns}
    want: set[int] = set()
    for ti in indices:
        for j in range(ti - window, ti + window + 1):
            if j in by_idx:
                want.add(j)
    out = []
    for ti in sorted(want):
        t = by_idx[ti]
        out.append((ti, "T" if t.speaker == "therapist" else "C", t.text))
    return out


def render_turns(rows: list[tuple[int, str, str]]) -> str:
    return "\n".join(f"turn {i} | {tag}: {txt}" for i, tag, txt in rows) or "(none)"


# ===========================================================================
# Stage 1 extraction contract
# ===========================================================================
# Stage 1 extracts label + text (+ optional group_key guess). Stage 1.5 fixes
# the group_key (partition key). Properties are filled in Stage 2.5.

EXTRACT_CLASSES: list[str] = [
    "Problem", "Goal", "Intervention", "Homework",
    "CoreBelief", "IntermediateBelief", "Situation",
    "AutomaticThought", "Reaction", "AdaptiveResponse",
]
# (Client / Session / Utterance are scaffold, not extracted. Diagnosis removed.)

# Per-class extraction definition (lifted from cbt_kg_extraction_descriptions_V4_flat.md)
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
    "Situation": ("A single concrete moment that triggered a specific automatic thought. "
                  "Emit \"text\" = the MINIMAL trigger in one short phrase (what happened / "
                  "the stimulus). If the client elaborates (where they were, who they were "
                  "with, the circumstances, the sequence of events), ALSO emit \"context\" = "
                  "that elaboration. If there is no elaboration, omit \"context\". "
                  "Do NOT put the elaboration in \"text\" — keep \"text\" short."),
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

# Speaker prior (soft): which classes each speaker typically produces.
SPEAKER_PRIOR = {
    "client":    ["Situation", "AutomaticThought", "Reaction", "CoreBelief",
                  "IntermediateBelief", "Problem", "Goal", "AdaptiveResponse"],
    "therapist": ["Intervention", "Homework", "Goal", "AdaptiveResponse"],
}

# Full Ontology Hierarchy (TBox) — flat: no subclass leaves
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
# Subclass partition keys (set in Stage 1.5)  —  family -> {group_key: leaf}
# ===========================================================================

SUBCLASS_RULES: dict[str, dict[str, str]] = {
    "Problem": {"academic": "AcademicProblem", "work": "WorkProblem",
                "social": "SocialProblem", "family": "FamilyProblem",
                "financial": "FinancialProblem", "health": "HealthProblem",
                "personal": "PersonalProblem", "other": "OtherProblem"},
    "CoreBelief": {"self": "SelfCoreBelief", "world": "WorldCoreBelief",
                   "others": "OthersCoreBelief"},
    "IntermediateBelief": {"attitude": "Attitude", "rule": "Rule",
                           "assumption": "Assumption"},
    "Reaction": {"emotional": "EmotionalReaction", "behavioral": "BehavioralReaction",
                 "physiological": "PhysiologicalReaction"},
}
SUBCLASSED = frozenset(SUBCLASS_RULES.keys())

# Glosses for the Stage 1.5 subclass pick-one prompts.
SUBCLASS_GLOSS: dict[str, dict[str, str]] = {
    "Problem": {
        "academic": "school, study, exams, grades",
        "work": "job, career, workplace",
        "social": "friends, peers, dating, social situations",
        "family": "parents, siblings, partner, children",
        "financial": "money, debt, finances",
        "health": "physical or mental health (incl. depression/anxiety)",
        "personal": ("self-concept, identity, meaning, purpose, existential concerns "
                     "(the client's relationship with themselves, not the world/others)"),
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


def leaf_label(label: str, group_key: str | None) -> str:
    return label  # flat: no subclass expansion


def parent_of(label: str, group_key: str | None) -> str | None:
    return None  # flat: discriminator lives in props, not the label/parent fields


# Stage 2.5 uses this map to copy `Node.group_key` into `Node.props[<prop_name>]`,
# so Stage 5 emits flat JSON like {"label": "CoreBelief", "properties": {"domain": "self"}}.
GROUP_KEY_PROP: dict[str, str] = {
    "Problem":            "domain",
    "CoreBelief":         "domain",
    "IntermediateBelief": "subtype",
    "Reaction":           "channel",
}

# Property-value enums (mirror the SUBCLASS_RULES vocabularies).
PROBLEM_DOMAINS     = ("academic", "work", "social", "family", "financial", "health", "personal", "other")
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
    "emotion": ("an emotion that triggered the thought — first link in the chain, OR the "
                "re-entry point of a cascade (a reaction that became a new trigger)"),
    "behavior": ("an action that triggered the thought — first link in the chain, OR a "
                 "cascade re-entry point"),
    "physiological": ("a body sensation that triggered the thought — first link in the "
                      "chain, OR a cascade re-entry point"),
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
# Subject-anchored families: for each subject node, the relations it can emit,
# with (predicate, object_label, hint). Stage 3 issues ONE call per subject node
# covering all its families, so the local chain is judged coherently.

ANCHOR_FAMILIES: dict[str, list[tuple[str, str, str]]] = {
    "Situation": [
        ("triggers", "AutomaticThought",
         "this situation sparked that automatic thought"),
        ("triggers", "Reaction",
         "this situation produced that reaction DIRECTLY, with no thought in between — "
         "use ONLY for a reflexive/pre-cognitive response (startle, immediate disgust or "
         "grief). If a thought came first, do NOT use this; the reaction belongs to the "
         "thought via leadsTo."),
    ],
    "AutomaticThought": [
        ("leadsTo", "Reaction",
         "the reaction that followed this thought (add reportedIntensity ONLY if the "
         "client states a strength like \"very anxious\"/\"8/10\")"),
        ("stemsFrom", "CoreBelief",
         "this thought reflects/derives from that core belief (downward-arrow / ladder)"),
        ("associatedWith", "Problem",
         "FALLBACK: the problem this thought is tied to — use when no Situation routes it"),
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
        ("leadsTo", "Reaction",
         "this reaction led to that next reaction in the SAME episode, as an explicit "
         "ordered sequence (scared -> goosebumps -> fled). Use ONLY when the transcript "
         "shows the order. If the reactions are independent responses to the same thought, "
         "do NOT link them — leave them parallel."),
        ("becomesSituation", "Situation",
         "this reaction itself became the trigger for a NEW thought (panic loop) — only "
         "if the transcript clearly shows the client noticing the reaction and a new "
         "thought following"),
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

# reinforces: wide-window pass, NOT subject-anchored (Reaction × CoreBelief).
REINFORCES = ("reinforces", "Reaction", "CoreBelief")

# Allowed (predicate, subject_label, object_label) signatures (LLM edges only),
# built from the anchor families + reinforces. Uses ABSTRACT labels.
ALLOWED_SIGNATURES: frozenset[tuple[str, str, str]] = frozenset(
    [(p, subj, o) for subj, fams in ANCHOR_FAMILIES.items() for (p, o, _h) in fams]
    + [REINFORCES]
)

# predicate -> set of allowed object ABSTRACT labels (for Stage 4a-repair re-pointing)
PREDICATE_OBJECTS: dict[str, set[str]] = {}
for _p, _s, _o in ALLOWED_SIGNATURES:
    PREDICATE_OBJECTS.setdefault(_p, set()).add(_o)

# Deterministic structural edges (no LLM): added in the final pipeline step.
DETERMINISTIC_PREDICATES = frozenset({
    "hasSession", "hasProblem", "hasIntervention", "hasHomework",
    "evidencedBy", "inSession",
})

# Never produced by the LLM in the chain pass.
FORBIDDEN_LLM_PREDICATES = frozenset({
    # reinforces is allowed ONLY via the dedicated wide-window pass, never the
    # subject-anchored pass — enforced in stage3, not here.
})

# Hard disjointness (must never appear, regardless of predicate).
DISJOINT_RULES: list[tuple[str, str, str]] = [
    ("AutomaticThought", "*", "AutomaticThought"),     # no AT->AT edges
    ("CoreBelief", "influencesPerceptionOf", "Situation"),  # skip-level dropped
    ("Intervention", "appliedTo", "Goal"),             # goals are outcomes, not targets
]

# Predicate -> Neo4j relationship type (Stage 5).
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

# ===========================================================================
# JSON export maps (match demo1_ground_truth_v4.json shape)
# ===========================================================================

ID_PREFIX: dict[str, str] = {
    "Client": "client", "Session": "session", "Problem": "prob", "Goal": "goal",
    "Intervention": "intv", "Homework": "hw", "CoreBelief": "cb",
    "IntermediateBelief": "ib", "Situation": "sit", "AutomaticThought": "at",
    "Reaction": "react", "AdaptiveResponse": "adapt",
}

# Which property key holds the node's main text, per family (for JSON `properties`).
TEXT_PROP: dict[str, str] = {
    "Problem": "description", "Goal": "statement", "Intervention": "techniqueLabel",
    "Homework": "taskDescription", "CoreBelief": "content",
    "IntermediateBelief": "content", "Situation": "description",
    "AutomaticThought": "content", "Reaction": "content", "AdaptiveResponse": "content",
}

# ===========================================================================
# Deterministic lexicons (STARTER lists — expand from corpus before trusting)
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
# English fallbacks (translated transcripts)
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
