"""All prompts used by the V4_flat CBT chatbot.

V4_flat-derived prompts (ported verbatim from V4_flat/cbt_stage*) drive the
extraction pipeline. The hard-coded "The text is in Thai" line has been
parameterized into a {language} slot (default: English). THERAPIST_SYSTEM is
the CACTUS therapist prompt (V4_flat is an extractor — does not provide one).
QUERY_PARSE / QUERY_ANSWER drive Part 2's natural-language query engine.
"""

from __future__ import annotations

# ===========================================================================
# Extraction prompts (Part 1)
# ===========================================================================

EXTRACT_PROMPT = """You extract CBT entities from a therapy transcript. Text is in {language}.
Keep entity text faithful to the speaker's words; extract the core concept,
not whole sentences.

CLASS DEFINITIONS:
{defs}

GLOBAL RULES:
1. Extract ONLY what is stated in the TARGET TURN. Do not infer or embellish.
2. If nothing fits, return []. Never invent a node to fill a gap.
3. One node per distinct item. Do not split one statement into many, or merge two.
4. Keep emotions OUT of AutomaticThought content (the feeling is a separate Reaction).
5. Do NOT extract a therapist question as an AutomaticThought.
6. Speaker prior — this turn's speaker is {speaker}. Typically extract: {prior}.
   (Goals and AdaptiveResponses may come from either speaker.)
7. Speaker grounding — Reaction / AutomaticThought / CoreBelief / IntermediateBelief
   must be grounded in the CLIENT's own words or explicit elaboration. Therapist
   framing language alone is not sufficient even if the client gives a minimal "yeah";
   the client must elaborate or clearly affirm in their own words.
8. `associatedWith` is a FALLBACK edge — attempt the full Problem→Situation→
   AutomaticThought chain first; only fall back to `associatedWith` when no Situation
   can be identified for an AutomaticThought.

The surrounding CONTEXT is for understanding only; entity text must be grounded in
the TARGET TURN.

CONTEXT:
{context}

TARGET TURN [{idx}] ({speaker}): {target}

For each entity output an object:
  {{"label": "<one class name>", "text": "<core concept, faithful wording>",
    "group_key": "<your best subclass guess or null>"}}
group_key is optional — null is acceptable (it is re-checked later).

Output ONLY a JSON array. Start with [ and end with ]. No other text.
Empty array [] if the target turn has nothing to extract."""


ATOMIZE_PROMPT = """You clean and split extracted CBT {class_label}s. Text is in {language}.

DEFINITION of {class_label}: {class_definition}

Rewrite the text into one or more ATOMIC, self-contained, first-person propositions:
- SPLIT only when the text contains genuinely DISTINCT {unit}s (e.g. two separate
  beliefs). Do NOT split one idea into clauses.
- CONDENSE rambling into a short, clear statement.
- {self_belief_clause}
- Stay FAITHFUL: add no meaning, infer nothing, do not generalize. {at_specific}
- Most inputs are a single idea — ONE cleaned proposition is the common, correct answer.

TEXT: '{node_text}'

Output ONLY a JSON array of strings. Start with [ and end with ].
1 to {max_splits} items. Most inputs are a single idea — a 1-item array is correct."""


SELF_BELIEF_CLAUSE = (
    "If the text states a belief about the SELF but phrases it through other people's "
    "eyes ('people will say I am ...', 'everyone can see I am ...', 'they think I am "
    "...'), rewrite it as a direct first-person statement ('I am ...'). Change ONLY the "
    "framing — keep the meaning identical."
)

AT_SPECIFIC_CLAUSE = (
    "Keep each thought situation-specific; keep emotions OUT "
    "(a feeling is a separate Reaction)."
)


# ── Property-classification prompts ───────────────────────────────────────

PROPERTY_CLASSIFY_PROMPT = """{task}
Texts are in {language}.

ALLOWED VALUES (choose exactly one):
{gloss_block}

{candidates}

Return one object per item: [{{"item":1,"{field}":"<value>"}}, ...].{extra}

Output ONLY a JSON array starting with [ and ending with ]."""


PROPERTY_ISOPTIONAL_PROMPT = """For each homework task, decide if the therapist framed it as
OPTIONAL (explicit "if you want to", "you don't have to") rather than expected.
Text is in {language}.

{candidates}

Return one object per item: [{{"item":1,"isOptional":true|false}}, ...].

Output ONLY a JSON array starting with [ and ending with ]."""


PROPERTY_TASKS = {
    "distortionType": "Label the cognitive-distortion pattern of each automatic thought. "
                      "Use 'none' if it is accurate or no pattern fits — do not force one.",
    "modality": "Label whether each automatic thought is a worded thought or a mental image.",
    "kind": "Identify the trigger CHANNEL of each situation (no time meaning).",
    "technique": "Identify which therapeutic technique each intervention uses. "
                 "Beck techniques outside the CACTUS-12 list must be 'other' with a "
                 "techniqueLabel (e.g. technique=other, techniqueLabel='laddering / downward arrow'; "
                 "technique=other, techniqueLabel='psychoeducation'). "
                 "Do not force-fit non-CACTUS-12 techniques into the nearest enum value.",
    "taskType": "Classify each homework task by type.",
    "category": "Categorize each self-directed core belief into Beck's three categories.",
    "domain_problem": "Identify the life domain of each Problem.",
    "domain_corebelief": "Identify the domain of each CoreBelief (self / world / others).",
    "subtype_ib": "Identify the subtype of each IntermediateBelief (attitude / rule / assumption).",
    "channel_reaction": "Identify the channel of each Reaction (emotional / behavioral / physiological).",
}


# ── Edge anchor prompts ────────────────────────────────────────────────────

EDGE_ANCHOR_PROMPT = """You extract CBT relationships from a therapy transcript ({language}).

SUBJECT ({subj_label}): '{subj_text}'

CONTEXT:
{context}

For the subject above, decide which of the following relationships hold. Each
relationship lists its candidate OBJECT nodes (numbered within that relationship).
Assert a relationship when the transcript supports the directional link from THIS
subject to THAT object. Do not invent links from pure theme; but do not under-
extract either — Beck's basic cognitive model expects these canonical edges:

  - a Situation usually `triggers` at least one AutomaticThought
  - an AutomaticThought usually `leadsTo` at least one Reaction
  - an AutomaticThought usually `stemsFrom` at least one CoreBelief OR IntermediateBelief
    (evaluate independently per thought — thoughts from the same Situation may stem from
    DIFFERENT CoreBeliefs; use IntermediateBelief as target when the thought is driven by
    the rule/demand itself, not by an identity claim)
  - a CoreBelief usually `givesRiseTo` at least one IntermediateBelief; fan-out to
    multiple IntermediateBeliefs from a single CoreBelief is normal — do not cap
  - `associatedWith` is a FALLBACK — use only when no Situation can be identified
  - a Problem usually `manifestsAs` at least one specific Situation
  - an Intervention is `appliedTo` whatever target it surfaced or examined
  - a Homework `targets` whatever it works on

If a candidate plausibly fits one of these canonical patterns and the topic is
consistent, fire the edge. An empty array is correct only when the transcript
clearly does not support any candidate.

{families}

Output one object per asserted edge:
  {{"relation": "<name>", "object": <number within that relation's list>,
    "reason": "<short>"{intensity}}}

Output ONLY a JSON array starting with [ and ending with ].
Empty array [] is a correct answer when no relationship holds."""


REINFORCES_PROMPT = """In this CBT session, which client REACTIONS are maintaining
or strengthening which CORE BELIEFS (a feedback loop that keeps the belief in
place)? Text is in {language}. Assert a pair only when the transcript shows the
reaction feeding back to the belief. Many sessions have none.

REACTIONS:
{reactions}

CORE BELIEFS:
{beliefs}

Output one object per pair: [{{"reaction":<n>,"belief":<n>,"reason":"<short>"}}, ...].

Output ONLY a JSON array starting with [ and ending with ].
Empty array [] is a correct answer when no relationship holds."""


# ── Session-level pass ─────────────────────────────────────────────────────

SESSION_LEVEL_PROMPT = """You are reviewing a full CBT therapy session transcript to extract
{class_label} entities that span multiple turns or only become clear when you
see the whole session — concepts a per-turn extractor would miss. Text is in {language}.

DEFINITION of {class_label}: {class_definition}

ALREADY EXTRACTED per-turn — same class (these may be partial, duplicated, or
mis-classified; correct them if needed):
{same_class_priors}

ADJACENT CONTEXT — other classes for grounding only (do NOT re-extract these):
{adjacent_class_priors}

TRANSCRIPT:
{transcript}

EXTRACTION RULES:
1. Output the {class_label}s the session as a whole supports — including any
   the per-turn extractor missed.
2. For each output, pick 1–3 MOST-EVIDENTIARY turn indices where the concept is
   grounded. Do not list every mention; pick the strongest turns.
3. Therapist meta-commentary about CBT terminology is NOT a node — extract only
   what the client actually believes or experiences.
4. Stay faithful: do not invent concepts not grounded in real evidence turns.
5. Intellectual disavowal: if a client denies a CoreBelief shortly after it clearly
   surfaced (e.g. "I don't think I'm worthless"), this is normal disavowal and does
   NOT negate extraction — keep the node. Remove a CoreBelief only if the transcript
   shows genuine, sustained revision of the belief, not a reflex denial.
6. CoreBelief dedup: two CoreBeliefs with DIFFERENT `category` values (helpless /
   unlovable / worthless) are DIFFERENT nodes even if their domain is both "self" or
   their text is somewhat similar. Do not collapse them into one.

Output JSON array, one object per {class_label}:
[{{"label":"{class_label}","text":"<short>","evidence_turns":[<int>,...]}}]

Output ONLY a JSON array. Start with [ and end with ]. Empty [] if nothing new
is supported."""


# ===========================================================================
# THERAPIST_SYSTEM — Part 1 generator system prompt (CACTUS, V4_flat-aligned)
# ===========================================================================

THERAPIST_SYSTEM = """You are a compassionate CBT therapist in a real counseling session.
Follow CACTUS principles: guided discovery, not advice-giving.

CORE RULES:
- QUESTIONER, not answer-giver. Help the client discover their own insights.
- Empathize only with what the client has actually said. Do not project emotions.
- Exactly ONE question per turn. Plain, warm language. No clinical jargon.
- Do not name CBT techniques to the client.

PHASES (advance only when criteria below are met; the system enforces gates):
  Rapport (turns 1-3) → Exploration once a Problem has emerged.
  Exploration (turns 4-8) → Technique once an AutomaticThought + Situation are on the table AND ≥5 turns.
  Technique (turns 9+) → Consolidation once an AdaptiveResponse has been internalised AND ≥12 turns.
  Consolidation: help the client articulate insight and next steps.

TECHNIQUES (CACTUS-12 — choose the one most fitting; never name it to the client):
  efficiencyEvaluation, pieChartTechnique, alternativePerspective, decatastrophizing,
  prosAndConsAnalysis, evidenceBasedQuestioning, realityTesting, continuumTechnique,
  changingRulesToWishes, behaviorExperiment, problemSolvingSkillsTraining, systematicExposure.
Use "Rapport Building" / "Assessment" when no specific technique is active.

CURRENT SESSION STATE (read-only — do not echo to client):
{cbt_context}

RESPONSE FORMAT — output ONLY this JSON object, nothing else:
{{"response": "<your warm, 1-2 sentence reply with exactly one question>",
  "technique": "<one of: efficiencyEvaluation | pieChartTechnique | alternativePerspective | "
  "decatastrophizing | prosAndConsAnalysis | evidenceBasedQuestioning | realityTesting | "
  "continuumTechnique | changingRulesToWishes | behaviorExperiment | problemSolvingSkillsTraining | "
  "systematicExposure | Rapport Building | Assessment>",
  "phase": "<Rapport | Exploration | Technique | Consolidation>"}}"""


# ===========================================================================
# Query prompts (Part 2)
# ===========================================================================

QUERY_PARSE_PROMPT = """You translate a therapist's natural-language question about a CBT
session into a structured query over the V4_flat knowledge graph.

V4_flat NODE CLASSES (use these exact names):
{node_classes}

V4_flat EDGE PREDICATES (use these exact names):
{predicates}

PROPERTY ENUMS (use only these values for filters):
- Problem.domain: academic, work, social, family, financial, health, other
- CoreBelief.domain: self, world, others
- CoreBelief.category: helpless, unlovable, worthless
- IntermediateBelief.subtype: attitude, rule, assumption
- Reaction.channel: emotional, behavioral, physiological
- Reaction.valence: positive, negative
- Situation.kind: externalSituation, thoughtStream, image, emotion, behavior, physiological
- Situation.temporality: past, present, anticipated
- AutomaticThought.modality: verbal, image
- AutomaticThought.distortionType: allOrNothing, catastrophizing, discountingPositive,
  fortuneTelling, labeling, mentalFiltering, mindReading, overgeneralization,
  personalization, shouldStatements, none
- Intervention.technique: CACTUS-12 enum
- Homework.taskType: thoughtRecord, behavioralExperiment, activityScheduling,
  copingCard, skillsPractice, reading, other

INTENTS:
  list       — return all matching nodes
  trace      — walk a chain along given predicates (e.g. Situation→AT→Reaction)
  count      — return counts by label
  describe   — describe one node and its immediate neighbours
  summarize  — high-level overview of the whole graph, plus the dialogue

DIALOGUE QUESTIONS:
Utterance nodes hold what was actually said (props: text, speaker, turnIndex).
- Asking what someone *said*, or for their exact/own words, or what was
  discussed at a point in the session -> intent "list", node_labels ["Utterance"].
- Asking broadly what the whole session or conversation was about -> intent
  "summarize" (it returns the transcript).
- Do NOT put a property_filter on Utterance text; retrieval is by label, and the
  answerer picks the relevant lines out of what you return.

PROPERTY FILTERS:
Keys must be the BARE property name as stored on the node — "domain", not
"Problem.domain" (the enums are listed qualified above only to show which class
each belongs to). Only add a filter the question actually asks for; a filter
that no node matches returns nothing at all.

USER QUESTION:
{question}

Return ONLY a JSON object of this shape (omit unused keys):
{{"intent": "list | trace | count | describe | summarize",
  "node_labels": ["<V4_flat class>", ...],
  "predicates": ["<V4_flat predicate>", ...],
  "property_filters": {{"<prop>": "<value>", ...}},
  "free_text": "<residual question to pass to the answerer>"}}"""


QUERY_ANSWER_PROMPT = """You are a clinical assistant answering a therapist's question about
a CBT session graph. You are given a structured RESULT SET extracted from the
graph by a deterministic query engine.

RULES:
- Answer ONLY from the result set. Do not invent nodes, edges, properties, or
  conclusions that are not present.
- Cite node ids and evidence turn indices in parentheses, e.g. (at_2, turns 7,9).
- Nodes and edges may carry "evidence_quotes" — what was actually said on those
  turns — and a "transcript" key may carry the dialogue in order. Quote this
  wording verbatim when it supports the answer, attributing the speaker, e.g.
  the client said "I froze in the meeting" (turn 3). Quote only what appears
  there; never paraphrase a quote into something that was not said.
- If the question is about the conversation itself (what was said, discussed,
  or asked) answer from those quotes. If the result set has no quotes, say the
  dialogue was not recorded for this graph rather than guessing at it.
- If the result set is empty, say "this isn't in the session's graph."
- Be concise. Therapist-facing tone.
- Do not name CBT techniques in jargon — but it is fine to use V4_flat terms
  (CoreBelief, AutomaticThought, etc.) since the therapist is the audience.

ORIGINAL QUESTION:
{question}

RESULT SET (JSON — nodes, edges, evidence):
{result_set}

Write the answer (plain text)."""
