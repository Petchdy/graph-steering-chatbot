# CBT Pipeline Report (V4_flat)

**Generated:** 2026-06-21 12:40:11  
**Turns:** 3  |  **Raw nodes:** 5  |  **After merge:** 4  |  **Edges:** 5

---

## Stage 1: Extracted Entities

| Class | Raw nodes |
|-------|-----------|
| Problem | 1 |
| Intervention | 1 |
| Situation | 1 |
| AutomaticThought | 1 |
| Reaction | 1 |
| **Total** | **5** |

### Problem

| ID | Text | Discriminator | Turns |
|----|------|-----------|-------|
| 5 | I will never find someone who loves me / I am unlovable due to age and past breakups | — | 1, 2 |

### Intervention

| ID | Text | Discriminator | Turns |
|----|------|-----------|-------|
| 4 | The therapist acknowledges the client's breakup and resulting negative cascade of thoughts about their future. They introduce 'laddering,' a technique to organize automatic thoughts (compared to an iceberg tip) by uncovering submerged core beliefs that drive feelings and behaviors. Starting with the thought 'I will never find somebody who loves me,' the therapist asks what this means to the client, aiming to trace how this specific belief influences their experience. | — | 2 |

### Situation

| ID | Text | Discriminator | Turns |
|----|------|-----------|-------|
| 1 | recently broke up with a long long-term partner | — | 1 |

### AutomaticThought

| ID | Text | Discriminator | Turns |
|----|------|-----------|-------|
| 2 | I will never find somebody who loves me. | — | 1 |

### Reaction

| ID | Text | Discriminator | Turns |
|----|------|-----------|-------|
| 3 | my depression is through the roof because of it | — | 1 |

---

## Stage 2: Merge Decisions

**Problem**: 1 raw → 1 survivors
**Goal**: 0 raw → 0 survivors
**Intervention**: 1 raw → 1 survivors
**Homework**: 0 raw → 0 survivors
**CoreBelief**: 0 raw → 0 survivors
**IntermediateBelief**: 0 raw → 0 survivors
**Situation**: 1 raw → 1 survivors
**AutomaticThought**: 1 raw → 1 survivors
**Reaction**: 1 raw → 0 survivors (0 merged at Stage 2, 1 dropped at Stage 1.5)
**AdaptiveResponse**: 0 raw → 0 survivors

*(no merges performed)*

---

## Stage 3 + Repair: Extracted Edges

**5 extracted** + **0 repaired** = 5 total

| Subject | Predicate | Object | Evidence | Reason |
|---------|-----------|--------|----------|--------|
| id=-1 | **hasSession** | id=-2 | — | — |
| id=-2 | **hasProblem** | I will never find someone who loves me / I am unlovable due to age and past breakups | — | — |
| id=-2 | **hasIntervention** | The therapist acknowledges the client's breakup and resulting negative cascade of thoughts about their future. They introduce 'laddering,' a technique to organize automatic thoughts (compared to an iceberg tip) by uncovering submerged core beliefs that drive feelings and behaviors. Starting with the thought 'I will never find somebody who loves me,' the therapist asks what this means to the client, aiming to trace how this specific belief influences their experience. | — | — |
| recently broke up with a long long-term partner | **triggers** | I will never find somebody who loves me. | 1 | The subject explicitly states the situation (recent breakup) leads to the automatic thought 'i might never find somebody who loves me'. |
| The therapist acknowledges the client's breakup and resulting negative cascade of thoughts about their future. They introduce 'laddering,' a technique to organize automatic thoughts (compared to an iceberg tip) by uncovering submerged core beliefs that drive feelings and behaviors. Starting with the thought 'I will never find somebody who loves me,' the therapist asks what this means to the client, aiming to trace how this specific belief influences their experience. | **appliedTo** | I will never find somebody who loves me. | 2 | The therapist explicitly selects the thought 'I might never find somebody who will love me' (from turn 1) as the target for the laddering technique in turn 2. |

---

## Stage 3 audit (per anchor)

Per-subject record: candidates offered to the LLM vs edges it proposed. A `0 / N` row means the anchor had N candidates but the LLM proposed nothing.

| Subject | Predicate | Candidates | Proposed | Note |
|---------|-----------|-----------:|---------:|------|
| [Situation] recently broke up with a long long-term partner | **triggers** | 1 | 1 |  |
| [AutomaticThought] I will never find somebody who loves me. | **associatedWith** | 1 | 1 |  |
| [Problem] I will never find someone who loves me / I am unlovable due  | **manifestsAs** | 1 | 0 |  |
| [Intervention] The therapist acknowledges the client's breakup and resultin | **appliedTo** | 2 | 2 |  |

---

## Stage 4: validate + verify

- **Kept:** 5
- **Dropped at 4a (deterministic):** 0
- **Repaired at 4a:** 0
- **Dropped at 4b (LLM verify):** 2

### Dropped at 4b
| Predicate | Subject | Object | Reason |
|-----------|---------|--------|--------|
| associatedWith | I will never find somebody who loves me. | I will never find someone who loves me / I am unlovable due to age and past breakups | The evidence does not support a direct 'associatedWith' link between the specific automatic thought and a distinct problem node; the text describes the thought as an iceberg tip rather than linking it to a separate underlying problem entity at this stage. |
| appliedTo | The therapist acknowledges the client's breakup and resulting negative cascade of thoughts about their future. They introduce 'laddering,' a technique to organize automatic thoughts (compared to an iceberg tip) by uncovering submerged core beliefs that drive feelings and behaviors. Starting with the thought 'I will never find somebody who loves me,' the therapist asks what this means to the client, aiming to trace how this specific belief influences their experience. | I will never find someone who loves me / I am unlovable due to age and past breakups | Similar to Candidate 2, the evidence focuses on examining the automatic thought itself as an iceberg tip; there is no explicit textual support for linking this specific intervention directly to a separate 'Problem' node defined by age/breakups at this moment. |

---

## Extraction Failures

### Stage 1 — Silent turns (1 turns produced no entities)

| Turn | Speaker | Text |
|------|---------|------|
| 0 | therapist | what brings you in today |
