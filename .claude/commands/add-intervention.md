# Add New Therapy Intervention

You are helping a CBT therapist add a new intervention module to the monitor agent at
`final_code/monitor_agent/monitor_agent.py`.

Conduct a structured interview with the therapist, then implement all required code changes.
Do not start implementing until the interview is complete and you have confirmed every detail.

---

## Step 1 — Interview the therapist

Ask these questions **one section at a time**, waiting for answers before continuing.

### Section A: Intervention overview
1. What is the name of this intervention? (e.g. "Cognitive Restructuring")
2. What is the snake_case identifier for it? (e.g. `cognitive_restructuring`) — this becomes the `therapy_stage` value and node name.
3. In 1–2 sentences, what does this intervention help the patient do?
4. What ABC data does it rely on? (situation, thought, emotion, behavior — or some subset)

### Section B: Phases
For each phase, collect:
- **Phase identifier** (snake_case, e.g. `identify`, `challenge`, `reframe`)
- **What the therapist is trying to elicit** in this phase
- **Completion criteria** — what must the patient have said for this phase to be done?
- **What to extract** when complete — one or more named data points (e.g. `extracted_core_belief: str`)

The final phase should transition to `"complete"` as its `next_phase`, which silently routes to `action_plan`.
Confirm the full phase sequence before moving on.

### Section C: Prompts
For each phase, ask the therapist:
- What tone/approach should the therapist take? (e.g. "gentle Socratic", "collaborative worksheet")
- Any specific instructions, constraints, or forbidden behaviours for this phase?

Use these to draft a prompt string. Show the draft to the therapist and ask for approval or edits before continuing.

### Section D: Treatment selection
The `select_treatment` node auto-selects an intervention using an LLM. Ask:
- In what situations should this intervention be selected over thought_record / socratic_questioning / behavioral_experiment?
  (e.g. "when the core belief is rigid and the patient resists reframing")

---

## Step 2 — Confirm the full spec

Before writing any code, present a summary table:

| Phase | Extracts | Completion criteria |
|-------|----------|---------------------|
| ...   | ...      | ...                 |

Ask the therapist to confirm or correct it.

---

## Step 3 — Implement

Make all of the following changes to `final_code/monitor_agent/monitor_agent.py` **in order**.
Read the file before editing. After all edits are done, read the changed sections back to verify correctness.

### 3a. Add to `TherapyStage` Literal (line ~15)
Add `"<stage_id>"` to the `TherapyStage = Literal[...]` definition.

### 3b. Add to `TreatmentSelected` (line ~417)
Add `"<stage_id>"` to the `selected: Literal[...]` field so the LLM can select this intervention.
Update the field description to mention when to choose it.

### 3c. Add prompt constants (after the existing prompt constants, before the Pydantic models section)
Add one `ALL_CAPS_PROMPT` string constant per phase. Follow this format exactly:

```python
# ── <InterventionName> prompts ────────────────────────────────────────────────

XX_PHASE_ONE_PROMPT = """You are a warm CBT therapist. <therapist instructions for this phase>.

<Any constraints or forbidden behaviours.>
Respond in one focused question or reflection. Do not use lists or headers."""

XX_PHASE_TWO_PROMPT = """..."""
```

### 3d. Add completion check Pydantic models (after the last existing completeness model block)
Add one class per phase following this exact pattern:

```python
class XXPhaseOneComplete(BaseModel):
    is_complete: bool = Field(description="<completion criterion>")
    extracted_field_name: str = Field(default="", description="<what was extracted, if complete>")
    reasoning: str = Field(description="Why this is or isn't complete enough to advance")
```

Each class must have:
- `is_complete: bool`
- `reasoning: str`
- One `extracted_*` field per data point collected in that phase (all `default=""`)

### 3e. Add state fields to `MonitorTherapistState` (line ~539)
Add a new comment block and fields:

```python
# <InterventionName>
xx_phase: Literal["phase_one", "phase_two", ..., "complete"] = Field(default="phase_one")
xx_phase_start_index: int = Field(default=0)
xx_extracted_field_one: str = Field(default="")
xx_extracted_field_two: str = Field(default="")
```

Use the two-letter prefix consistently (e.g. `cr_` for cognitive_restructuring).

### 3f. Add `_xx_phase_messages` helper (after the last existing phase-messages helper, line ~647)

```python
def _xx_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    return state.messages[state.xx_phase_start_index:]
```

### 3g. Add the node function (after `behavioral_experiment`, before `action_plan`)
Follow this exact pattern — do not deviate:

```python
def <stage_id>(state: MonitorTherapistState):
    """<One-line description of what this node does>."""
    phase = state.xx_phase
    if phase == "complete":
        new_start = len(state.messages)
        return {
            "therapy_stage": "action_plan",
            "ap_phase": "propose",
            "stage_start_index": new_start,
            "ap_phase_start_index": new_start,
        }

    phase_config = {
        "phase_one": (
            XX_PHASE_ONE_PROMPT.format(<abc fields used>),
            XXPhaseOneComplete,
            "phase_two",
        ),
        "phase_two": (
            XX_PHASE_TWO_PROMPT.format(<abc fields and prior extracted fields>),
            XXPhaseTwoComplete,
            "complete",
        ),
        # ... etc
    }

    prompt, check_model, next_phase = phase_config[phase]

    check_llm = model.with_structured_output(check_model)
    phase_msgs = _xx_phase_messages(state)
    if not phase_msgs:
        check = check_model(is_complete=False, reasoning="No messages in phase yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(f"Assess whether the '{phase}' phase of <InterventionName> is complete based on the conversation."),
            *phase_msgs,
        ])
    if DEBUG:
        print(f"[XX {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_phase_start = len(state.messages)
        updates = {
            "xx_phase": next_phase,
            "xx_phase_start_index": new_phase_start,
        }
        if phase == "phase_one":
            updates["xx_extracted_field_one"] = check.extracted_field_one
        elif phase == "phase_two":
            updates["xx_extracted_field_two"] = check.extracted_field_two
        # ... etc
        return updates

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
```

### 3h. Update `select_treatment` phase initialisation (line ~921)
Add an `elif` branch for the new intervention:

```python
elif selected == "<stage_id>":
    updates["xx_phase"] = "phase_one"
    updates["xx_phase_start_index"] = new_stage_start
```

### 3i. Register the node in the graph (line ~1390)
Add:
```python
monitor_graph.add_node("<stage_id>", <stage_id>)
```

### 3j. Add to `route_after_classify` target map (~line 1420)
Add `"<stage_id>": "<stage_id>"` to the `add_conditional_edges` call for `"classify"`.

### 3k. Add to `_stage_nodes` list (~line 1413)
Add `"<stage_id>"` to the `_stage_nodes` list so it gets the `route_after_stage` conditional edge.

---

## Step 4 — Verify

After all edits:
1. Read back the `TherapyStage` Literal, `TreatmentSelected`, `MonitorTherapistState`, the new node function, and the graph construction block to confirm correctness.
2. Point out any inconsistencies to the therapist (e.g. a phase that uses an extracted field before it's been set).
3. Commit with a descriptive message referencing the intervention name.
