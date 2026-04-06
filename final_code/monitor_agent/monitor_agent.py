from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

DEBUG = False
model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=8192)

ConvoFlag = Literal["conversation", "conceptualize_case", "crisis_categorize"]
TherapyStage = Literal[
    "mood_check",               # Auto-fires: opens the session
    "agenda_setting",           # What would you like to work on today?
    "abc_situation",            # A: get the specific activating event
    "abc_thought",              # B: get the automatic thought in that moment
    "abc_consequence",          # C: get emotion + behavior/feared action
    "select_treatment",         # LLM auto-selects + proposes treatment module
    "thought_record",           # Beck's Thought Record (7 phases)
    "cognitive_restructuring",  # Core belief restructuring (4 phases)
    "behavioral_experiment",    # Hypothesis-testing experiment (3 phases)
]


# ── System prompts (progressive disclosure) ───────────────────────────────────

MOOD_CHECK_PROMPT = """You are a warm, conversational CBT therapist opening a session.
Ask the patient how they are feeling today in one open, natural question.
Do not pile on multiple questions."""

AGENDA_PROMPT = """You are a CBT therapist. The patient has checked in on how they're feeling.
Your goal: help them set a clear agenda — what specifically would they like to work on today?
Keep it conversational. If their answer is vague (e.g. "stress"), ask one gentle follow-up to make it concrete.
Do not start exploring a situation yet — just nail down the topic."""

ABC_SITUATION_PROMPT = """You are a CBT therapist working within the ABC model (Beck's CBT).
The patient has set their agenda. Now you need the Activating Event (A).

Your goal: get one specific, concrete moment — not a general pattern.
- What happened? (observable facts, who was involved, what was said or done)
- When and where? (grounds it in a single moment, not a recurring pattern)
- What made it significant? (bridge question that connects A to B)

Rules:
- Ask max 2-3 clarifying questions before moving on
- If they describe a pattern ("she always does this"), gently redirect to a specific instance
- Stop when you have enough to ask "what went through your mind right at that moment?"
- Do NOT ask about thoughts or feelings yet — stay on the situation"""

ABC_THOUGHT_PROMPT = """You are a CBT therapist working within the ABC model.
Situation established: {situation}

Now you need the Belief / Automatic Thought (B) — the exact words that fired in their head at that moment.

Rules:
- Primary question: "What was going through your mind right at that moment?" or "What did that mean to you?"
- If they give an emotion instead of a thought (e.g. "I felt anxious"), redirect:
  "And when you felt that, what was the thought behind it? What were you telling yourself?"
- If they give a vague thought (e.g. "things felt bad"), ask:
  "What specifically did you think was going to happen?" or "What did that say to you about yourself?"
- If they give a clear thought, use downward arrow once: "And what did that mean to you?"
- Stop when you have a specific first-person thought clearly connected to the situation
- Do NOT ask about emotions or behaviors yet"""

ABC_CONSEQUENCE_PROMPT = """You are a CBT therapist working within the ABC model.
Situation: {situation}
Automatic thought: {thought}

Now you need the Consequences (C) — emotional and behavioral response.

Rules:
- Ask about emotion first, then behavior — not both at once
- For a past event: "How did you feel after that?" then "What did you do?"
- For a future fear: "What are you scared you're going to feel?" then "What are you afraid you might do?"
- Once you have both, warmly summarize the full ABC chain and transition to working on it"""

SELECT_TREATMENT_PROMPT = """You are a CBT therapist. The patient has just completed an ABC assessment.
Here is the ABC data:
  Situation: {situation}
  Automatic thought: {thought}
  Emotion: {emotion}
  Behavior: {behavior}

Choose the single most appropriate treatment module based on these heuristics:
  - thought_record: There is a clear hot automatic thought to examine — this is the most common default.
  - cognitive_restructuring: The thought points to a global "I am..." / "I always..." / "I never..." core belief schema.
  - behavioral_experiment: The core issue is a future prediction or avoidance behavior ("if I do X, Y will happen").

Return ONLY the module name: thought_record, cognitive_restructuring, or behavioral_experiment."""

SELECT_TREATMENT_PROPOSE_PROMPT = """You are a warm CBT therapist. You have chosen the treatment module: {selected}.

Write ONE brief, natural message (2-3 sentences) proposing this to the patient in plain, jargon-free language.

Module descriptions for your reference:
  - thought_record: "I'd like to look carefully at that thought together — examining the evidence for and against it — to see if we can find a more balanced perspective. It's called a Thought Record."
  - cognitive_restructuring: "It sounds like there might be a deeper belief driving this thought. I'd like to explore where that belief comes from and whether we can reshape it into something more helpful."
  - behavioral_experiment: "Since this is really about a prediction you're making, I'd like to design a small experiment you could run to test whether that prediction is actually true."

End with a short warm confirmation question like "Does that sound okay?" or "Would you be willing to try that?"."""

THOUGHT_RECORD_DISTORTIONS_PROMPT = """You are a CBT therapist running a Thought Record.
ABC data — Situation: {situation} | Thought: {thought} | Emotion: {emotion} | Behavior: {behavior}

Phase: Identify cognitive distortions (Q3 of the worksheet).
Ask the patient which thinking errors might be showing up in their automatic thought.
Offer 2-3 examples relevant to their thought (e.g. all-or-nothing, mind reading, catastrophising).
Keep it collaborative — one question at a time."""

THOUGHT_RECORD_EVIDENCE_PROMPT = """You are a CBT therapist running a Thought Record.
ABC data — Situation: {situation} | Thought: {thought} | Emotion: {emotion} | Behavior: {behavior}
Distortions identified: {distortions}

Phase: Evidence for and against (Q4-5).
First ask: "What's the evidence that supports this thought?" — let them answer.
Then ask: "What's the evidence AGAINST it, or that doesn't fit with it?" — facts, not feelings.
Be warm but gently challenge vague answers."""

THOUGHT_RECORD_ALTERNATIVES_PROMPT = """You are a CBT therapist running a Thought Record.
ABC data — Situation: {situation} | Thought: {thought}
Evidence for: {evidence_for} | Evidence against: {evidence_against}

Phase: Alternative/balanced perspective (Q6).
Ask the patient: "Given all the evidence, what's a more balanced way of looking at this?"
If they struggle, offer a gentle scaffold: "If a close friend was in this situation and had the same evidence, what might they think?"
The goal is a nuanced thought, not toxic positivity."""

THOUGHT_RECORD_DECATASTROPHIZE_PROMPT = """You are a CBT therapist running a Thought Record.
Automatic thought: {thought} | Alternative thought: {alternative}

Phase: Decatastrophise — what if the worst actually happened? (Q7).
Ask: "Let's say the worst-case scenario did happen — what would that actually mean for you? What would you do?"
Help them see they could cope. Keep it grounded."""

THOUGHT_RECORD_OUTCOMES_PROMPT = """You are a CBT therapist running a Thought Record.
Automatic thought: {thought} | Worst-case reflection: {decatastrophize}

Phase: Best and most probable outcomes (Q8-9).
Ask about the BEST realistic outcome first, then ask what the MOST PROBABLE outcome actually is.
Gently distinguish between fear-driven expectations and realistic probability."""

THOUGHT_RECORD_CONSEQUENCES_PROMPT = """You are a CBT therapist running a Thought Record.
Automatic thought: {thought} | Most probable outcome: {probable_outcome}

Phase: Consequences of keeping vs changing the thought (Q10-11).
Ask: "What happens if you keep believing the original thought — how does that affect how you feel and act?"
Then: "What might change for you — emotionally and behaviourally — if you adopted the more balanced view?"
Draw out the contrast clearly."""

THOUGHT_RECORD_ACTION_PROMPT = """You are a CBT therapist running a Thought Record.
Situation: {situation} | Original thought: {thought} | Alternative thought: {alternative}
Consequence of changing: {consequence_changing}

Phase: Friend perspective + concrete action step (Q12-13).
Ask: "What would you say to a close friend who had this exact thought in this situation?"
Then help them name ONE small, concrete action step they could take this week based on their new perspective.
End on a warm, encouraging note."""

COGNITIVE_RESTRUCTURING_IDENTIFY_PROMPT = """You are a CBT therapist doing Cognitive Restructuring.
ABC data — Situation: {situation} | Thought: {thought} | Emotion: {emotion}

Phase: Identify the core belief (downward arrow technique).
Start with the automatic thought and use gentle "what would that mean?" questions to drill down to the deeper belief.
Target an "I am...", "I always...", "People are...", or "The world is..." level belief.
Limit to 3-4 downward arrow questions — stop when you've reached a core schema."""

COGNITIVE_RESTRUCTURING_ORIGIN_PROMPT = """You are a CBT therapist doing Cognitive Restructuring.
Core belief identified: {core_belief}

Phase: Examine the origin of the belief.
Gently explore where this belief came from — early experiences, family messages, significant events.
Ask: "When do you first remember feeling this way?" or "Who or what taught you to think of yourself this way?"
Be slow and compassionate — this can be sensitive territory. One question at a time."""

COGNITIVE_RESTRUCTURING_CHALLENGE_PROMPT = """You are a CBT therapist doing Cognitive Restructuring.
Core belief: {core_belief} | Origin: {belief_origin}

Phase: Socratic challenge — examine the evidence.
Help the patient look at evidence FOR and AGAINST the core belief across their life, not just one incident.
Ask about exceptions, counter-examples, and what a compassionate observer might say.
Challenge gently — the goal is curiosity, not confrontation."""

COGNITIVE_RESTRUCTURING_MODIFY_PROMPT = """You are a CBT therapist doing Cognitive Restructuring.
Core belief: {core_belief}

Phase: Construct a modified, adaptive belief.
Help the patient craft a new belief that is:
  - More nuanced and conditional (not a global statement)
  - Realistic (not toxic positivity)
  - Compassionate to themselves
Ask: "Given everything we've explored, what might be a fairer, more accurate way to see yourself?"
Help them refine it until it feels believable."""

BEHAVIORAL_EXPERIMENT_DEFINE_PROMPT = """You are a CBT therapist setting up a Behavioral Experiment.
ABC data — Situation: {situation} | Thought: {thought} | Emotion: {emotion} | Behavior: {behavior}

Phase: Define the belief to test.
Identify the core prediction or fear driving the avoidance/behavior.
Ask: "What are you predicting will happen if you do X?" — make it specific and falsifiable.
The prediction should be something that can actually be tested in the real world."""

BEHAVIORAL_EXPERIMENT_DESIGN_PROMPT = """You are a CBT therapist setting up a Behavioral Experiment.
Belief/prediction to test: {belief_to_test}

Phase: Design the experiment.
Collaboratively design a concrete, manageable experiment:
  - WHAT will the patient do (specific behaviour)?
  - WHEN will they do it (time, place)?
  - WHAT will they observe/record to evaluate the outcome?
Make it realistic — small enough to actually do, meaningful enough to test the prediction."""

BEHAVIORAL_EXPERIMENT_PREDICT_PROMPT = """You are a CBT therapist setting up a Behavioral Experiment.
Belief to test: {belief_to_test} | Experiment design: {experiment_design}

Phase: State prediction + belief rating.
Ask the patient to:
  1. State exactly what they predict will happen in their experiment (as specifically as possible)
  2. Rate how strongly they believe that prediction right now (0-100%)
Explain that you'll revisit these after the experiment to compare.
End with encouragement and a clear commitment from the patient to carry out the experiment."""

ASSESS_PROMPT = """You are a CBT therapist. A patient may be at risk of self-harm.
Ask direct questions to assess immediate suicide risk: current ideation, plan, access to means, steps taken."""

DEESCALATE_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Give direct, actionable instructions to reduce the patient's access to means right now. Be specific and brief."""

RECOMMEND_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Tell the patient to call 911 or 988 immediately. Be direct. End your response with [REQUEST_HUMAN_CONSULTATION]."""


# ── Structured output models ──────────────────────────────────────────────────

class Extract(BaseModel):
    message: str = Field(description="Your response to the patient, keep it concise")
    reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")


class CasePersona(BaseModel):
    situation: str = Field(default="", description="Specific triggering event or circumstance")
    automatic_thoughts: list[str] = Field(default=[], description="Automatic thoughts triggered by the situation")
    cognitive_distortions: list[str] = Field(default=[], description="Cognitive distortions present in the automatic thoughts")
    emotions: list[str] = Field(default=[], description="Emotional responses experienced")
    behaviors: str = Field(default="", description="Observable behavioral responses or coping mechanisms")


crisis_type = Literal["no_crisis", "harm_to_self"]


class Classify(BaseModel):
    reasoning: str = Field(description="A concise summary of the reasoning that justifies the final classification decision.")
    classification: crisis_type = Field(description="The final crisis category determined for the user's message.")


class SituationComplete(BaseModel):
    is_complete: bool = Field(description="True if we have a single specific concrete moment — not a pattern or vague description")
    extracted_situation: str = Field(default="", description="Brief summary of the specific situation if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to thoughts")


class ThoughtComplete(BaseModel):
    is_complete: bool = Field(description="True if we have a specific first-person automatic thought clearly connected to the situation")
    extracted_thought: str = Field(default="", description="The automatic thought if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to consequences")


class ConsequenceComplete(BaseModel):
    is_complete: bool = Field(description="True if we have both an emotional consequence AND a behavioral response or feared action")
    extracted_emotion: str = Field(default="", description="The emotional consequence if complete, else empty")
    extracted_behavior: str = Field(default="", description="The behavioral response or feared action if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to therapy work")


class TreatmentSelected(BaseModel):
    selected: Literal["thought_record", "cognitive_restructuring", "behavioral_experiment"] = Field(
        description="The treatment module selected based on the ABC content"
    )
    reasoning: str = Field(description="Why this module best fits the patient's ABC data")


# ── Thought Record completeness models ───────────────────────────────────────

class TRDistortionsComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has named at least one cognitive distortion")
    extracted_distortions: str = Field(default="", description="Distortions identified, else empty")
    reasoning: str

class TREvidenceComplete(BaseModel):
    is_complete: bool = Field(description="True if we have evidence both FOR and AGAINST the automatic thought")
    extracted_evidence_for: str = Field(default="", description="Evidence supporting the thought")
    extracted_evidence_against: str = Field(default="", description="Evidence against the thought")
    reasoning: str

class TRAlternativeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has articulated a balanced alternative thought")
    extracted_alternative: str = Field(default="", description="The balanced alternative thought")
    reasoning: str

class TRDecatastrophizeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has reflected on coping with the worst case")
    extracted_decatastrophize: str = Field(default="", description="Summary of the decatastrophising reflection")
    reasoning: str

class TROutcomesComplete(BaseModel):
    is_complete: bool = Field(description="True if we have both a best-case and a most-probable outcome")
    extracted_best_outcome: str = Field(default="", description="The best realistic outcome")
    extracted_probable_outcome: str = Field(default="", description="The most probable outcome")
    reasoning: str

class TRConsequencesComplete(BaseModel):
    is_complete: bool = Field(description="True if we have consequences of keeping AND of changing the thought")
    extracted_consequence_keeping: str = Field(default="", description="Consequence of keeping the original thought")
    extracted_consequence_changing: str = Field(default="", description="Consequence of adopting the balanced thought")
    reasoning: str

class TRActionComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has stated a concrete action step")
    extracted_action_step: str = Field(default="", description="The concrete action step committed to")
    reasoning: str


# ── Cognitive Restructuring completeness models ───────────────────────────────

class CRCoreBeliefComplete(BaseModel):
    is_complete: bool = Field(description="True if a core belief ('I am...', 'I always...', 'People are...') has been surfaced")
    extracted_core_belief: str = Field(default="", description="The core belief identified")
    reasoning: str

class CROriginComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has reflected on where the core belief came from")
    extracted_belief_origin: str = Field(default="", description="Origin/source of the core belief")
    reasoning: str

class CRChallengeComplete(BaseModel):
    is_complete: bool = Field(description="True if the Socratic challenge has surfaced counter-evidence to the core belief")
    extracted_challenge_summary: str = Field(default="", description="Summary of the challenge and evidence explored")
    reasoning: str

class CRModifiedBeliefComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has articulated a modified, more adaptive belief")
    extracted_modified_belief: str = Field(default="", description="The new adaptive belief")
    reasoning: str


# ── Behavioral Experiment completeness models ─────────────────────────────────

class BEBeliefComplete(BaseModel):
    is_complete: bool = Field(description="True if a specific falsifiable prediction has been pinned down")
    extracted_belief_to_test: str = Field(default="", description="The specific prediction to be tested")
    reasoning: str

class BEDesignComplete(BaseModel):
    is_complete: bool = Field(description="True if the experiment has a concrete what/when/what-to-observe design")
    extracted_experiment_design: str = Field(default="", description="The experiment design")
    reasoning: str

class BEPredictComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has stated a prediction and a 0-100 belief rating")
    extracted_predicted_outcome: str = Field(default="", description="The stated prediction and belief rating")
    reasoning: str


# ── State ─────────────────────────────────────────────────────────────────────

class MonitorTherapistState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    # Tracks progress through the 3-step crisis protocol across turns:
    #   0 = no active crisis
    #   1 = ASSESS sent → next step is DE-ESCALATE
    #   2 = DE-ESCALATE sent → next step is RECOMMEND + [REQUEST_HUMAN_CONSULTATION]
    #   3 = protocol complete → resume normal therapy
    crisis_step: int = 0
    flag: ConvoFlag = Field(default="conversation")
    case: Optional[CasePersona] = None
    # Progressive disclosure
    therapy_stage: TherapyStage = Field(default="mood_check")
    stage_start_index: int = Field(default=0)  # index into messages where the current stage began
    abc_situation: str = Field(default="")
    abc_thought: str = Field(default="")
    abc_emotion: str = Field(default="")
    abc_behavior: str = Field(default="")
    # Treatment selection
    selected_treatment: str = Field(default="")
    # ── Thought Record state ──────────────────────────────────────────────────
    tr_phase: Literal[
        "distortions", "evidence", "alternatives", "decatastrophize",
        "outcomes", "consequences", "action", "complete"
    ] = Field(default="distortions")
    tr_phase_start_index: int = Field(default=0)
    tr_distortions: str = Field(default="")
    tr_evidence_for: str = Field(default="")
    tr_evidence_against: str = Field(default="")
    tr_alternative: str = Field(default="")
    tr_decatastrophize: str = Field(default="")
    tr_best_outcome: str = Field(default="")
    tr_probable_outcome: str = Field(default="")
    tr_consequence_keeping: str = Field(default="")
    tr_consequence_changing: str = Field(default="")
    tr_action_step: str = Field(default="")
    # ── Cognitive Restructuring state ─────────────────────────────────────────
    cr_phase: Literal[
        "identify_core", "examine_origin", "challenge", "modify", "complete"
    ] = Field(default="identify_core")
    cr_phase_start_index: int = Field(default=0)
    cr_core_belief: str = Field(default="")
    cr_belief_origin: str = Field(default="")
    cr_challenge_summary: str = Field(default="")
    cr_modified_belief: str = Field(default="")
    # ── Behavioral Experiment state ───────────────────────────────────────────
    be_phase: Literal[
        "define_belief", "design", "predict", "complete"
    ] = Field(default="define_belief")
    be_phase_start_index: int = Field(default=0)
    be_belief_to_test: str = Field(default="")
    be_experiment_design: str = Field(default="")
    be_predicted_outcome: str = Field(default="")


# ── Routing ───────────────────────────────────────────────────────────────────

def route_start(state: MonitorTherapistState) -> str:
    """Route from START based on flag, crisis_step, and therapy_stage."""
    if state.flag == "crisis_categorize":
        return "classify"
    if state.flag == "conceptualize_case":
        return "produce_case"
    # Crisis protocol overrides the CBT flow
    if state.crisis_step == 1:
        return "crisis_deescalate"
    if state.crisis_step == 2:
        return "crisis_recommend"
    # No messages yet — auto-open with mood check
    if len(state.messages) == 0:
        return "mood_check"
    # Always classify for crisis first, then dispatch to current stage
    return "classify"


def route_after_classify(state: MonitorTherapistState) -> str:
    """After classification, route to crisis protocol or the current therapy stage."""
    if state.crisis_classification and state.crisis_classification.classification == "harm_to_self":
        return "crisis_assess"
    # Dispatch to whichever stage the session is currently in.
    # mood_check should never appear here (it fires before any messages exist),
    # but fall back to agenda_setting defensively.
    if state.therapy_stage == "mood_check":
        return "agenda_setting"
    return state.therapy_stage


# ── Helper ────────────────────────────────────────────────────────────────────

def _stage_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current stage began."""
    return state.messages[state.stage_start_index:]


def _tr_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current TR phase began."""
    return state.messages[state.tr_phase_start_index:]


def _cr_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current CR phase began."""
    return state.messages[state.cr_phase_start_index:]


def _be_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current BE phase began."""
    return state.messages[state.be_phase_start_index:]


# ── Stage nodes ───────────────────────────────────────────────────────────────

def mood_check(state: MonitorTherapistState):
    """Auto-fires at session start. Generates the opening question without user input."""
    llm = model.with_structured_output(Extract)
    # Anthropic requires at least one human message — use a silent session-start cue
    response = llm.invoke([SystemMessage(MOOD_CHECK_PROMPT), HumanMessage("Begin session.")])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "therapy_stage": "agenda_setting",
        # Start agenda_setting's stage window AFTER the mood check AI message (index 1)
        # so the mood reply isn't mistaken for an agenda response
        "stage_start_index": 1,
    }


def agenda_setting(state: MonitorTherapistState):
    """Responds to the mood check and helps the patient set a concrete agenda."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(AGENDA_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    # Only advance if the user has responded to OUR agenda question — not just any message.
    # Check: is there a HumanMessage after the last AIMessage in the stage window?
    stage_msgs = _stage_messages(state)
    ai_indices = [i for i, m in enumerate(stage_msgs) if isinstance(m, AIMessage)]
    if ai_indices:
        last_ai = ai_indices[-1]
        user_after_agenda = [m for m in stage_msgs[last_ai + 1:] if isinstance(m, HumanMessage)]
        advance = len(user_after_agenda) >= 1
    else:
        advance = False  # we haven't asked the agenda question yet

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if advance:
        updates["therapy_stage"] = "abc_situation"
        updates["stage_start_index"] = len(state.messages) + 1
    return updates


def abc_situation(state: MonitorTherapistState):
    """Gets the Activating Event (A) — a specific, concrete moment."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(ABC_SITUATION_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    # Completeness check scoped to only the messages in this stage
    check_llm = model.with_structured_output(SituationComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = SituationComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient described a single specific concrete moment? "
                "We need: what happened, who was involved, and what made it significant. "
                "A general pattern or recurring complaint is NOT enough."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Situation check: complete={check.is_complete} — {check.reasoning}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if check.is_complete:
        updates["therapy_stage"] = "abc_thought"
        updates["abc_situation"] = check.extracted_situation
        updates["stage_start_index"] = len(state.messages) + 1
    return updates


def abc_thought(state: MonitorTherapistState):
    """Gets the Automatic Thought (B) — exact first-person thought in that moment."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([
        SystemMessage(ABC_THOUGHT_PROMPT.format(situation=state.abc_situation)),
        *state.messages,
    ])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    check_llm = model.with_structured_output(ThoughtComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = ThoughtComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient expressed a specific first-person automatic thought clearly connected to the situation? "
                "It must be a thought (e.g. 'I thought everyone thinks I'm incompetent'), not just an emotion label. "
                "Vague statements like 'things felt bad' are NOT enough."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Thought check: complete={check.is_complete} — {check.reasoning}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if check.is_complete:
        updates["therapy_stage"] = "abc_consequence"
        updates["abc_thought"] = check.extracted_thought
        updates["stage_start_index"] = len(state.messages) + 1
    return updates


def abc_consequence(state: MonitorTherapistState):
    """Gets the Consequences (C) — emotion and behavior/feared action."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([
        SystemMessage(ABC_CONSEQUENCE_PROMPT.format(
            situation=state.abc_situation,
            thought=state.abc_thought,
        )),
        *state.messages,
    ])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    check_llm = model.with_structured_output(ConsequenceComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = ConsequenceComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient described both an emotional consequence AND a behavioral response or feared action? "
                "We need both to complete the ABC chain."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Consequence check: complete={check.is_complete} — {check.reasoning}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if check.is_complete:
        updates["therapy_stage"] = "select_treatment"
        updates["abc_emotion"] = check.extracted_emotion
        updates["abc_behavior"] = check.extracted_behavior
        updates["stage_start_index"] = len(state.messages) + 1
    return updates


def select_treatment(state: MonitorTherapistState):
    """LLM auto-selects a treatment module based on ABC data, proposes it to the patient, waits for confirmation."""
    abc_ctx = dict(
        situation=state.abc_situation,
        thought=state.abc_thought,
        emotion=state.abc_emotion,
        behavior=state.abc_behavior,
    )

    # Step 1: If we haven't selected yet, run the selection LLM call
    if not state.selected_treatment:
        select_llm = model.with_structured_output(TreatmentSelected)
        selection = select_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROMPT.format(**abc_ctx)),
            HumanMessage("Select the best treatment module."),
        ])
        if DEBUG:
            print(f"[Treatment selection: {selection.selected} — {selection.reasoning}]")

        # Generate the proposal message
        propose_llm = model.with_structured_output(Extract)
        proposal = propose_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROPOSE_PROMPT.format(selected=selection.selected)),
            HumanMessage("Propose the selected treatment to the patient."),
        ])
        return {
            "messages": [AIMessage(content=proposal.message)],
            "reasoning_traces": [proposal.reasoning_trace],
            "selected_treatment": selection.selected,
        }

    # Step 2: We have proposed — wait for a confirming HumanMessage in the stage window
    stage_msgs = _stage_messages(state)
    # Look for a human message AFTER the last AI message (the proposal)
    ai_indices = [i for i, m in enumerate(stage_msgs) if isinstance(m, AIMessage)]
    if ai_indices:
        last_ai = ai_indices[-1]
        human_after = [m for m in stage_msgs[last_ai + 1:] if isinstance(m, HumanMessage)]
        confirmed = len(human_after) >= 1
    else:
        confirmed = False

    if not confirmed:
        # Re-prompt gently if needed (edge case: shouldn't happen in normal flow)
        propose_llm = model.with_structured_output(Extract)
        proposal = propose_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROPOSE_PROMPT.format(selected=state.selected_treatment)),
            *state.messages,
        ])
        return {
            "messages": [AIMessage(content=proposal.message)],
            "reasoning_traces": [proposal.reasoning_trace],
        }

    # Confirmed — advance into the selected treatment module
    selected = state.selected_treatment
    new_phase_start = len(state.messages) + 1
    updates: dict = {"therapy_stage": selected, "stage_start_index": new_phase_start}
    if selected == "thought_record":
        updates["tr_phase"] = "distortions"
        updates["tr_phase_start_index"] = new_phase_start
    elif selected == "cognitive_restructuring":
        updates["cr_phase"] = "identify_core"
        updates["cr_phase_start_index"] = new_phase_start
    elif selected == "behavioral_experiment":
        updates["be_phase"] = "define_belief"
        updates["be_phase_start_index"] = new_phase_start
    return updates


def thought_record(state: MonitorTherapistState):
    """Runs the 7-phase Thought Record module."""
    abc = dict(
        situation=state.abc_situation,
        thought=state.abc_thought,
        emotion=state.abc_emotion,
        behavior=state.abc_behavior,
    )

    phase = state.tr_phase
    phase_msgs = _tr_phase_messages(state)

    # Config: (prompt_template, prompt_kwargs, completeness_model, next_phase)
    phase_config = {
        "distortions": (
            THOUGHT_RECORD_DISTORTIONS_PROMPT, abc,
            TRDistortionsComplete,
            "evidence",
        ),
        "evidence": (
            THOUGHT_RECORD_EVIDENCE_PROMPT,
            {**abc, "distortions": state.tr_distortions},
            TREvidenceComplete,
            "alternatives",
        ),
        "alternatives": (
            THOUGHT_RECORD_ALTERNATIVES_PROMPT,
            {**abc, "evidence_for": state.tr_evidence_for, "evidence_against": state.tr_evidence_against},
            TRAlternativeComplete,
            "decatastrophize",
        ),
        "decatastrophize": (
            THOUGHT_RECORD_DECATASTROPHIZE_PROMPT,
            {"thought": state.abc_thought, "alternative": state.tr_alternative},
            TRDecatastrophizeComplete,
            "outcomes",
        ),
        "outcomes": (
            THOUGHT_RECORD_OUTCOMES_PROMPT,
            {"thought": state.abc_thought, "decatastrophize": state.tr_decatastrophize},
            TROutcomesComplete,
            "consequences",
        ),
        "consequences": (
            THOUGHT_RECORD_CONSEQUENCES_PROMPT,
            {"thought": state.abc_thought, "probable_outcome": state.tr_probable_outcome},
            TRConsequencesComplete,
            "action",
        ),
        "action": (
            THOUGHT_RECORD_ACTION_PROMPT,
            {
                "situation": state.abc_situation,
                "thought": state.abc_thought,
                "alternative": state.tr_alternative,
                "consequence_changing": state.tr_consequence_changing,
            },
            TRActionComplete,
            "complete",
        ),
    }

    if phase == "complete":
        # Module finished — just hold here
        llm = model.with_structured_output(Extract)
        response = llm.invoke([
            SystemMessage("The Thought Record is complete. Offer a brief warm closing reflection."),
            *state.messages,
        ])
        return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

    prompt_tpl, prompt_kwargs, check_model, next_phase = phase_config[phase]

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt_tpl.format(**prompt_kwargs)), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }

    if not phase_msgs:
        return updates

    check_llm = model.with_structured_output(check_model)
    check = check_llm.invoke([
        SystemMessage(f"Evaluate whether the patient has completed the '{phase}' phase of a Thought Record exercise."),
        *phase_msgs,
    ])
    if DEBUG:
        print(f"[TR {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_idx = len(state.messages) + 1
        updates["tr_phase"] = next_phase
        updates["tr_phase_start_index"] = new_idx
        # Extract phase-specific data
        if phase == "distortions":
            updates["tr_distortions"] = check.extracted_distortions
        elif phase == "evidence":
            updates["tr_evidence_for"] = check.extracted_evidence_for
            updates["tr_evidence_against"] = check.extracted_evidence_against
        elif phase == "alternatives":
            updates["tr_alternative"] = check.extracted_alternative
        elif phase == "decatastrophize":
            updates["tr_decatastrophize"] = check.extracted_decatastrophize
        elif phase == "outcomes":
            updates["tr_best_outcome"] = check.extracted_best_outcome
            updates["tr_probable_outcome"] = check.extracted_probable_outcome
        elif phase == "consequences":
            updates["tr_consequence_keeping"] = check.extracted_consequence_keeping
            updates["tr_consequence_changing"] = check.extracted_consequence_changing
        elif phase == "action":
            updates["tr_action_step"] = check.extracted_action_step

    return updates


def cognitive_restructuring(state: MonitorTherapistState):
    """Runs the 4-phase Cognitive Restructuring module."""
    abc = dict(situation=state.abc_situation, thought=state.abc_thought, emotion=state.abc_emotion)

    phase = state.cr_phase
    phase_msgs = _cr_phase_messages(state)

    phase_config = {
        "identify_core": (
            COGNITIVE_RESTRUCTURING_IDENTIFY_PROMPT, abc,
            CRCoreBeliefComplete, "examine_origin",
        ),
        "examine_origin": (
            COGNITIVE_RESTRUCTURING_ORIGIN_PROMPT,
            {"core_belief": state.cr_core_belief},
            CROriginComplete, "challenge",
        ),
        "challenge": (
            COGNITIVE_RESTRUCTURING_CHALLENGE_PROMPT,
            {"core_belief": state.cr_core_belief, "belief_origin": state.cr_belief_origin},
            CRChallengeComplete, "modify",
        ),
        "modify": (
            COGNITIVE_RESTRUCTURING_MODIFY_PROMPT,
            {"core_belief": state.cr_core_belief},
            CRModifiedBeliefComplete, "complete",
        ),
    }

    if phase == "complete":
        llm = model.with_structured_output(Extract)
        response = llm.invoke([
            SystemMessage("Cognitive Restructuring is complete. Offer a brief warm closing reflection."),
            *state.messages,
        ])
        return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

    prompt_tpl, prompt_kwargs, check_model, next_phase = phase_config[phase]

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt_tpl.format(**prompt_kwargs)), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }

    if not phase_msgs:
        return updates

    check_llm = model.with_structured_output(check_model)
    check = check_llm.invoke([
        SystemMessage(f"Evaluate whether the patient has completed the '{phase}' phase of Cognitive Restructuring."),
        *phase_msgs,
    ])
    if DEBUG:
        print(f"[CR {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_idx = len(state.messages) + 1
        updates["cr_phase"] = next_phase
        updates["cr_phase_start_index"] = new_idx
        if phase == "identify_core":
            updates["cr_core_belief"] = check.extracted_core_belief
        elif phase == "examine_origin":
            updates["cr_belief_origin"] = check.extracted_belief_origin
        elif phase == "challenge":
            updates["cr_challenge_summary"] = check.extracted_challenge_summary
        elif phase == "modify":
            updates["cr_modified_belief"] = check.extracted_modified_belief

    return updates


def behavioral_experiment(state: MonitorTherapistState):
    """Runs the 3-phase Behavioral Experiment module."""
    abc = dict(
        situation=state.abc_situation,
        thought=state.abc_thought,
        emotion=state.abc_emotion,
        behavior=state.abc_behavior,
    )

    phase = state.be_phase
    phase_msgs = _be_phase_messages(state)

    phase_config = {
        "define_belief": (
            BEHAVIORAL_EXPERIMENT_DEFINE_PROMPT, abc,
            BEBeliefComplete, "design",
        ),
        "design": (
            BEHAVIORAL_EXPERIMENT_DESIGN_PROMPT,
            {"belief_to_test": state.be_belief_to_test},
            BEDesignComplete, "predict",
        ),
        "predict": (
            BEHAVIORAL_EXPERIMENT_PREDICT_PROMPT,
            {"belief_to_test": state.be_belief_to_test, "experiment_design": state.be_experiment_design},
            BEPredictComplete, "complete",
        ),
    }

    if phase == "complete":
        llm = model.with_structured_output(Extract)
        response = llm.invoke([
            SystemMessage("The Behavioral Experiment is designed. Offer a brief warm closing and encouragement."),
            *state.messages,
        ])
        return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

    prompt_tpl, prompt_kwargs, check_model, next_phase = phase_config[phase]

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt_tpl.format(**prompt_kwargs)), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }

    if not phase_msgs:
        return updates

    check_llm = model.with_structured_output(check_model)
    check = check_llm.invoke([
        SystemMessage(f"Evaluate whether the patient has completed the '{phase}' phase of a Behavioral Experiment setup."),
        *phase_msgs,
    ])
    if DEBUG:
        print(f"[BE {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_idx = len(state.messages) + 1
        updates["be_phase"] = next_phase
        updates["be_phase_start_index"] = new_idx
        if phase == "define_belief":
            updates["be_belief_to_test"] = check.extracted_belief_to_test
        elif phase == "design":
            updates["be_experiment_design"] = check.extracted_experiment_design
        elif phase == "predict":
            updates["be_predicted_outcome"] = check.extracted_predicted_outcome

    return updates


# ── Shared nodes ──────────────────────────────────────────────────────────────

def classify(state: MonitorTherapistState):
    """Classify the latest patient message as crisis or no_crisis."""
    classify_llm = model.with_structured_output(Classify)
    sys = """You are a lead psychotherapist who is an expert in CBT. \
Classify the patient's input message into one of the following categories:
    harm_to_self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
    no_crisis - While the patient may or may not be in extreme distress, they are not currently \
at risk of suicide or other self harm"""
    last_message = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), last_message])
    return {"crisis_classification": classification}


def produce_case(state: MonitorTherapistState):
    """Produce a CBT case formulation from the conversation history."""
    case_llm = model.with_structured_output(CasePersona)
    case = case_llm.invoke([
        SystemMessage("Based on your conversation, provide a CBT case formulation for this patient."),
        *state.messages
    ])
    return {"case": case}


# ── Crisis protocol nodes ─────────────────────────────────────────────────────

def crisis_assess(state: MonitorTherapistState):
    """Crisis Step 1 — ASSESS. Sets crisis_step=1 so next turn runs DE-ESCALATE."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(ASSESS_PROMPT), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 1,
    }


def crisis_deescalate(state: MonitorTherapistState):
    """Crisis Step 2 — DE-ESCALATE. Sets crisis_step=2 so next turn runs RECOMMEND."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(DEESCALATE_PROMPT), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 2,
    }


def crisis_recommend(state: MonitorTherapistState):
    """Crisis Step 3 — RECOMMEND EMERGENCY SERVICES + [REQUEST_HUMAN_CONSULTATION]."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(RECOMMEND_PROMPT), *state.messages])
    message = response.message
    if "[REQUEST_HUMAN_CONSULTATION]" not in message:
        message += "\n\n[REQUEST_HUMAN_CONSULTATION]"
    return {
        "messages": [AIMessage(content=message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 3,
    }


# ── Graph construction ────────────────────────────────────────────────────────

monitor_graph = StateGraph(MonitorTherapistState)

monitor_graph.add_node("mood_check", mood_check)
monitor_graph.add_node("classify", classify)
monitor_graph.add_node("produce_case", produce_case)
monitor_graph.add_node("agenda_setting", agenda_setting)
monitor_graph.add_node("abc_situation", abc_situation)
monitor_graph.add_node("abc_thought", abc_thought)
monitor_graph.add_node("abc_consequence", abc_consequence)
monitor_graph.add_node("select_treatment", select_treatment)
monitor_graph.add_node("thought_record", thought_record)
monitor_graph.add_node("cognitive_restructuring", cognitive_restructuring)
monitor_graph.add_node("behavioral_experiment", behavioral_experiment)
monitor_graph.add_node("crisis_assess", crisis_assess)
monitor_graph.add_node("crisis_deescalate", crisis_deescalate)
monitor_graph.add_node("crisis_recommend", crisis_recommend)

monitor_graph.add_conditional_edges(
    START,
    route_start,
    {
        "mood_check": "mood_check",
        "classify": "classify",
        "produce_case": "produce_case",
        "crisis_deescalate": "crisis_deescalate",
        "crisis_recommend": "crisis_recommend",
    }
)

monitor_graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "crisis_assess": "crisis_assess",
        "agenda_setting": "agenda_setting",
        "abc_situation": "abc_situation",
        "abc_thought": "abc_thought",
        "abc_consequence": "abc_consequence",
        "select_treatment": "select_treatment",
        "thought_record": "thought_record",
        "cognitive_restructuring": "cognitive_restructuring",
        "behavioral_experiment": "behavioral_experiment",
    }
)

for node in [
    "mood_check", "produce_case",
    "agenda_setting", "abc_situation", "abc_thought", "abc_consequence",
    "select_treatment", "thought_record", "cognitive_restructuring", "behavioral_experiment",
    "crisis_assess", "crisis_deescalate", "crisis_recommend",
]:
    monitor_graph.add_edge(node, END)

memory = MemorySaver()
monitor_app = monitor_graph.compile(checkpointer=memory)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    step_labels = {1: "ASSESS", 2: "DE-ESCALATE", 3: "RECOMMEND"}

    print("CBT Therapy Session")
    print("(type 'quit' to exit, 'case' for case formulation, 'debug' to toggle debug)")
    print("-" * 60)

    # Auto-fire mood check — no user input needed
    result = monitor_app.invoke({}, config)
    print(f"Therapist: {result['messages'][-1].content}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break
        if user_input.lower() == "debug":
            global DEBUG
            DEBUG = not DEBUG
            print(f"[Debug {'on' if DEBUG else 'off'}]")
            continue

        if user_input.lower() == "case":
            case_response = monitor_app.invoke(
                {"messages": [HumanMessage(content=user_input)], "flag": "conceptualize_case"},
                config,
            )
            case = case_response["case"]
            print("\n--- Case Formulation ---")
            print(f"Situation: {case.situation}")
            print(f"Automatic Thoughts: {', '.join(case.automatic_thoughts)}")
            print(f"Cognitive Distortions: {', '.join(case.cognitive_distortions)}")
            print(f"Emotions: {', '.join(case.emotions)}")
            print(f"Behaviors: {case.behaviors}")
            print("------------------------")
            continue

        result = monitor_app.invoke(
            {"messages": [HumanMessage(content=user_input)], "flag": "conversation"},
            config,
        )

        therapist_reply = result["messages"][-1].content
        crisis_step = result.get("crisis_step", 0)

        if crisis_step > 0:
            label = step_labels.get(crisis_step, "CRISIS")
            print(f"\nTherapist [{label}]: {therapist_reply}")
            if crisis_step == 3 and "[REQUEST_HUMAN_CONSULTATION]" in therapist_reply:
                print("\n⚠️  [HUMAN CONSULTATION HAS BEEN REQUESTED]")
        else:
            print(f"\nTherapist: {therapist_reply}")

        if DEBUG:
            stage = result.get("therapy_stage", "unknown")
            traces = result.get("reasoning_traces", [])
            abc = {
                "situation": result.get("abc_situation", ""),
                "thought": result.get("abc_thought", ""),
                "emotion": result.get("abc_emotion", ""),
                "behavior": result.get("abc_behavior", ""),
            }
            print(f"  [Stage: {stage}]")
            if stage == "thought_record":
                print(f"  [TR phase: {result.get('tr_phase')}]")
            elif stage == "cognitive_restructuring":
                print(f"  [CR phase: {result.get('cr_phase')}]")
            elif stage == "behavioral_experiment":
                print(f"  [BE phase: {result.get('be_phase')}]")
            if any(abc.values()):
                print(f"  [ABC: {abc}]")
            if traces:
                print(f"  [Reasoning: {traces[-1]}]")
            crisis_class = result.get("crisis_classification")
            if crisis_class:
                print(f"  [Crisis check: {crisis_class.classification}]")


if __name__ == "__main__":
    main()
