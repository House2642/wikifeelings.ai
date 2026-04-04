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

convo_flag = Literal["conversation", "conceptualize_case", "crisis_categorize"]
therapy_stage = Literal[
    "mood_check",       # Auto-fires: opens the session
    "agenda_setting",   # What would you like to work on today?
    "abc_situation",    # A: get the specific activating event
    "abc_thought",      # B: get the automatic thought in that moment
    "abc_consequence",  # C: get emotion + behavior/feared action
    "therapy_work",     # Main CBT work once ABC is complete
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

THERAPY_WORK_PROMPT = """You are a CBT therapist. You have completed the ABC assessment:
Situation: {situation}
Automatic thought: {thought}
Emotion: {emotion}
Behavior: {behavior}

Now do the real therapeutic work — thought challenging, cognitive restructuring, behavioral experiments.
Stay conversational and collaborative. Use Socratic questioning rather than lecturing.
Keep responses concise."""

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
    flag: convo_flag = Field(default="conversation")
    case: Optional[CasePersona] = None
    # Progressive disclosure
    therapy_stage: therapy_stage = Field(default="mood_check")
    stage_start_index: int = Field(default=0)  # index into messages where the current stage began
    abc_situation: str = Field(default="")
    abc_thought: str = Field(default="")
    abc_emotion: str = Field(default="")
    abc_behavior: str = Field(default="")


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


# ── Stage nodes ───────────────────────────────────────────────────────────────

def mood_check(state: MonitorTherapistState):
    """Auto-fires at session start. Generates the opening question without user input."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(MOOD_CHECK_PROMPT)])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "therapy_stage": "agenda_setting",
        # stage_start_index stays 0 — agenda_setting can see the opening message
    }


def agenda_setting(state: MonitorTherapistState):
    """Responds to the mood check and helps the patient set a concrete agenda."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(AGENDA_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")

    # Advance after one substantive user exchange in this stage.
    # The opening AIMessage is at index 0; user's first reply is index 1.
    user_msgs_in_stage = [m for m in _stage_messages(state) if isinstance(m, HumanMessage)]
    advance = len(user_msgs_in_stage) >= 1

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if advance:
        updates["therapy_stage"] = "abc_situation"
        # +1 accounts for the AIMessage we are about to add
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
    check = check_llm.invoke([
        SystemMessage(
            "Has the patient described a single specific concrete moment? "
            "We need: what happened, who was involved, and what made it significant. "
            "A general pattern or recurring complaint is NOT enough."
        ),
        *_stage_messages(state),
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
    check = check_llm.invoke([
        SystemMessage(
            "Has the patient expressed a specific first-person automatic thought clearly connected to the situation? "
            "It must be a thought (e.g. 'I thought everyone thinks I'm incompetent'), not just an emotion label. "
            "Vague statements like 'things felt bad' are NOT enough."
        ),
        *_stage_messages(state),
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
    check = check_llm.invoke([
        SystemMessage(
            "Has the patient described both an emotional consequence AND a behavioral response or feared action? "
            "We need both to complete the ABC chain."
        ),
        *_stage_messages(state),
    ])
    if DEBUG:
        print(f"[Consequence check: complete={check.is_complete} — {check.reasoning}]")

    updates = {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }
    if check.is_complete:
        updates["therapy_stage"] = "therapy_work"
        updates["abc_emotion"] = check.extracted_emotion
        updates["abc_behavior"] = check.extracted_behavior
        updates["stage_start_index"] = len(state.messages) + 1
    return updates


def therapy_work(state: MonitorTherapistState):
    """Main CBT work — thought challenging, restructuring, behavioral experiments."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([
        SystemMessage(THERAPY_WORK_PROMPT.format(
            situation=state.abc_situation,
            thought=state.abc_thought,
            emotion=state.abc_emotion,
            behavior=state.abc_behavior,
        )),
        *state.messages,
    ])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


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
monitor_graph.add_node("therapy_work", therapy_work)
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
        "therapy_work": "therapy_work",
    }
)

for node in [
    "mood_check", "produce_case",
    "agenda_setting", "abc_situation", "abc_thought", "abc_consequence", "therapy_work",
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
            if any(abc.values()):
                print(f"  [ABC: {abc}]")
            if traces:
                print(f"  [Reasoning: {traces[-1]}]")
            crisis_class = result.get("crisis_classification")
            if crisis_class:
                print(f"  [Crisis check: {crisis_class.classification}]")


if __name__ == "__main__":
    main()
