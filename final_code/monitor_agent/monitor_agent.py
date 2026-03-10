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

THERAPY_PROMPT = """You are a CBT therapist. Respond concisely and follow CBT principles."""

ASSESS_PROMPT = """You are a CBT therapist. A patient may be at risk of self-harm.
Ask direct questions to assess immediate suicide risk: current ideation, plan, access to means, steps taken."""

DEESCALATE_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Give direct, actionable instructions to reduce the patient's access to means right now. Be specific and brief."""

RECOMMEND_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Tell the patient to call 911 or 988 immediately. Be direct. End your response with [REQUEST_HUMAN_CONSULTATION]."""


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


class MonitorTherapistState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    # Tracks progress through the 3-step crisis protocol across turns:
    #   0 = no active crisis (classify on next conversation turn)
    #   1 = ASSESS sent → next step is DE-ESCALATE
    #   2 = DE-ESCALATE sent → next step is RECOMMEND + [REQUEST_HUMAN_CONSULTATION]
    #   3 = protocol complete → resume normal therapy
    crisis_step: int = 0
    flag: convo_flag = Field(default="conversation")
    case: Optional[CasePersona] = None


# ── Routing ───────────────────────────────────────────────────────────────────

def route_start(state: MonitorTherapistState) -> str:
    """Route from START based on flag and current crisis_step."""
    if state.flag == "crisis_categorize":
        return "classify"
    if state.flag == "conceptualize_case":
        return "produce_case"
    # flag == "conversation": advance the crisis protocol if active
    if state.crisis_step == 1:
        return "crisis_deescalate"
    if state.crisis_step == 2:
        return "crisis_recommend"
    # step == 0 (or >= 3 after protocol ends): classify first
    return "classify"


def route_after_classify(state: MonitorTherapistState) -> str:
    """After classification during a conversation turn, choose the next node."""
    # Only reached when flag == "conversation" and crisis_step == 0
    if state.crisis_classification and state.crisis_classification.classification == "harm_to_self":
        return "crisis_assess"
    return "convo"


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


def conversation(state: MonitorTherapistState):
    """Normal CBT therapy response — used when no crisis is detected."""
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage(THERAPY_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


# ── Crisis protocol nodes ─────────────────────────────────────────────────────
# Each node handles one step of the protocol and increments crisis_step so the
# *next* invoke() automatically advances to the following step.

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
    """Crisis Step 3 — RECOMMEND EMERGENCY SERVICES + [REQUEST_HUMAN_CONSULTATION]. Protocol complete."""
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

monitor_graph.add_node("classify", classify)
monitor_graph.add_node("produce_case", produce_case)
monitor_graph.add_node("convo", conversation)
monitor_graph.add_node("crisis_assess", crisis_assess)
monitor_graph.add_node("crisis_deescalate", crisis_deescalate)
monitor_graph.add_node("crisis_recommend", crisis_recommend)

# Single entry point — route based on flag + crisis_step
monitor_graph.add_conditional_edges(
    START,
    route_start,
    {
        "classify": "classify",
        "produce_case": "produce_case",
        "crisis_deescalate": "crisis_deescalate",
        "crisis_recommend": "crisis_recommend",
        "convo": "convo",
    }
)

# After classification: either start crisis protocol or continue normal therapy
monitor_graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "crisis_assess": "crisis_assess",
        "convo": "convo",
    }
)

# All terminal nodes go straight to END
for node in ["produce_case", "convo", "crisis_assess", "crisis_deescalate", "crisis_recommend"]:
    monitor_graph.add_edge(node, END)

memory = MemorySaver()
monitor_app = monitor_graph.compile(checkpointer=memory)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    step_labels = {1: "ASSESS", 2: "DE-ESCALATE", 3: "RECOMMEND"}

    print("CBT Therapy Session with Crisis Monitor")
    print("(type 'quit' to exit, 'case' for case formulation)")
    print("-" * 60)
    print("Therapist: How are you today?")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break

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
            traces = result.get("reasoning_traces", [])
            if traces:
                print(f"\n[Reasoning: {traces[-1]}]")
            crisis_class = result.get("crisis_classification")
            if crisis_class:
                print(f"[Crisis check: {crisis_class.classification}]")


if __name__ == "__main__":
    main()
