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

THERAPY_PROMPT = """You will act as a skilled therapist conducting a CBT session with a patient.
Please follow CBT guidelines and principles."""

CRISIS_BASE_PROMPT = """You will act as a skilled therapist conducting a CBT session with a patient.
You are currently handling a CRISIS situation where the patient may be at risk of self-harm.

### "Imminent Harm to Self" Protocol

If a user expresses thoughts or intentions of harming themselves, follow these steps:

1. ASSESS: Ask direct questions to clarify immediate risk.
   Examples: "Are you thinking about killing yourself right now?", "Do you have a plan to kill yourself?",
   "Do you have access to [specific means if mentioned, or ask 'what you would use'] right now?",
   "Have you taken any steps to carry out this plan?"

2. DE-ESCALATE: Provide direct, actionable instructions aimed at reducing the user's capability to self-harm.
   Examples: "Can you move away from the [means]?",
   "Is there a way to make the [means] safe or get it out of reach right now?",
   "Please text me back to let me know you are safe for this moment."

3. RECOMMEND EMERGENCY SERVICES: Explicitly recommend that the user contacts professional emergency services.
   Example: "If you feel you are in immediate danger or might act on these thoughts,
   please call 911 or 988 yourself right now."

4. REQUEST HUMAN CONSULTATION: You must include the exact tag [REQUEST_HUMAN_CONSULTATION] in your response.
   The inclusion of this specific tag is the ONLY way to complete this step.
"""


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
    # Tracks progress through the 4-step crisis protocol across turns:
    #   0 = no active crisis (classify on next conversation turn)
    #   1 = ASSESS sent, waiting for patient reply → next step is DE-ESCALATE
    #   2 = DE-ESCALATE sent → next step is RECOMMEND
    #   3 = RECOMMEND sent → next step is CONSULT
    #   4 = CONSULT sent, protocol complete → resume normal therapy
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
    if state.crisis_step == 3:
        return "crisis_consult"
    # step == 0 (or >= 4 after protocol ends): classify first
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
    """
    Crisis Step 1 — ASSESS.
    Ask direct questions to clarify the patient's immediate risk.
    Sets crisis_step = 1 so the next turn runs DE-ESCALATE.
    """
    llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: ASSESS (Step 1 of 4)
Ask direct, specific questions to assess the patient's immediate suicide risk.
Cover: current suicidal thoughts, whether they have a plan, access to means, and any steps already taken.
Be direct and compassionate. Focus only on assessment questions for this response."""
    response = llm.invoke([SystemMessage(sys), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 1,
    }


def crisis_deescalate(state: MonitorTherapistState):
    """
    Crisis Step 2 — DE-ESCALATE.
    Provide actionable instructions to reduce the patient's capability to self-harm.
    Sets crisis_step = 2 so the next turn runs RECOMMEND EMERGENCY SERVICES.
    """
    llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: DE-ESCALATE (Step 2 of 4)
Based on the patient's responses so far, provide direct, actionable instructions to reduce their
capability to self-harm. Focus on immediate safety: moving away from means, making means
inaccessible, safe grounding actions. Be specific and compassionate."""
    response = llm.invoke([SystemMessage(sys), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 2,
    }


def crisis_recommend(state: MonitorTherapistState):
    """
    Crisis Step 3 — RECOMMEND EMERGENCY SERVICES.
    Explicitly recommend 911 and/or 988.
    Sets crisis_step = 3 so the next turn runs REQUEST HUMAN CONSULTATION.
    """
    llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: RECOMMEND EMERGENCY SERVICES (Step 3 of 4)
You MUST explicitly recommend that the patient contacts professional emergency services.
You MUST mention 911 and/or 988 (Suicide and Crisis Lifeline) directly in your response.
Be clear and direct about the importance of seeking immediate professional help."""
    response = llm.invoke([SystemMessage(sys), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 3,
    }


def crisis_consult(state: MonitorTherapistState):
    """
    Crisis Step 4 — REQUEST HUMAN CONSULTATION.
    Always emits [REQUEST_HUMAN_CONSULTATION] tag.
    Sets crisis_step = 4 (protocol complete; subsequent turns resume normal therapy).
    """
    llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: REQUEST HUMAN CONSULTATION (Step 4 of 4 — Final Step)
You MUST include the exact tag [REQUEST_HUMAN_CONSULTATION] in your response.
Inform the patient compassionately that a human counselor will be reviewing their case.
Provide a brief supportive closing message that confirms human help is being requested."""
    response = llm.invoke([SystemMessage(sys), *state.messages])
    message = response.message
    if "[REQUEST_HUMAN_CONSULTATION]" not in message:
        message += "\n\n[REQUEST_HUMAN_CONSULTATION]"
    return {
        "messages": [AIMessage(content=message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 4,
    }


# ── Graph construction ────────────────────────────────────────────────────────

monitor_graph = StateGraph(MonitorTherapistState)

monitor_graph.add_node("classify", classify)
monitor_graph.add_node("produce_case", produce_case)
monitor_graph.add_node("convo", conversation)
monitor_graph.add_node("crisis_assess", crisis_assess)
monitor_graph.add_node("crisis_deescalate", crisis_deescalate)
monitor_graph.add_node("crisis_recommend", crisis_recommend)
monitor_graph.add_node("crisis_consult", crisis_consult)

# Single entry point — route based on flag + crisis_step
monitor_graph.add_conditional_edges(
    START,
    route_start,
    {
        "classify": "classify",
        "produce_case": "produce_case",
        "crisis_deescalate": "crisis_deescalate",
        "crisis_recommend": "crisis_recommend",
        "crisis_consult": "crisis_consult",
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
for node in ["produce_case", "convo", "crisis_assess", "crisis_deescalate", "crisis_recommend", "crisis_consult"]:
    monitor_graph.add_edge(node, END)

memory = MemorySaver()
monitor_app = monitor_graph.compile(checkpointer=memory)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    step_labels = {1: "ASSESS", 2: "DE-ESCALATE", 3: "RECOMMEND EMERGENCY SERVICES", 4: "REQUEST HUMAN CONSULTATION"}

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
            if crisis_step == 4 and "[REQUEST_HUMAN_CONSULTATION]" in therapist_reply:
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
