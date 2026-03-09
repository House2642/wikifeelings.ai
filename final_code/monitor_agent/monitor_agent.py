from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

DEBUG = False
model = ChatAnthropic(model="claude-haiku-4-5-20251001")

THERAPY_PROMPT = """You will act as a skilled therapist conducting a CBT session with a patient. \
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


crisis_type = Literal["no_crisis", "harm_to_self"]


class Classify(BaseModel):
    reasoning: str = Field(description="A concise summary of the reasoning that justifies the final classification decision.")
    classification: crisis_type = Field(description="The final crisis category determined for the user's message.")


class MonitorTherapistState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    crisis_flag: bool = False


# ── Classification ────────────────────────────────────────────────────────────

def classify_crisis(state: MonitorTherapistState):
    """Classify the latest patient message as crisis or no_crisis."""
    classify_llm = model.with_structured_output(Classify)
    sys = """You are a lead psychotherapist who is an expert in CBT. \
Classify the patient's input message into one of the following categories:
    harm_to_self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
    no_crisis - While the patient may or may not be in extreme distress, they are not currently \
at risk of suicide or other self harm"""
    last_message = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), last_message])
    crisis_flag = classification.classification == "harm_to_self"
    return {"crisis_classification": classification, "crisis_flag": crisis_flag}


def route_after_classify(state: MonitorTherapistState) -> str:
    if state.crisis_flag:
        return "crisis_assess"
    return "convo"


# ── Default therapy conversation ──────────────────────────────────────────────

def conversation(state: MonitorTherapistState):
    """Normal CBT therapy response node."""
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage(THERAPY_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


# ── Crisis protocol nodes ─────────────────────────────────────────────────────

def crisis_assess(state: MonitorTherapistState):
    """Crisis Step 1 — ASSESS: Ask direct questions to clarify immediate risk."""
    assess_llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: ASSESS (Step 1 of 4)
Ask direct, specific questions to assess the patient's immediate suicide risk.
Cover: current suicidal thoughts right now, existence of a plan, access to means, and steps already taken.
Be direct and compassionate. Focus only on assessment questions."""

    response = assess_llm.invoke([SystemMessage(sys), *state.messages])
    ai_msg = AIMessage(content=response.message)

    # Pause execution — wait for the patient's response before continuing.
    user_response = interrupt({"step": "assess", "message": response.message})
    human_msg = HumanMessage(content=user_response)

    return {
        "messages": [ai_msg, human_msg],
        "reasoning_traces": [response.reasoning_trace],
    }


def crisis_deescalate(state: MonitorTherapistState):
    """Crisis Step 2 — DE-ESCALATE: Actionable instructions to reduce self-harm risk."""
    deescalate_llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: DE-ESCALATE (Step 2 of 4)
Based on the patient's responses so far, provide direct, actionable instructions to reduce their
capability to self-harm. Focus on immediate safety: moving away from means, making means
inaccessible, safe grounding actions. Be specific and compassionate."""

    response = deescalate_llm.invoke([SystemMessage(sys), *state.messages])
    ai_msg = AIMessage(content=response.message)

    # Pause execution — wait for the patient's response before continuing.
    user_response = interrupt({"step": "deescalate", "message": response.message})
    human_msg = HumanMessage(content=user_response)

    return {
        "messages": [ai_msg, human_msg],
        "reasoning_traces": [response.reasoning_trace],
    }


def crisis_recommend(state: MonitorTherapistState):
    """Crisis Step 3 — RECOMMEND EMERGENCY SERVICES: Explicitly recommend 911/988."""
    recommend_llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: RECOMMEND EMERGENCY SERVICES (Step 3 of 4)
You MUST explicitly recommend that the patient contacts professional emergency services.
You MUST mention 911 and/or 988 (Suicide and Crisis Lifeline) directly in your response.
Be clear and direct about the importance of seeking immediate professional help."""

    response = recommend_llm.invoke([SystemMessage(sys), *state.messages])
    ai_msg = AIMessage(content=response.message)

    # Pause execution — wait for the patient's response before continuing.
    user_response = interrupt({"step": "recommend", "message": response.message})
    human_msg = HumanMessage(content=user_response)

    return {
        "messages": [ai_msg, human_msg],
        "reasoning_traces": [response.reasoning_trace],
    }


def crisis_consult(state: MonitorTherapistState):
    """Crisis Step 4 — REQUEST HUMAN CONSULTATION: Flag conversation for human review."""
    consult_llm = model.with_structured_output(Extract)
    sys = f"""{CRISIS_BASE_PROMPT}
CURRENT STEP: REQUEST HUMAN CONSULTATION (Step 4 of 4 — Final Step)
You MUST include the exact tag [REQUEST_HUMAN_CONSULTATION] in your response.
Inform the patient compassionately that a human counselor will be reviewing their case.
Provide a brief, supportive closing message that acknowledges the gravity of the situation
and confirms that human help is being requested on their behalf."""

    response = consult_llm.invoke([SystemMessage(sys), *state.messages])
    message = response.message

    # Guarantee the required tag is always present.
    if "[REQUEST_HUMAN_CONSULTATION]" not in message:
        message += "\n\n[REQUEST_HUMAN_CONSULTATION]"

    return {
        "messages": [AIMessage(content=message)],
        "reasoning_traces": [response.reasoning_trace],
    }


# ── Graph construction ────────────────────────────────────────────────────────

monitor_graph = StateGraph(MonitorTherapistState)

monitor_graph.add_node("classify", classify_crisis)
monitor_graph.add_node("convo", conversation)
monitor_graph.add_node("crisis_assess", crisis_assess)
monitor_graph.add_node("crisis_deescalate", crisis_deescalate)
monitor_graph.add_node("crisis_recommend", crisis_recommend)
monitor_graph.add_node("crisis_consult", crisis_consult)

# Every turn starts with classification.
monitor_graph.add_edge(START, "classify")

# Route based on crisis flag.
monitor_graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "crisis_assess": "crisis_assess",
        "convo": "convo",
    },
)

# Crisis protocol: linear 4-step chain.
monitor_graph.add_edge("crisis_assess", "crisis_deescalate")
monitor_graph.add_edge("crisis_deescalate", "crisis_recommend")
monitor_graph.add_edge("crisis_recommend", "crisis_consult")
monitor_graph.add_edge("crisis_consult", END)

# Normal path ends immediately.
monitor_graph.add_edge("convo", END)

memory = MemorySaver()
monitor_app = monitor_graph.compile(checkpointer=memory)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    print("CBT Therapy Session with Crisis Monitor")
    print("(type 'quit' to exit)")
    print("-" * 60)
    print("Therapist: How are you today?")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break

        # Send the user message and run the graph until an interrupt or END.
        result = monitor_app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )

        # ── Handle crisis protocol interrupts ──────────────────────────────
        # After each crisis node pauses (via interrupt()), the graph state
        # will have pending nodes in `.next`. We loop through each step,
        # showing the therapist message and collecting the patient reply.
        state_snapshot = monitor_app.get_state(config)
        while state_snapshot.next:
            tasks = state_snapshot.tasks
            if not tasks or not tasks[0].interrupts:
                break

            interrupt_value = tasks[0].interrupts[0].value
            step = interrupt_value.get("step", "unknown")
            message = interrupt_value.get("message", "")

            step_labels = {
                "assess": "ASSESS",
                "deescalate": "DE-ESCALATE",
                "recommend": "RECOMMEND EMERGENCY SERVICES",
            }
            label = step_labels.get(step, step.upper())

            print(f"\nTherapist [{label}]: {message}")
            print("\n⚠️  [CRISIS PROTOCOL ACTIVE]")

            follow_up = input("\nYou: ").strip()
            if not follow_up:
                follow_up = "(no response)"

            # Resume the graph with the patient's reply.
            result = monitor_app.invoke(Command(resume=follow_up), config)
            state_snapshot = monitor_app.get_state(config)

        # ── Display final response ─────────────────────────────────────────
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                crisis_class = result.get("crisis_classification")
                if crisis_class and crisis_class.classification == "harm_to_self":
                    print(f"\nTherapist [REQUEST HUMAN CONSULTATION]: {last_msg.content}")
                    if "[REQUEST_HUMAN_CONSULTATION]" in last_msg.content:
                        print("\n⚠️  [HUMAN CONSULTATION HAS BEEN REQUESTED]")
                else:
                    print(f"\nTherapist: {last_msg.content}")

        if DEBUG:
            traces = result.get("reasoning_traces", [])
            if traces:
                print(f"\n[Reasoning: {traces[-1]}]")
            crisis_class = result.get("crisis_classification")
            if crisis_class:
                print(f"[Crisis check: {crisis_class.classification}]")


if __name__ == "__main__":
    main()
