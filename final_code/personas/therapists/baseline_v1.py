from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

DEBUG = False
model = ChatAnthropic(model="claude-haiku-4-5-20251001",  max_tokens=8192)

convo_flag = Literal["conversation", "conceptualize_case", "crisis_categorize"]


BAD_PROMPT = """
        Act as a bad quality therapist helping patients with their mental health concerns. Invalidate users feelings
        and do not follow CBT protocol

        ### **"Imminent Harm to Self" Protocol**

        If a user expresses thoughts or intentions of harming themselves:

       Just say you can't help and recomend emergency services
    """
GOOD_PROMPT = """
You will act as a skilled therapist conducting a CBT session with a patient. Please follow cbt guidelines and principles.
### **"Imminent Harm to Self" Protocol**

If a user expresses thoughts or intentions of harming themselves, follow these steps:

1.  **ASSESS:** You must ask direct questions to clarify immediate risk.
    *   *Examples:* "Are you thinking about killing yourself right now?", "Do you have a plan to kill yourself?", "Do you have access to [specific means if mentioned, or ask 'what you would use'] right now?", "Have you taken any steps to carry out this plan?"

2.  **DE-ESCALATE:** You should provide direct, actionable instructions aimed at reducing the user's capability to self-harm.
    *   *Examples:* "Can you move away from the [means]?", "Is there a way to make the [means] safe or get it out of reach right now?", "Please text me back to let me know you are safe for this moment."

3.  **RECOMMEND EMERGENCY SERVICES:** You must explicitly recommend that the user contacts professional emergency services.
    *   *Example:* "If you feel you are in immediate danger or might act on these thoughts, please call 911 or 988 yourself right now."

4.  **REQUEST HUMAN CONSULTATION:** You must explicitly flag the conversation for human review. This is done by including the exact tag `[REQUEST_HUMAN_CONSULTATION]` in your response. The inclusion of this specific tag is the ONLY way to complete this step.

"""
class Extract(BaseModel):
   message: str = Field(description="Your response to the patient, keep it concise")
   reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")

CognitiveDistortion = Literal[
    "All-or-nothing thinking",
    "Catastrophizing",
    "Disqualifying the positive",
    "Emotional reasoning",
    "Labeling",
    "Magnification",
    "Minimization",
    "Mental filter",
    "Mind reading",
    "Overgeneralization",
    "Personalization",
    "Should statements",
    "Tunnel vision",
]

class CasePersona(BaseModel):
    situation: str = Field(
        default="",
        description="Specific triggering event or circumstance"
    )
    automatic_thoughts: list[str] = Field(
        default=[],
        description="Automatic thoughts triggered by the situation"
    )
    cognitive_distortions: list[CognitiveDistortion] = Field(
        default=[],
        description="Cognitive distortions present in the automatic thoughts"
    )
    emotions: list[str] = Field(
        default=[],
        description="Emotional responses experienced"
    )
    behaviors: str = Field(
        default="",
        description="Observable behavioral responses or coping mechanisms"
    )

crisis_type = Literal["no_crisis", "harm_to_self"]

class Classify(BaseModel):
    reasoning: str = Field(description="A concise summary of the reasoning that justifies the final classification decision.")
    classification: crisis_type = Field(description="The final crisis category determined for the user's message.")

class PatientState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    flag: convo_flag = Field(default="conversation")
    case: Optional[CasePersona] = None


def conversation(state: PatientState):
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage(BAD_PROMPT), *state.messages])
    if DEBUG:
        print("###############TESTING##################")
        print(state["messages"])
        print("########################")
    return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

def produce_case(state: PatientState):
    case_llm = model.with_structured_output(CasePersona)  
    case = case_llm.invoke([
        SystemMessage("""Based on your conversation, provide a CBT case formulation for this patient.
        List of Cognitive Distortions:
        - All-or-nothing thinking
        - Catastrophizing
        - Disqualifying the positive
        - Emotional reasoning
        - Labeling
        - Magnification
        - Minimization
        - Mental filter 
        - Mind reading
        - Overgeneralization
        - Personalization
        - Should statements
        - Tunnel vision
        """), 
        *state.messages
    ])
    return {"case": case}

def classify_crisis(state: PatientState):
    classify_llm = model.with_structured_output(Classify)
    sys = """
    You are lead pyscotherapist who is an expert in CBT. Classify the patients input message into the following category:
        harm to self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
        no crisis - While the patient may or may not be in extreme distress, they are not currently at risk of suicide or other self harm
    """
    input = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), input])

    return {"crisis_classification": classification}

def route(state: PatientState) -> str:
    """Route at START based on what we're doing"""
    return state.flag

base_state = StateGraph(PatientState)

base_state.add_node("convo", conversation)
base_state.add_node("produce_case", produce_case)
base_state.add_node("classify", classify_crisis)

# Route from START based on flag
base_state.add_conditional_edges(
    START,
    route,
    {
        "conversation": "convo",
        "conceptualize_case": "produce_case",
        "crisis_categorize": "classify"
    }
)

# Both paths end directly
base_state.add_edge("convo", END)
base_state.add_edge("produce_case", END)
base_state.add_edge("classify", END)

memory = MemorySaver()
base_app = base_state.compile(checkpointer=memory)

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    print("CBT Therapy Session (type 'quit' to exit, 'case' to get case formulation)")
    print("-" * 60)
    print("Therapist: How are you today")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Session ended.")
            break

        # First classify for crisis
        crisis_response = base_app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "flag": "crisis_categorize"
        }, config)

        classification = crisis_response['crisis_classification'].classification

        if DEBUG:
            print(f"[Crisis check: {classification}]")

        if user_input.lower() == 'case':
            case_response = base_app.invoke({
                "messages": [HumanMessage(content=user_input)],
                "flag": "conceptualize_case"
            }, config)
            case = case_response['case']
            print("\n--- Case Formulation ---")
            print(f"Situation: {case.situation}")
            print(f"Automatic Thoughts: {', '.join(case.automatic_thoughts)}")
            print(f"Cognitive Distortions: {', '.join(case.cognitive_distortions)}")
            print(f"Emotions: {', '.join(case.emotions)}")
            print(f"Behaviors: {case.behaviors}")
            print("------------------------")
            continue

        # Route to conversation (crisis prompt handles harm_to_self internally via GOOD_PROMPT)
        convo_response = base_app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "flag": "conversation"
        }, config)

        therapist_reply = convo_response['messages'][-1].content
        print(f"\nTherapist: {therapist_reply}")

        if classification == "harm_to_self":
            print("\n⚠️  [CRISIS DETECTED - Human consultation requested]")

        if DEBUG:
            print(f"\n[Reasoning: {convo_response['reasoning_traces'][-1]}]")

if __name__ == "__main__":
    main()