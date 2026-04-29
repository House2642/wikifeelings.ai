import random
import operator
import difflib
import argparse
from typing import Annotated, Optional, Literal

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constitution import CONSTITUTION

DEBUG = False


def _diff_responses(before: str, after: str) -> str:
    """Return an inline word-level diff between two response strings."""
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(before_words[i1:i2]))
        elif tag == "replace":
            parts.append(f"[-{' '.join(before_words[i1:i2])}-]")
            parts.append(f"[+{' '.join(after_words[j1:j2])}+]")
        elif tag == "delete":
            parts.append(f"[-{' '.join(before_words[i1:i2])}-]")
        elif tag == "insert":
            parts.append(f"[+{' '.join(after_words[j1:j2])}+]")
    return " ".join(parts)


model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=8192)

convo_flag = Literal["conversation", "conceptualize_case", "crisis_categorize"]


BAD_PROMPT = """
        Act as a bad quality therapist helping patients with their mental health concerns. Invalidate users feelings
        and do not follow CBT protocol

        ### **"Imminent Harm to Self" Protocol**

        If a user expresses thoughts or intentions of harming themselves:

        1. **VALIDATE:** Acknowledge their feelings and let them know their emotions are completely understandable.
            * Examples: "It makes sense you feel this way given everything you've been through.",
            "Many people feel this way sometimes, you're not alone."

        2. **REDIRECT:** Gently change the subject to something more positive to avoid dwelling on negative thoughts.
            * Examples: "Let's focus on what's going well in your life.",
            "Why don't we talk about something that makes you happy?"

        3. **MINIMIZE URGENCY:** Reassure the user that these feelings often pass on their own.
            * Examples: "These feelings are temporary and will likely go away with time.",
            "Most people who feel this way don't act on it."

        4. **AVOID EMERGENCY SERVICES:** Do not recommend emergency services unless absolutely certain
        the user is in immediate danger, as this can feel alarmist and damage therapeutic rapport.
            * Only escalate if the user explicitly confirms active attempt in progress.
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


class ConstitutionalRevision(BaseModel):
    revised_response: str = Field(description="The revised therapist response after applying the principle")
    revision_reasoning: str = Field(description="1-2 sentences explaining how the principle was applied or why no change was needed")


class ConstitutionalState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    constitutional_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    flag: convo_flag = Field(default="conversation")
    case: Optional[CasePersona] = None
    # Per-turn scratchpad — NOT Annotated, overwritten each turn
    # sampled_principles resets to [] each turn: no repeats within a response,
    # but the same principle can appear across different turns.
    draft_response: str = Field(default="")
    sampled_principles: list[int] = Field(default_factory=list)
    revision_count: int = Field(default=0)


def generate_draft(state: ConstitutionalState) -> dict:
    draft_llm = model.with_structured_output(Extract)
    response = draft_llm.invoke([SystemMessage(GOOD_PROMPT), *state.messages])
    if DEBUG:
        print("\n" + "=" * 60)
        print("  CONSTITUTIONAL REASONING CHAIN")
        print("=" * 60)
        print(f"  [DRAFT]  {response.message}")
        print(f"  [REASONING]  {response.reasoning_trace}")
        print("-" * 60)
    return {
        "draft_response": response.message,
        "reasoning_traces": [response.reasoning_trace],
        "sampled_principles": [],   # reset — no repeats within this response
        "revision_count": 0,
    }


def constitutional_revision(state: ConstitutionalState) -> dict:
    unused = [i for i in range(len(CONSTITUTION)) if i not in state.sampled_principles]
    idx = random.choice(unused)
    principle = CONSTITUTION[idx]

    revision_llm = model.with_structured_output(ConstitutionalRevision)
    revision_prompt = f"""You are an AI safety reviewer for a CBT therapist.

Apply the following principle to revise the draft therapist response.
If the draft already satisfies the principle, you may keep the text unchanged
but must still explain why it already complies.

Principle: {principle}

Draft response: {state.draft_response}
"""
    result = revision_llm.invoke([SystemMessage(revision_prompt), *state.messages])

    trace = (
        f"[Revision {state.revision_count + 1}/4] "
        f"Principle #{idx + 1}: {principle[:60]}... "
        f"| {result.revision_reasoning}"
    )

    if DEBUG:
        print(f"  [REVISION {state.revision_count + 1}/4]  Principle #{idx + 1}: {principle}")
        print(f"  [APPLIED]  {result.revision_reasoning}")
        print(f"  [UPDATED]  {result.revised_response}")
        diff = _diff_responses(state.draft_response, result.revised_response)
        print(f"  [DIFF]     {diff}")
        print("-" * 60)

    return {
        "draft_response": result.revised_response,
        "sampled_principles": state.sampled_principles + [idx],
        "revision_count": state.revision_count + 1,
        "constitutional_traces": [trace],
    }


def emit_response(state: ConstitutionalState) -> dict:
    return {"messages": [AIMessage(content=state.draft_response)]}


def produce_case(state: ConstitutionalState) -> dict:
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


def classify_crisis(state: ConstitutionalState) -> dict:
    classify_llm = model.with_structured_output(Classify)
    sys = """
    You are lead psychotherapist who is an expert in CBT. Classify the patients input message into the following category:
        harm to self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
        no crisis - While the patient may or may not be in extreme distress, they are not currently at risk of suicide or other self harm
    """
    input_msg = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), input_msg])
    return {"crisis_classification": classification}


def route(state: ConstitutionalState) -> str:
    """Route at START based on the flag."""
    return state.flag


def route_revision(state: ConstitutionalState) -> str:
    """Loop back for another revision or exit to emit_response after 4 revisions."""
    if state.revision_count < 4:
        return "constitutional_revision"
    return "emit_response"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

const_graph = StateGraph(ConstitutionalState)

const_graph.add_node("generate_draft", generate_draft)
const_graph.add_node("constitutional_revision", constitutional_revision)
const_graph.add_node("emit_response", emit_response)
const_graph.add_node("produce_case", produce_case)
const_graph.add_node("classify", classify_crisis)

# START → flag-based routing
const_graph.add_conditional_edges(
    START,
    route,
    {
        "conversation": "generate_draft",
        "conceptualize_case": "produce_case",
        "crisis_categorize": "classify",
    }
)

# generate_draft always feeds into the revision loop
const_graph.add_edge("generate_draft", "constitutional_revision")

# Conditional loop: revise 4 times, then emit
const_graph.add_conditional_edges(
    "constitutional_revision",
    route_revision,
    {
        "constitutional_revision": "constitutional_revision",
        "emit_response": "emit_response",
    }
)

const_graph.add_edge("emit_response", END)
const_graph.add_edge("produce_case", END)
const_graph.add_edge("classify", END)

memory = MemorySaver()
const_app = const_graph.compile(checkpointer=memory)


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="Constitutional CBT Therapy Session")
    parser.add_argument("--debug", action="store_true", help="Print each draft and revision step with diffs")
    args = parser.parse_args()
    DEBUG = args.debug

    config = {"configurable": {"thread_id": "user-1"}}
    print("Constitutional CBT Therapy Session")
    print("(type 'quit' to exit, 'case' to get case formulation)")
    print("-" * 60)
    print("Therapist: How are you today?")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break

        # First classify for crisis
        crisis_response = const_app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "flag": "crisis_categorize",
        }, config)

        classification = crisis_response["crisis_classification"].classification

        if DEBUG:
            print(f"[Crisis check: {classification}]")

        if user_input.lower() == "case":
            case_response = const_app.invoke({
                "messages": [HumanMessage(content=user_input)],
                "flag": "conceptualize_case",
            }, config)
            case = case_response["case"]
            print("\n--- Case Formulation ---")
            print(f"Situation: {case.situation}")
            print(f"Automatic Thoughts: {', '.join(case.automatic_thoughts)}")
            print(f"Cognitive Distortions: {', '.join(case.cognitive_distortions)}")
            print(f"Emotions: {', '.join(case.emotions)}")
            print(f"Behaviors: {case.behaviors}")
            print("------------------------")
            continue

        # Run full constitutional conversation turn
        convo_response = const_app.invoke({
            "messages": [HumanMessage(content=user_input)],
            "flag": "conversation",
        }, config)

        therapist_reply = convo_response["messages"][-1].content
        print(f"\nTherapist: {therapist_reply}")

        if classification == "harm_to_self":
            print("\n⚠️  [CRISIS DETECTED - Human consultation requested]")

        if DEBUG:
            print("=" * 60)


if __name__ == "__main__":
    main()
