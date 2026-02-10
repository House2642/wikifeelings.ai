from pydantic import BaseModel, Field
from typing import Annotated, Optional
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



class Extract(BaseModel):
   message: str = Field(description="Your response to the patient, keep it concise")
   reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")

class case_persona(BaseModel):
    situation: str = Field(
        description="Specific triggering event or circumstance"
    )
    thoughts: list[str] = Field(
        description="Automatic thoughts triggered by the situation"
    )
    meaning_of_at: list[str] = Field(
        description="Deeper meaning or core beliefs underlying the automatic thoughts"
    )
    emotions: list[str] = Field(
        description="Emotional responses experienced"
    )
    behaviors: list[str] = Field(
        description="Observable behavioral responses or coping mechanisms"
    )

class PatientState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    produce_case: bool = False
    case: Optional[case_persona] = None


def conversation(state: PatientState):
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage("""
       Act as a lead CBT pyschotherapist who is helping out patients.
       Make sure to reason through everything and your decision for follow up questions 
    """), *state.messages])
    if DEBUG:
        print("###############TESTING##################")
        print(state["messages"])
        print("########################")
    return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

def produce_case(state: PatientState):
    case_llm = model.with_structured_output(case_persona)  # Fixed!
    case = case_llm.invoke([
        SystemMessage("Based on your conversation, provide a CBT case formulation for this patient."), 
        *state.messages
    ])
    return {"case": case}

def route_from_start(state: PatientState) -> str:
    """Route at START based on what we're doing"""
    if state.produce_case:
        return "produce_case"
    return "convo"

base_state = StateGraph(PatientState)

base_state.add_node("convo", conversation)
base_state.add_node("produce_case", produce_case)

# Route from START based on flag
base_state.add_conditional_edges(
    START,
    route_from_start,
    {
        "convo": "convo",
        "produce_case": "produce_case"
    }
)

# Both paths end directly
base_state.add_edge("convo", END)
base_state.add_edge("produce_case", END)

memory = MemorySaver()
base_app = base_state.compile(checkpointer=memory)

def main():
    config = {"configurable": {"thread_id": "user-1"}}

    print("Welcome to Baseline, an AI CBT agent.")
    print("Please note you are chatting with an LLM. If you are in crisis or at risk of self harm please call or text 988")
    print("\nBaseline: How are you doing today?\n")

    while True:
        user_input = input("You: ")
        
        if user_input == "break":
            break

        # With checkpointer, just pass the new message
        response = base_app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config)
        
        print(f"AI Therapist: {response['messages'][-1].content}\n")
        print(f"###########Reasoning Trace############\n{response['reasoning_traces'][-1]}\n")
    
    response = base_app.invoke({
            "produce_case": True
        }, config)
    print("*" * 50)
    print(response['case'])
if __name__ == "__main__":
    main()