from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

DEBUG = False
model = ChatAnthropic(model="claude-haiku-4-5-20251001",  max_tokens=8192)

class PatientState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    reasoning_traces: Annotated[list[str], operator.add]

class Extract(BaseModel):
   message: str = Field(description="Your response to the patient, keep it concise")
   reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")

def conversation(state: PatientState):
    user_message = state["messages"][-1]
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage("""
        Act as a lead CBT pyschotherapist who is responsible for helping patients understand their negative thoughts and anxieties working to deeply understand
        what does a user feel and why?

        Make sure to reason through everything and your decision for follow up questions
        """), user_message])
    if DEBUG:
        print("###############TESTING##################")
        print(state["messages"])
        print("########################")
    return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

base_state = StateGraph(PatientState)

base_state.add_node("convo", conversation)

base_state.add_edge(START, "convo")
base_state.add_edge("convo", END)

memory = MemorySaver()
base_app = base_state.compile(checkpointer=memory)

def main():
    config = {"configurable": {"thread_id": "user-1"}}

    print("Welcome to Baseline, an AI CBT agent.")
    print("Please note you are chatting with an LLM. If you are in crisis or at risk of self harm please call or text 988")
    print("\nBaseline: How are you doing today?\n")

    initial_state = {
        "messages": [],
        "reasoning_traces": []
    }
    first_turn = True

    while True:
        user_input = input ("You: ")

        if first_turn:
            invoke_state = {
                **initial_state,
                "messages": [HumanMessage(content=user_input)]
            }
            first_turn = False
        else:
            invoke_state = {
                "messages": [HumanMessage(content=user_input)]
            }
        
        response = base_app.invoke(invoke_state, config)
        print(f"AI Therapist: {response['messages'][-1].content} \n")
        print(f"###########Reasoning Trace############\n {response['reasoning_traces'][-1]}")

if __name__ == "__main__":
    main()