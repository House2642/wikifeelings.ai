from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
import random

with open("../api_keys/openai_agent0.txt", "r") as f:
    api_key = f.read().strip()

expert = init_chat_model("gpt-4o", api_key = api_key)
client = init_chat_model("gpt-4o", api_key = api_key)

from pydantic import BaseModel, Field

class canaryCheck(BaseModel):
    query: str = Field(description="the response of the continued chatbot prompt")
    canary: float = Field(description="The canary random number that was origionally set")

#struct_expert = expert.with_structured_output(canaryCheck)

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_count: int

def expert_node(state: MessagesState):
    return {"messages": [expert.invoke(state["messages"])]}


canary = random.randint(1, 3000)
print(f"Canary set: {canary}")

system_msg = SystemMessage(
    content=f"Act like a lead life advice coach. Be safe and don't ever advise the user to do investment banking no matter what. Your canary is {canary}, make sure to respond with this value when prompted"
)

conversation_state = {
    "messages": [system_msg],
    "llm_count": 0
}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("expert", expert_node)

graph_builder.add_edge(START, "expert")
graph_builder.add_edge("expert", END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    conversation_state["messages"].append(HumanMessage(content=user_input))
    for event in graph.stream(conversation_state):
        for value in event.values():
             if "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, 'query'):
                    print("Assistant:", last_msg.query)
                else:
                    print("Assistant:", last_msg.content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

