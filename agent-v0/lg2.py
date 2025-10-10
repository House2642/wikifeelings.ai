from langchain_core.tools import tool 
from langchain.chat_models import init_chat_model

with open("../api_keys/openai_agent0.txt","r") as f:
    api_key = f.read().strip()

llm = init_chat_model(model="gpt-4o", temperature=0, api_key=api_key)

#define tools

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def compoundInterest(p: int, t: int, i: float, ) -> float:
    """Compount p over t years at an interest rate i
    Args:
        p: first int
        t: second int
        i: first float
    """
    n = 1
    return p * (1+i/1)**(n*t)

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def divide(a:int, b:int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide, compoundInterest]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

#Define State
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Define model node gives the system prompt and the user message

from langchain_core.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic or compounding interest on a set of inputs"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) +1
    }

#define tool node
from langchain_core.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id= tool_call["id"]))
    return {"messages": result}

#define end logic --> conditional edge function, return if the agent doesn't call a tool meaning they have the answer

from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]

    #was the last message a tool call?
    if last_message.tool_calls:
        return "tool_node"
    
    return END

#No that I have all the components, I can build the workflow
agent_builder = StateGraph(MessagesState)

#Add Nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue, 
    ["tool_node", END]
)

agent_builder.add_edge("tool_node", "llm_call")

#Compile the agent
agent = agent_builder.compile()

from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

#Invoke
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="How much would $1000 invest of 30 years at an interest rate of 7 percent be?")]
messages = agent.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()