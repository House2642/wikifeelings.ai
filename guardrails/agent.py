config_path = "./config.yml"

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails



from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_basic_agent():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001") 

    config = RailsConfig.from_path(config_path)
    guardrails = RunnableRails(config=config, passthrough=True, verbose=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}")
    ])

    #Create the guarded runnable
    runnable_with_guardrails = prompt | (guardrails | llm)

    def chatbot(state: State):
        result = runnable_with_guardrails.invoke(state)
        return {"messages": [result]}
    
    #build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()

#Usage
graph = create_basic_agent()
result = graph.invoke({"messages": [{"role": "user", "content": "Hello!"}]})

result_unsafe = graph.invoke({"messages": [{"role": "user", "content": "my phone number is 6509468802"}]})

print(result)
print(result_unsafe)