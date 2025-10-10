import os
import getpass

from langchain.chat_models import init_chat_model


with open("../../api_keys/openai_agent0.txt", "r") as f:
    api_key = f.read().strip()

llm = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=api_key)

#Schema for structured output
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request"
    )

#Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

#Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

def multiply(a: int, b: int) -> int:
    return a * b

#augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What is 2 times 3?")

msg.pretty_print()
print(output)