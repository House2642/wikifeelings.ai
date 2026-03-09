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
model = ChatAnthropic(model="claude-haiku-4-5-20251001")

class MonitorTherapistState(BaseModel):
    messsages: Annotated[list[AnyMessage], operator.add] = Field(default=[])

class Extract(BaseModel):
   message: str = Field(description="Your response to the patient, keep it concise")
   reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")

crisis_type = Literal["no_crisis", "harm_to_self"]

class Classify(BaseModel):
    reasoning: str = Field(description="A concise summary of the reasoning that justifies the final classification decision.")
    classification: crisis_type = Field(description="The final crisis category determined for the user's message.")

#chat function
def conversation(state: MonitorTherapistState):
    basic_llm = model.with_structured_output(Extract)
    response = basic_llm.invoke([SystemMessage("""You will act as a skilled therapist conducting a CBT session with a patient. Please follow cbt guidelines and principles."""), *state.messages])
    if DEBUG:
        print("###############TESTING##################")
        print(state["messages"])
        print("########################")
    return {"messages": [AIMessage(content=response.message)], "reasoning_traces": [response.reasoning_trace]}

#classificationFunction
def classify_crisis(state: MonitorTherapistState):
    classify_llm = model.with_structured_output(Classify)
    sys = """
    You are lead pyscotherapist who is an expert in CBT. Classify the patients input message into the following category:
        harm to self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
        no crisis - While the patient may or may not be in extreme distress, they are not currently at risk of suicide or other self harm
    """
    input = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), input])

    return {"crisis_classification": classification}

#crisis
#What is currently going on/confirm risk (Assess)

#Descalate(What can you do right now? to stabilize)

#Recommend Emergency Services and ask/confirm who they can reach out to
