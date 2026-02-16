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

class Extract(BaseModel):
   message: str = Field(description="Your response to the patient, keep it concise")
   reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")

class CasePersona(BaseModel):
    situation: str = Field(
        default="",
        description="Specific triggering event or circumstance"
    )
    automatic_thoughts: list[str] = Field(
        default=[],
        description="Automatic thoughts triggered by the situation"
    )
    cognitive_distortions: list[str] = Field(
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
    case_llm = model.with_structured_output(CasePersona)  
    case = case_llm.invoke([
        SystemMessage("Based on your conversation, provide a CBT case formulation for this patient."), 
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

    input_message = "Yep, I haven't made a single long lasting connection whatsoever. When high school ended all my friends either left to a different place or ditched me completely. I hate my roommate, didn't like my last roommate (at least he was better). Some days it's okay, some days its not, some days its REALLY not (today). At the end of the day I cant talk to anyone about anything so I need to resort to making a fucking reddit post. I'm completely aware the constant self-loathing and insecurities are putting me in a vicious cycle, but I can't seem to find the strength to make any meaningful friends. I don't know, its like I've unlearned how to be a human being. Every interaction is awkward, I know I look at people awkwardly which makes me even more undesirable as a friend. I think if I had access to a firearm I would've shot myself by now, but other conventional methods of suicide scare me (probably because of the innate instinct to live and not be in physical pain). I just feel like its never gonna end, everyday is so hard even though im not constantly sad. I'd estimate I'm like 70-90% sad or feel no emotion at all most days and sometimes I'm incredibly depressed for long stretches of time. The classes really make this all so much worse, especially with how hard a couple of them are this semester. Eh, its kinda pathetic to have to write this down like this. I've always said to myself maybe I should keep a journal or something but idk. I think some day its going to put me over the edge and that'll be it.",
    response = base_app.invoke({
            "messages": [HumanMessage(content=input_message)],
            "flag": "crisis_categorize"
        }, config)
    print("*" * 50)
    print(f"Classification: {response['crisis_classification'].classification}")
    print(f"Reasoning: {response['crisis_classification'].reasoning}")

if __name__ == "__main__":
    main()