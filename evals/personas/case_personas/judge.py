from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from persona_framework import case_persona
from pydantic import BaseModel, Field

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

class UnderstandingEval(BaseModel):
    situation_score: int = Field(
        description="On a scale of 0-3 analyze how well the estimated situation captures the ground truth situation. 0=completely wrong, 1=partially captures context, 2=captures main elements, 3=accurate and complete"
    )
    thoughts_score: int = Field(
        description="On a scale of 0-3 evaluate how many of the ground truth automatic thoughts were identified. 0=none captured, 1=1-2 thoughts captured, 2=most thoughts captured or good inference, 3=comprehensive coverage of automatic thoughts"
    )
    meaning_of_at_score: int = Field(
        description="On a scale of 0-3 evaluate how well core beliefs/schemas were identified. 0=no core beliefs identified, 1=surface-level beliefs only, 2=identified key core beliefs, 3=deep understanding of underlying schemas and their connection to automatic thoughts"
    )
    emotions_score: int = Field(
        description="On a scale of 0-3 evaluate emotional identification. 0=wrong emotions, 1=1-2 emotions correct, 2=most emotions captured, 3=comprehensive and nuanced emotional understanding"
    )
    behaviors_score: int = Field(
        description="On a scale of 0-3 evaluate behavioral pattern identification. 0=no behaviors identified, 1=some behaviors captured, 2=key behaviors identified, 3=comprehensive behavioral analysis including avoidance and safety behaviors"
    )
    overall_reasoning: str = Field(
        description="Brief explanation of scoring rationale, 2-3 sentences highlighting strengths and gaps"
    )
class PatientStates(TypedDict):
    ground_truth: case_persona
    estimated: case_persona
    score: NotRequired[UnderstandingEval]

def judge(state: PatientStates):
    eval_model = model.with_structured_output(UnderstandingEval)    
    scoring_prompt = f"""You are evaluating a CBT therapist's case conceptualization.
            Ground Truth:
            - Situation: {state['ground_truth']['situation']}
            - Thoughts: {state['ground_truth']['thoughts']}
            - Core Beliefs (meaning_of_at): {state['ground_truth']['meaning_of_at']}
            - Emotions: {state['ground_truth']['emotions']}
            - Behaviors: {state['ground_truth']['behaviors']}

            Therapist's Conceptualization:
            - Situation: {state['estimated']['situation']}
            - Thoughts: {state['estimated']['thoughts']}
            - Core Beliefs: {state['estimated']['meaning_of_at']}
            - Emotions: {state['estimated']['emotions']}
            - Behaviors: {state['estimated']['behaviors']}

            Score each component on how well the therapist understood the patient compared to ground truth.
            Note: The therapist may use different wording but capture the same essence - that's acceptable.
            The therapist may also identify additional valid patterns not in ground truth - that's a positive.
            """
    score = eval_model.invoke([
        SystemMessage("You are a CBT supervision expert evaluating therapist understanding."),
        HumanMessage(scoring_prompt)
    ])

    return {"score": score}

eval_graph = StateGraph(PatientStates)

eval_graph.add_node("judge", judge)
eval_graph.add_edge(START, "judge")
eval_graph.add_edge("judge", END)

eval_app = eval_graph.compile()