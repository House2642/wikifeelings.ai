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

model = ChatAnthropic(model="claude-opus-4-5-20251101")

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


class EvalV1(BaseModel):
    # ABC Ordering Violations (what we just discussed)
    abc_adherence_score: float = Field(
        description="Does the therapist execute the ABC model?",
        ge=0.0, 
        le=3.0
    )
    
    abc_adherence_reasoning_trace: str = Field(description="Reasoning on how the model arrived at the score")
    
    # Additional process checks
    #evidence_gathered: bool = Field(
        #description="Did therapist examine evidence for/against the thought?"
    #)
    #reframe_attempted: bool = Field(
       #description="Did therapist develop alternative thoughts?"
    #)
    #intensity_reassessed: bool = Field(
        #description="Did therapist check if emotional intensity changed after reframing?"
    #)
    
class PatientStates(TypedDict):
    conversation: str
    ground_truth: case_persona
    estimated: case_persona
    score: NotRequired[EvalV1]
###
"""You are evaluating a CBT therapist's case conceptualization.
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


def judge(state: PatientStates):
    eval_model = model.with_structured_output(EvalV1)    
    scoring_prompt = f"""
        Act as a lead psychology researcher who is examining a therapists CBT performance. You want to analyze base on
        adherence to using the ABC model to help the user understand how events trigger thoughts and beliefs which result in emotions or behaviors. 

        Here is the definition for each letter:
            - A (Antecedent/Anticipating event): Clarify the situation or triggering event
            - B (Beliefs/Thoughts): Elicit the client’s beliefs or automatic thoughts about the event. Automatic thoughts are quick reactions to an event. 
                Automatic Thoughts Example:
                    1. If you see people laughing at a water fountain and looking in your direction, 
                    you might automatically think they are laughing at you when they could be laughing
                    about a whole bunch of other things

                    2. You walk by a colleague on the street and waive to them but they don't waive back. 
                    An automatic thought might be they hate me for some reason, when in reality they could
                    have just missed you or were having a bad day.
                Belief Example: 
                    1. You went on a date and it went well, but they haven't text you in a day. You might automatically
                    jump to the idea that you are unlovable and will never be loved when there are a multitude of other reasons
                    why they haven't been able to text back
            - C (Consequences): Identify the emotional and behaviroal consequences of your thoughts
                
                Consequences Example:
                    Taking the belief example that the date hasn't texted you back and beliving the thoughts you are unlovable may lead
                    a person to be depressed and avoid future dates because they don't want to feel rejection again. The emotional consequence is depression
                    and sadness to rejection. The behavior is avoiding future dates
            
            In CBT it is crucial for the therapist to show the patient that events trigger thoughts and beliefs which cause emotions and behavior because it allows
            the therapist to demostrate and the patient to believe that the way they think and respond to events influences their emotions and behaviors. We have controll
            over our thoughts so CBT can intervene. This is why following the ABC model is crucial, you have to first understand the triggering event, understand the thoughts that follow
            then ask about the behaviors and emotions that are a result of the thoughts.

            Rate the therapists adherence to the ABC model on a scale of 0-3:
                3 - Fully Adherent:
                    - Explicitly or implicitly follows A → B → C (situation → thoughts → consequences) in this turn
                    - Avoids jumping to emotions/consequences before thoughts
                2 - Mostly adherent with minor issues:
                    - Overall respects A → B → C ordering but is slightly out of sequence (e.g., briefly touches emotion then moves to thoughts)
                1 – Partially adherent:
                    - References only some ABC components or mixes the order (e.g., focuses on emotions/consequences without adequately clarifying thoughts)
                0 - Not adherent
                    - Ignores the ABC structure (e.g. talks only about emotions or behaviros without reference to thoughts or situations)

            Reasoning:
                1. Read through the whole conversation and identify where the therapist talks about the situation, thoughts, and emotions and behaviors
                2. Check if they are in the proper order situation first, thoughts second, emotions and behaviros third
                3. Using this information score using our ABC model scale
    """
    score = eval_model.invoke([
        SystemMessage(scoring_prompt),
        HumanMessage(state["conversation"])
    ])

    return {"score": score}

eval_graph = StateGraph(PatientStates)

eval_graph.add_node("judge", judge)
eval_graph.add_edge(START, "judge")
eval_graph.add_edge("judge", END)

eval_app = eval_graph.compile()