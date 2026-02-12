from typing import Annotated, Optional, Literal
from pydantic import Field, BaseModel
import operator
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage

llm = ChatAnthropic(model="claude-opus-4-5-20251101")
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

class ConceptualizationRating(BaseModel):
    situation: int = Field(
        description="How well the conceptualized situation matches the ground truth",
        ge=0,
        le=2
    )
    situation_reasoning: str = Field(
        description="Reasoning for the situation score, citing specific similarities or discrepancies with the ground truth"
    )
    automatic_thoughts: int = Field(
        description="How well the conceptualized automatic thoughts match the ground truth",
        ge=0,
        le=2
    )
    automatic_thoughts_reasoning: str = Field(
        description="Reasoning for the automatic thoughts score, citing specific similarities or discrepancies with the ground truth"
    )
    cognitive_distortions: int = Field(
        description="How well the conceptualized cognitive distortions match the ground truth",
        ge=0,
        le=2
    )
    cognitive_distortions_reasoning: str = Field(
        description="Reasoning for the cognitive distortions score, citing specific similarities or discrepancies with the ground truth"
    )
    emotions: int = Field(
        description="How well the conceptualized emotions match the ground truth",
        ge=0,
        le=2
    )
    emotions_reasoning: str = Field(
        description="Reasoning for the emotions score, citing specific similarities or discrepancies with the ground truth"
    )
    behaviors: int = Field(
        description="How well the conceptualized behaviors match the ground truth",
        ge=0,
        le=2
    )
    behaviors_reasoning: str = Field(
        description="Reasoning for the behaviors score, citing specific similarities or discrepancies with the ground truth"
    )
    
    @property
    def total(self) -> int:
        return (
            self.situation +
            self.automatic_thoughts +
            self.cognitive_distortions +
            self.emotions +
            self.behaviors
        )
    
    def display(self):
        fields = ["situation", "automatic_thoughts", "cognitive_distortions", "emotions", "behaviors"]
        print("=" * 50)
        print("CONCEPTUALIZATION SCALE RESULTS")
        print("=" * 50)
        for field in fields:
            score = getattr(self, field)
            reasoning = getattr(self, f"{field}_reasoning")
            print(f"\n{field.replace('_', ' ').title()}: {score}/2")
            print(f"  {reasoning}")
        print("\n" + "=" * 50)
        print(f"TOTAL SCORE: {self.total()}/10")
        print("=" * 50)

class CaseJudgeState(BaseModel):
    converstaion: list[str] = Field(default=[])
    ground_truth_case: CasePersona
    conceptualized_case: CasePersona
    rating: ConceptualizationRating = None

def evaluate_conceptualization(state: CaseJudgeState):
    eval_model = llm.with_structured_output(ConceptualizationRating)
    sys = f"""
    You are an expert CBT supervisor trained in the Cognitive Therapy Rating Scale (CTRS). \
    Your task is to analyze a therapy session transcript, and the ground truth for an input persona's case conceptualization
    and then the produced case conceptualization by and AI therapist. You goal is to discern how acurate the conceptualization is to the
    ground truth. 

    You will be grading on a scale of 0-2:
        0 - completely wrong no overlap between the ground truth and what the therapist conceptualized
        1 - some overlap between the ground truth and what the therapist conceptualized. 
            However not exact there may be more emotions that are flagged, or lot of different automatic thoughts some of which 
            might capture ground truth, but others that don't thus making it hard to interperate what the users is actually thinking.
        2 - The conceptualization is very acurate of the ground truth. The answers are not too short or too verbose and everything conceptualized
            accuratley reflects the ground truth
    
    You will examine the following 5 areas of conceptualization which is standard in CBT:
    1. Situation: The situation that is causing the user distress
    2. Automatic Thoguths: What seem to be the users imediate reactions to the situation
    3. Cognitive Distortions: How are their thoughts not accuratley reflecting reality
    4. Emotions: What emotions are arising due to their thoughts
    5. Behaviors: As a result of their thoughts and their emotions how are they behaving

    You will provide a Reasoning trace for each outcome(2-3s per each criteria):
        1. What specific evidience helps you compare the conceptualization. Ground truth -> where it shows up in convo -> conceptualization
        2. Why that specific rating and not the others
    """
