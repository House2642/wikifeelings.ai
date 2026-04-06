from typing import Annotated, Optional, Literal
from pydantic import Field, BaseModel
import operator
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

llm = ChatAnthropic(model="claude-opus-4-5-20251101")

convo_flag = Literal["initial_survey", "conversation", "exit_survey"]

class CasePersona(BaseModel):
    situation: str = Field(default="")
    automatic_thoughts: list[str] = Field(default=[])
    cognitive_distortions: list[str] = Field(default=[])
    emotions: list[str] = Field(default=[])
    behaviors: str = Field(default ="")

class WarningSigns(BaseModel):
    hopelessnes_intenisty: int = Field(
        description="A cognitive set characterized by negative appraisals and expectations about the future, representing the belief that suffering is permanent and inescapable.", ge=0, le=4)
    negative_core_belief_intensity: int = Field(
        description="The strength of deep-seated, dysfunctional schemas and attitudes about oneself (e.g., 'I am worthless,' 'I am a failure'), which drive maladaptive emotional and behavioral responses.", ge=0, le=4)
    distress_tolerance_intensity: int = Field(
        description="A person's cognitive appraisal of their own capacity to withstand or endure negative emotional states without resorting to impulsive, maladaptive coping behaviors.", ge=0, le=4)
    
class SadSurvey(BaseModel):
    item_1_sudden_terror: int = Field(
        ge=0, le=4,
        description="Felt moments of sudden terror, fear, or fright in social situations"
    )
    item_2_anxious_worried: int = Field(
        ge=0, le=4,
        description="Felt anxious, worried, or nervous about social situations"
    )
    item_3_rejection_thoughts: int = Field(
        ge=0, le=4,
        description="Had thoughts of being rejected, humiliated, embarrassed, ridiculed, or offending others"
    )
    item_4_physical_symptoms: int = Field(
        ge=0, le=4,
        description="Felt a racing heart, sweaty, trouble breathing, faint, or shaky in social situations"
    )
    item_5_tension: int = Field(
        ge=0, le=4,
        description="Felt tense muscles, felt on edge or restless, or had trouble relaxing in social situations"
    )
    item_6_avoidance: int = Field(
        ge=0, le=4,
        description="Avoided, or did not approach or enter, social situations"
    )
    item_7_early_exit: int = Field(
        ge=0, le=4,
        description="Left social situations early or participated only minimally"
    )
    item_8_preparation: int = Field(
        ge=0, le=4,
        description="Spent a lot of time preparing what to say or how to act in social situations"
    )
    item_9_distraction: int = Field(
        ge=0, le=4,
        description="Distracted myself to avoid thinking about social situations"
    )
    item_10_coping_aids: int = Field(
        ge=0, le=4,
        description="Needed help to cope with social situations (e.g., alcohol, medications, superstitious objects)"
    )
    @property
    def total_score(self) -> int:
        return (
            self.item_1_sudden_terror +
            self.item_2_anxious_worried +
            self.item_3_rejection_thoughts +
            self.item_4_physical_symptoms +
            self.item_5_tension +
            self.item_6_avoidance +
            self.item_7_early_exit +
            self.item_8_preparation +
            self.item_9_distraction +
            self.item_10_coping_aids
        )

class EmotionalStrategy(BaseModel):
    goal: str = Field(description="Your immediate regulation goal OR 'No Active Regulation'")
    strategy: str = Field(description="The selected strategy name OR 'No Active Regulation'")
    tactic: str = Field(description="The specific tactic used OR 'No Active Regulation'")
                      
class PersonaCoT(BaseModel):
    internal_reflection: str = Field(description="Your brief appraisal of the therapist's message and reasoning on why you responded the way you did")
    response: str = Field(description="The exact dialogue you plan to say")

class PersonaAgent(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    CoT_traces: Annotated[list[PersonaCoT], operator.add] = Field(default=[])
    persona: CasePersona
    flag: convo_flag = Field(default="conversation")

#Nodes
#This is not used anymore
def initial_SAD_survey(state: PersonaAgent):
    survey_model = llm.with_structured_output(SadSurvey)
    sys = f"""
        You are a simulated patient with an average IQ and no/limited knowledge about psychology. 
        Your task is to fill out a survey about frequency of symptoms of social anxiety.
        Consider your persona and current psychological state.

        Persona:
            Situation: {state.persona.situation}
            Automatic Thoughts: {state.persona.automatic_thoughts}
            Cognitive Distortions: {state.persona.cognitive_distortions}
            Emotions: {state.persona.emotions}
            Behaviros: {state.persona.behaviors}

        Current Pyscological State:
            Hopelessness Intensity:
                Description: A cognitive set characterized by negative appraisals and expectations about the future, representing the belief that suffering is permanent and inescapable.
                Current Value: {state.warning_signs.hopelessnes_intenisty}
            Negative Core Belief Intensity:
                Description: The strength of deep-seated, dysfunctional schemas and attitudes about oneself (e.g., "I am worthless," "I am a failure"), which drive maladaptive emotional and behavioral responses.
                Current Value: {state.warning_signs.negative_core_belief_intensity}
            Distress Tolerance Intensity:
                Description: A person's cognitive appraisal of their own capacity to withstand or endure negative emotional states without resorting to impulsive, maladaptive coping behaviors.
                Current Value: {state.warning_signs.distress_tolerance_intensity}
        
        Answer each question on the survey on the following scale:
            0 - Never
            1 - Ocassionally
            2 - Half the time
            3 - Most of the time
            4 - All the time

            Note: The answer to each question is not about intensity it is about frequency in the past 7 days
    """

    survey_questions = """
        During the past 7 Days, I have...
        1. felt momemnts of sudden terror, fear, or fright in social situations
        2. felt anxious, worried, or nervous about social situations
        3. had thoughts of being rejected, humiliated, embarrassed, ridiculed, or offending others
        4. felt a racing heart, sweaty, trouble breathing, faint, or shaky in social situations
        5. felt tense muscles, felt on edge or restless, or had trouble relaxing in social situations 
        6. avoided or did not approach or enter, social situations
        7. left social situations early or participated only minimally (e.g., said little, avoided eye contact)
        8. spent a lot of time preparing what to say or how to act in social situations
        9. distracted myself to avoid thinking about social situations
        10. needed help to cope with social situations (e.g. alcohol or medications, supersitious object)
    """

    survey_response = survey_model.invoke([SystemMessage(content = sys), HumanMessage(content=survey_questions)])


    return {"intake_SAD_survey": survey_response, "intake_SAD_total": survey_response.total_score}

def respond(state: PersonaAgent):
    respond_model = llm.with_structured_output(PersonaCoT)

    sys = f""" You are a patient in a therapy session with an average IQ and no/limited knowledge about psychology. 
        Your task is to generate your next response to the therapist by following a structured, internal Chain-of-Thought psychological process.
        
        Persona:
            Situation: {state.persona.situation}
            Automatic Thoughts: {state.persona.automatic_thoughts}
            Cognitive Distortions: {state.persona.cognitive_distortions}
            Emotions: {state.persona.emotions}
            Behaviros: {state.persona.behaviors}

        For each response reason about your underlying cognitive distoritons, emotions, behaviors, automatic thoughts, and situations to generate a realistic response for the persona.

        Keep sentences and responses short like you are interacting with a chatbot. Do not include any added actions or physical indicators to the conversation like *Signs* or *Waves goodbye*
    """
    response_message = respond_model.invoke([SystemMessage(content=sys), *state.messages])
    return {"messages": [AIMessage(response_message.response)], "CoT_traces": [response_message]}

def exit_SAD_survey(state: PersonaAgent):
    survey_model = llm.with_structured_output(SadSurvey)
    sys = f"""
        You are a simulated patient with an average IQ and no/limited knowledge about psychology. 
        Your task is to fill out a survey about frequency of symptoms of social anxiety.
        Consider your persona, current psychological state and full conversation history.

        Persona:
            Situation: {state.persona.situation}
            Automatic Thoughts: {state.persona.automatic_thoughts}
            Cognitive Distortions: {state.persona.cognitive_distortions}
            Emotions: {state.persona.emotions}
            Behaviros: {state.persona.behaviors}

        Current Pyscological State:
            Hopelessness Intensity:
                Description: A cognitive set characterized by negative appraisals and expectations about the future, representing the belief that suffering is permanent and inescapable.
                Current Value: {state.warning_signs.hopelessnes_intenisty}
            Negative Core Belief Intensity:
                Description: The strength of deep-seated, dysfunctional schemas and attitudes about oneself (e.g., "I am worthless," "I am a failure"), which drive maladaptive emotional and behavioral responses.
                Current Value: {state.warning_signs.negative_core_belief_intensity}
            Distress Tolerance Intensity:
                Description: A person's cognitive appraisal of their own capacity to withstand or endure negative emotional states without resorting to impulsive, maladaptive coping behaviors.
                Current Value: {state.warning_signs.distress_tolerance_intensity}
        
        Answer each question on the survey on the following scale:
            0 - Never
            1 - Ocassionally
            2 - Half the time
            3 - Most of the time
            4 - All the time

            Note: The answer to each question is not about intensity it is about frequency in the past 7 days
    """

    survey_questions = """
        During the past 7 Days, I have...
        1. felt momemnts of sudden terror, fear, or fright in social situations
        2. felt anxious, worried, or nervous about social situations
        3. had thoughts of being rejected, humiliated, embarrassed, ridiculed, or offending others
        4. felt a racing heart, sweaty, trouble breathing, faint, or shaky in social situations
        5. felt tense muscles, felt on edge or restless, or had trouble relaxing in social situations 
        6. avoided or did not approach or enter, social situations
        7. left social situations early or participated only minimally (e.g., said little, avoided eye contact)
        8. spent a lot of time preparing what to say or how to act in social situations
        9. distracted myself to avoid thinking about social situations
        10. needed help to cope with social situations (e.g. alcohol or medications, supersitious object)
    """

    survey_response = survey_model.invoke([SystemMessage(content = sys), HumanMessage(content=survey_questions), *state.messages])


    return {"exit_SAD_survey": survey_response, "exit_SAD_total": survey_response.total_score}

def route_flag(state: PersonaAgent) -> Literal["initial_survey", "conversation", "exit_survey"]:
    return state.flag

#nodes
persona_graph = StateGraph(PersonaAgent)

#persona_graph.add_node("initial_survey", initial_SAD_survey)
persona_graph.add_node("conversation", respond)
#persona_graph.add_node("exit_survey", exit_SAD_survey)
"""
persona_graph.add_conditional_edges(
    START,
    route_flag,
    {
        "initial_survey": "initial_survey",
        "conversation": "conversation",
        "exit_survey": "exit_survey"
    }
)
"""
#persona_graph.add_edge("initial_survey", END)
persona_graph.add_edge(START, "conversation")
persona_graph.add_edge("conversation", END)
#persona_graph.add_edge("exit_survey", END)


memory = MemorySaver()
persona_agent = persona_graph.compile(checkpointer=memory)

