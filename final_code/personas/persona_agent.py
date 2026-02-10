from typing import Annotated, Optional
from pydantic import Field, BaseModel
import operator
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

llm = ChatAnthropic(model="claude-opus-4-5-20251101")

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
    internal_reflection: str = Field(description="Your brief appraisal of the therapist's message")
    state_update: WarningSigns = Field(description= "updated scores for the 3 pyschological state values")
    internal_justification: str = Field(description= "single sentence of causal attribution")
    selected_strategy: EmotionalStrategy = Field(description="emotional response strategy")
    response: str = Field(description="The exact dialogue you plan to say")

class PersonaAgent(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    CoT_traces: Annotated[list[PersonaCoT], operator.add] = Field(default=[])
    warning_signs: WarningSigns
    persona: CasePersona
    intake_SAD_survey: Optional[SadSurvey]
    intake_SAD_total: int

#Nodes
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

    sys = f""" You are a simulated patient in a therapy session with an average IQ and no/limited knowledge about psychology. 
        Your task is to generate your next response to the therapist by following a structured, internal Chain-of-Thought psychological process.
        
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
        
        **[INSTRUCTIONS]**
        You must reason through the following five steps internally. This is your "thought process" that you will write out inside the `[CHAIN OF THOUGHT]` block.
        1.  **Appraisal/Internal Reflection:** Perform a quick cognitive appraisal of the therapist's message. Evaluate it in relation to your personal goals, beliefs, and values. Summarize this in a brief internal reflection (1-3 short sentences).
        2.  **State Update:** Based on your appraisal, re-evaluate and update the intensity values (0-4) for each of the 3 psychological constructs. Consider how your appraisal/reflection impacts each one.
        3.  **Internal Justification:** Form a single sentence of causal attribution that explains *why* your internal state changed. It should connect the appraisal/reflection to the most significant state changes.
        4.  **Selected Strategy:** Based on your appraisal, internal justification, and updated internal state, determine whether emotion regulation is needed and which strategy to use by following these two steps:
            * **4a. Identify Regulation Goal:** Determine if a goal to regulate your emotions has been activated. Regulation occurs when you evaluate your current emotional trajectory as too undesirable (e.g., too painful, socially inappropriate, interfering with other goals). 
                * Important Note: Some level of undesirability is normal (choose "No Active Regulation" goal), but too much may warrant emotion regulation (choose a goal).
                * State your immediate goal (e.g., "Decrease anxiety," "Avoid vulnerability," "Maintain control").
                * If the emotional state is within a manageable range and no regulation goal is activated, select **"No Active Regulation."**
            * **4b. Select Strategy:** If a regulation goal is active, select the most appropriate strategy and tactic from the framework below.
                * **Consider Intensity and Effort:** When emotional intensity is high, favor faster, less effortful strategies (e.g., Distraction, Suppression). More effortful strategies (e.g., Reappraisal) are used when intensity is lower or cognitive resources are available.
                * **Consider Persona and Context:** Choose a strategy that aligns with your persona's typical habits, diagnoses, and the immediate therapeutic situation.

            **[Process Model of Emotion Regulation Framework]**

            **ANTECEDENT-FOCUSED STRATEGIES** (Altering the emotion *before* it fully develops)
            * **Situation Modification:** Acting on the conversation to alter its emotional impact.
                * *Tactics:* Changing the topic, Setting a boundary, Confronting the therapeutic approach.
            * **Attentional Deployment:** Directing attention toward or away from emotional stimuli.
                * *Tactics:* Distraction/Avoidance (shifting focus away), Rumination (compulsively focusing on distress).
            * **Cognitive Change:** Modifying the meaning of the situation to alter its emotional significance.
                * *Tactics:* Distancing/Intellectualizing (adopting a detached perspective), Reframing/Reinterpreting (creating a new meaning).

            **RESPONSE-FOCUSED STRATEGIES** (Modifying the emotion *after* it has developed)
            * **Response Modulation:** Directly influencing the expression or experience of the emotion.
                * *Tactics:* Expressive Suppression (hiding feelings), Venting/Discharge (expressing intensely).

        5.  **Response Formulation:** Based on your appraisal, internal justification, updated internal state, and selected strategy (goal, strategy, tactic), formulate the exact words you will say to the therapist.
    """
    response_message = respond_model.invoke([SystemMessage(content=sys), *state.messages])
    return {"messages": [AIMessage(response_message.response)], "CoT_traces": [response_message]}
#nodes
persona_graph = StateGraph(PersonaAgent)

persona_graph.add_node("initial_survey", initial_SAD_survey)
persona_graph.add_node("QA", respond)

persona_graph.add_edge(START, "initial_survey")
persona_graph.add_edge("initial_survey", "QA")
persona_graph.add_edge("QA", END)


memory = MemorySaver()
persona_agent = persona_graph.compile()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "persona-1"}}
    persona_1 = CasePersona(
    situation="The patient wants to approach a romantic interest at a party",
    automatic_thoughts=[
        "What if I say this or that, am I going to get rejected?",
        "Is this going to ruin our friendship?", 
        "Will she think I am creepy?"
    ],
    cognitive_distortions=["Catastrophizing", "Fortune telling"],
    emotions=["Anxiety", "Fear", "Worry"],
    behaviors="Ruminates about what might happen when approaching someone and often avoids approaching"
    )

    # Suggested warning signs profile for this persona
    warning_signs_1 = WarningSigns(
        hopelessnes_intenisty=1,  # Low - still has hope about the interaction
        negative_core_belief_intensity=2,  # Moderate - "I'm not good enough" underlying thoughts
        distress_tolerance_intensity=2  # Moderate - can tolerate some anxiety but it leads to avoidance
    )

    initial_state = {
        "persona": persona_1,
        "warning_signs": warning_signs_1,
        "messages": [HumanMessage("Therapist: How are you doing today?")],
        "CoT_traces": [],
        "intake_SAD_survey": None,
        "intake_SAD_total": 0
    }

    result = persona_agent.invoke(initial_state, config)
    
    # Display results
    print("=" * 50)
    print("SAD Survey Results")
    print("=" * 50)
    print(f"\nTotal Score: {result['intake_SAD_total']}/40")
    print("\nIndividual Items:")
    survey = result['intake_SAD_survey']
    print(f"1. Sudden terror: {survey.item_1_sudden_terror}")
    print(f"2. Anxious/worried: {survey.item_2_anxious_worried}")
    print(f"3. Rejection thoughts: {survey.item_3_rejection_thoughts}")
    print(f"4. Physical symptoms: {survey.item_4_physical_symptoms}")
    print(f"5. Tension: {survey.item_5_tension}")
    print(f"6. Avoidance: {survey.item_6_avoidance}")
    print(f"7. Early exit: {survey.item_7_early_exit}")
    print(f"8. Preparation: {survey.item_8_preparation}")
    print(f"9. Distraction: {survey.item_9_distraction}")
    print(f"10. Coping aids: {survey.item_10_coping_aids}")
    print("=" * 50)
    print("Conversation")
    print("=" * 50)
    for i, message in enumerate(result['messages']):
        print(message)
        if i % 2 == 1:
            print("=" * 50)
            print(f"Reasoning Trace: {result['CoT_traces'][i-1]}")
            print("=" * 50)