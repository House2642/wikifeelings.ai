from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

DEBUG = False

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

class AnxiousFeelings(BaseModel):
    """Burns Anxiety Inventory Category I: Anxious Feelings (Items 1-6)"""
    anxiety_nervousness_worry_fear: int = Field(default=0, ge=0, le=3, description="Anxiety, nervousness, worry or fear")
    feeling_things_strange_unreal: int = Field(default=0, ge=0, le=3, description="Feeling that things around you are strange or unreal")
    feeling_detached_from_body: int = Field(default=0, ge=0, le=3, description="Feeling detached from all or part of your body")
    sudden_unexpected_panic_spells: int = Field(default=0, ge=0, le=3, description="Sudden unexpected panic spells")
    apprehension_sense_of_impending_doom: int = Field(default=0, ge=0, le=3, description="Apprehension or a sense of impending doom")
    feeling_tense_stressed_uptight_on_edge: int = Field(default=0, ge=0, le=3, description="Feeling tense, stressed, 'uptight' or on edge")


class AnxiousThoughts(BaseModel):
    """Burns Anxiety Inventory Category II: Anxious Thoughts (Items 7-17)"""
    difficulty_concentrating: int = Field(default=0, ge=0, le=3, description="Difficulty concentrating")
    racing_thoughts: int = Field(default=0, ge=0, le=3, description="Racing thoughts")
    frightening_fantasies_or_daydreams: int = Field(default=0, ge=0, le=3, description="Frightening fantasies or daydreams")
    feeling_on_verge_of_losing_control: int = Field(default=0, ge=0, le=3, description="Feeling that you're on the verge of losing control")
    fears_of_cracking_up_or_going_crazy: int = Field(default=0, ge=0, le=3, description="Fears of cracking up or going crazy")
    fears_of_fainting_or_passing_out: int = Field(default=0, ge=0, le=3, description="Fears of fainting or passing out")
    fears_of_physical_illness_heart_attack_dying: int = Field(default=0, ge=0, le=3, description="Fears of physical illnesses or heart attacks or dying")
    concerns_about_looking_foolish_or_inadequate: int = Field(default=0, ge=0, le=3, description="Concerns about looking foolish or inadequate")
    fears_of_being_alone_isolated_abandoned: int = Field(default=0, ge=0, le=3, description="Fears of being alone, isolated, or abandoned")
    fears_of_criticism_or_disapproval: int = Field(default=0, ge=0, le=3, description="Fears of criticism or disapproval")
    fears_something_terrible_about_to_happen: int = Field(default=0, ge=0, le=3, description="Fears that something terrible is about to happen")


class AnxiousPhysical(BaseModel):
    """Burns Anxiety Inventory Category III: Physical Symptoms (Items 18-33)"""
    skipping_racing_pounding_heart: int = Field(default=0, ge=0, le=3, description="Skipping, racing or pounding of the heart (palpitations)")
    pain_pressure_tightness_in_chest: int = Field(default=0, ge=0, le=3, description="Pain, pressure, or tightness in chest")
    tingling_numbness_toes_fingers: int = Field(default=0, ge=0, le=3, description="Tingling or numbness of toes and fingers")
    butterflies_discomfort_in_stomach: int = Field(default=0, ge=0, le=3, description="Butterflies or discomfort in the stomach")
    constipation_or_diarrhea: int = Field(default=0, ge=0, le=3, description="Constipation or diarrhea")
    restlessness_or_jumpiness: int = Field(default=0, ge=0, le=3, description="Restlessness or jumpiness")
    tight_tense_muscles: int = Field(default=0, ge=0, le=3, description="Tight, tense muscles")
    sweating_not_brought_on_by_heat: int = Field(default=0, ge=0, le=3, description="Sweating not brought on by heat")
    lump_in_throat: int = Field(default=0, ge=0, le=3, description="A lump in the throat")
    trembling_or_shaking: int = Field(default=0, ge=0, le=3, description="Trembling or shaking")
    rubbery_or_jelly_legs: int = Field(default=0, ge=0, le=3, description="Rubbery or 'jelly' legs")
    feeling_dizzy_lightheaded_off_balance: int = Field(default=0, ge=0, le=3, description="Feeling dizzy, lightheaded or off balance")
    choking_smothering_difficulty_breathing: int = Field(default=0, ge=0, le=3, description="Choking or smothering sensations or difficulty breathing")
    headaches_pains_neck_or_back: int = Field(default=0, ge=0, le=3, description="Headaches or pains in the neck or back")
    hot_flashes_or_cold_chills: int = Field(default=0, ge=0, le=3, description="Hot flashes or cold chills")
    feeling_tired_weak_easily_exhausted: int = Field(default=0, ge=0, le=3, description="Feeling tired, weak, or easily exhausted")


class annotated_message(BaseModel):
    message: str
    physical_ratings: AnxiousPhysical
    feeling_ratings: AnxiousFeelings
    thought_rating: AnxiousThoughts

class PatientState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    annotated_messages: Annotated[list[annotated_message], operator.add]
    patient_physical_ratings:  AnxiousPhysical
    patient_thoughts_rating: AnxiousThoughts
    patient_feelings_rating: AnxiousFeelings

def extract_beliefts(anxiousbeliefs):
    context = []
    for field_name, field_info in type(anxiousbeliefs).model_fields.items():
        value = getattr(anxiousbeliefs, field_name)
        if value > 0:
            severity = ["not present", "Somewhat", "Moderatley", "A lot"][value]
            context.append(f"{field_info.description}: {severity} ({value}/3)")
    return context if context else ["No anxious thoughts detected yet"]

def ask_patient_question(state: PatientState):
    anxious_thought_context = extract_beliefts(state["patient_thoughts_rating"])
    user_input = state["messages"][-1]
    follow_up = model.invoke([SystemMessage(f"""
        As a lead pyscotherapist, ask a thoughtful empathetic question that will help you better understand the users
        anxiety using the beck anxienty inventory.

        Here is what we currently believe about their anxious thoughts:
        {anxious_thought_context}
    """), user_input])    
    
    return {"messages": [follow_up]}

def annotate_and_update(PatientState):
    user_input = PatientState["messages"][-1]
    #analyze the current thought
    thought_llm = model.with_structured_output(AnxiousThoughts)
    updated_beliefs = thought_llm.invoke([SystemMessage("""
        You are a lead psycotherapist analyzing a users thoughts, specifically you want to classify a users thoughts \
            for the following anxious thought categories derived from the burns anxious though inventory. Use the following scale.
                0 = Not at all relevant
                1 = Somewhat
                2 = Moderatley
                3 = A Lot
    """), user_input])
    return {"patient_thoughts_rating": updated_beliefs}

questioning = StateGraph(PatientState)

questioning.add_node("ask", ask_patient_question)
questioning.add_node("analyze", annotate_and_update)

questioning.add_edge(START, "analyze")
questioning.add_edge("analyze", "ask")
questioning.add_edge("ask", END)



def main():
    memory = MemorySaver()
    therapyApp = questioning.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "user-1"}}
    
    print("Welcome to Cowsel, an AI CBT agent.")
    print("Please note you are chatting with an LLM. If you are in crisis or at risk of self harm please call or text 988")
    print("\nCowsel: How are you doing today?\n")
    
    # Initial state for first invoke
    initial_state = {
        "patient_thoughts_rating": AnxiousThoughts(),
        "patient_feelings_rating": AnxiousFeelings(),
        "patient_physical_ratings": AnxiousPhysical(),
    }
    
    first_turn = True
    
    while True:
        user_input = input("You: ")
        if user_input in ["quit", "exit"]:
            break
        
        if first_turn:
            invoke_state = {
                **initial_state,
                "messages": [HumanMessage(content=user_input)]
            }
            first_turn = False
        else:
            invoke_state = {
                "messages": [HumanMessage(content=user_input)]
            }
        
        result = therapyApp.invoke(invoke_state, config)
        print(f"\nCowsel: {result['messages'][-1].content}\n")
        print(result['patient_thoughts_rating'])

if __name__ == "__main__":
    main()