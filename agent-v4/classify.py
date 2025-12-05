from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic

DEBUG = False

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

class anxiety_rating_extractor(BaseModel):
    rating: int = Field(default=[], ge=0, le=4, description=(
                        """Composite anxiety score ranging from 
                            0 = no anxiety signal
                            1 = maybe / very mild
                            2 = mild
                            3 = moderate
                            4 = severe"""
        ),)

class analyzed_message(BaseModel):
    message: str
    rating: int

class patientState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    analyzed_messages: Annotated[list[analyzed_message], operator.add] = Field(default=[])
    #Default is hard to start for population, but then can start to narrow in for a user across messages
    anxiety_belief: float = Field(default = 0.3, ge=0, le=1, description="Probability estimate the user is actually axious or not")

#Nodes
def ask_question(patientState):
    if len(patientState.messages) == 0:
        q = "Cowsel: How are you feeling today?"
        user_input = input(f"{q}\nUser: ")
        new_messages = [SystemMessage(content=q), HumanMessage(content=user_input)]
    else:    
        response = model.invoke([SystemMessage("""You are a compassionate therapist who's goal is to understand 
        how the user is doing and what emotions they are feeling. You will only briefly respond to users inputs
        then generate a relevant question that will help you better understand them or their situation.
        """), *patientState.messages])
        user_message = input(f"Cowsel: {response.content}\nuser:")
        new_messages = [response, HumanMessage(content=user_message)]
        
    return {"messages": new_messages} 

def update_anxiety_belief(patientState):
    #prior beliefs
    #A = True Anxiety Belief, A = 0 is truly not anxious, B = 0 is truly anxious
    #B = LLM Estimated Belief
    #(P(B| A = 0))
    LIKELIHOOD_NOT_ANXIOUS = {
        0: 0.60,
        1: 0.25,
        2: 0.10,
        3: 0.04,
        4: 0.01,
    }

    #(P(B| A = 1))
    LIKELIHOOD_ANXIOUS = {
        0: 0.05,
        1: 0.15,
        2: 0.30,
        3: 0.25,
        4: 0.25,
    }
    recent_message = patientState.analyzed_messages[-1]
    rating = recent_message.rating

    #p(b | A=1) * prior
    if DEBUG:
        print("Calculating Update")
    numerator = LIKELIHOOD_ANXIOUS[rating] * patientState.anxiety_belief
    denominator = numerator + LIKELIHOOD_NOT_ANXIOUS[rating] * (1-patientState.anxiety_belief)
    posterior = numerator/denominator

    if DEBUG:
        print(f"numerator:{numerator}, denominator: {denominator}, posterior: {posterior}")
    return {"anxiety_belief": posterior}

def analyze_anxiety(patientState):
    m = patientState.messages[-1]
    q = patientState.messages[-2]
    anxiety_detector = model.with_structured_output(anxiety_rating_extractor)
    rating = anxiety_detector.invoke([SystemMessage(f"""
    Act as a lead pyschotherapist Analyze the following question, answer. If the statement seems to have no indication of an axious person, rate them a 0
    If there is some indication, rate the message on a scale of 0-5 indicating the degree to which the model detects the following criteria. 
    Here is the scale:
    Composite anxiety score ranging from 
        0 = no anxiety signal
        1 = maybe / very mild
        2 = mild
        3 = moderate
        4 = severe
    
    DSM-5 Criteria for Anxiety:
        According to DSM-5, anxiety (specifically Generalized Anxiety Disorder) is defined by:
        Excessive anxiety and worry occurring more days than not, about multiple events or activities (e.g., work, school performance).
        The person finds it difficult to control the worry.
        The anxiety and worry are associated with at least three of the following symptoms (only one required in children):
        Restlessness or feeling keyed up or on edge
        Easily fatigued
        Difficulty concentrating or mind going blank
        Irritability
        Muscle tension
        Sleep disturbance (difficulty falling or staying asleep, or restless/unsatisfying sleep)
        The anxiety causes clinically significant distress or impairment in social, occupational, or other important areas of functioning.
        The disturbance is not due to substance use, medical conditions, or better explained by another mental disorder.
    """), q, m])
    new_rating = analyzed_message(message=m.content, rating=rating.rating)
    return {"analyzed_messages": [new_rating]}

def debug_print(state: patientState):
    last_rating = state.analyzed_messages[-1] if state.analyzed_messages else None

    print("\n========== DEBUG ==========")
    if last_rating:
        print(f"Anxiety rating (0–4): {last_rating.rating}")
    print(f"Current P(user is truly anxious): {state.anxiety_belief:.3f}")
    print("===========================\n")

    # Do NOT modify state in a debug node
    return {}

#Create Graph
situation = StateGraph(patientState)
situation.add_node("askQ", ask_question)
situation.add_node("analyze", analyze_anxiety)
situation.add_node("update_belief", update_anxiety_belief)
situation.add_node("debug", debug_print)

situation.add_edge(START, "askQ")
situation.add_edge("askQ", "analyze")
situation.add_edge("analyze", "update_belief")
situation.add_edge("update_belief", "debug")
situation.add_edge("debug", "askQ")

chain = situation.compile()
initial_state=patientState()
final_state = chain.invoke(initial_state)
print(final_state)