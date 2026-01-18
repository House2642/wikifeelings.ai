from langchain_core.messages import SystemMessage, HumanMessage
from persona_framework import persona_app, case_persona
from therapists.therapists import baseline
import uuid
from judge import eval_app, PatientStates
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

CONVO_PRINT = True

def eval_run(persona: case_persona, therapist, max_turns: int = 10):
    run_id = uuid.uuid4().hex[:8]

    persona_config = {"configurable": {"thread_id": f"persona-{run_id}"}}
    therapist_config = {"configurable": {"thread_id": f"therapist-{run_id}"}}
    convo = ["Therapist: How are you?"]

    persona_state = {
        "persona": persona,
        "messages": [HumanMessage("Therapist: How are you today?")]
    }
    
    patient_response = persona_app.invoke(persona_state, persona_config)
    convo.append(patient_response['messages'][-1].content)
    if CONVO_PRINT:
        print("Therapist: How are you?")
        print(f"Persona: {patient_response['messages'][-1].content}")

    therapist_state = therapist.get_initial_state([HumanMessage("Therapist: How are you?"), patient_response['messages'][-1]])
    
    therapist_response = therapist.app.invoke(therapist_state, therapist_config)
    convo.append(therapist_response['messages'][-1].content)

    if CONVO_PRINT:
        print(f"Therapist: {therapist_response['messages'][-1].content}")
    
    num_rounds = 0
    while num_rounds < max_turns:
        patient_response = persona_app.invoke({"messages": [HumanMessage(therapist_response['messages'][-1].content)]}, persona_config)
        convo.append(patient_response['messages'][-1].content)
        
        if CONVO_PRINT:
            print(f"Persona: {patient_response['messages'][-1].content}")
        therapist_response = therapist.app.invoke({"messages": [HumanMessage(patient_response['messages'][-1].content)]}, therapist_config)
        convo.append(therapist_response['messages'][-1].content)
        
        if CONVO_PRINT:
            print(f"Therapist: {therapist_response['messages'][-1].content}")
        num_rounds += 1

    hum_mes = """Based on your conversation fill out the following typed dict base on how you have thought about the patient.
                You can have things be empty if not detected, but give you best estimate
                class case_persona(TypedDict):
                    situation: str
                    thoughts: list[str]
                    meaning_of_at: list[str]
                    emotions: list[str]
                    behaviors: list[str]
                output as only a json
    """
    therapist_response = therapist.app.invoke({"messages":[HumanMessage(content = hum_mes)]}, therapist_config)
    print("#######conceptualization############")
    model_conceptualization = json.loads(therapist_response['messages'][-1].content)
    print(model_conceptualization)
    print("##########EVALUATING#################")
    judge_state = {
        "ground_truth": persona,
        "estimated": model_conceptualization,
    }
    eval = eval_app.invoke(judge_state)
    score = eval["score"]

    print(f"Situation: {score.situation_score}/3")
    print(f"Thoughts: {score.thoughts_score}/3")
    print(f"Core Beliefs: {score.meaning_of_at_score}/3")
    print(f"Emotions: {score.emotions_score}/3")
    print(f"Behaviors: {score.behaviors_score}/3")
    print(f"\nReasoning: {score.overall_reasoning}")

    # Calculate total
    total = (score.situation_score + score.thoughts_score + 
            score.meaning_of_at_score + score.emotions_score + 
            score.behaviors_score)
    print(f"Total: {total}/15")

def main():
    case_persona_example: case_persona = {
        "situation": "Matched with someone on a dating app. They viewed my profile but haven't responded to my message for 3 days.",
        
        "thoughts": [
            "They probably think I'm boring",
            "I must have said something wrong in my message",
            "They're talking to someone more attractive",
            "I'm never going to find anyone"
        ],
        
        "meaning_of_at": [
            "I'm not interesting enough for people to like me",
            "There's something fundamentally wrong with me",
            "I'll always be alone because I'm defective"
        ],
        
        "emotions": [
            "anxious",
            "rejected", 
            "hopeless",
            "worthless"
        ],
        
        "behaviors": [
            "Checking the app every 15 minutes",
            "Re-reading my message looking for mistakes",
            "Avoiding sending messages to other matches",
            "Scrolling through their profile repeatedly",
            "Considering deleting the app entirely"
        ]
    }
    eval_run(persona=case_persona_example, therapist=baseline, max_turns=10)

if __name__ == "__main__":
    main()