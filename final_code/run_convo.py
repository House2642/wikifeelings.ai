from personas.persona_agent import persona_agent, CasePersona, WarningSigns, PersonaAgent
from fidelity_judge import fidelity_judge_app, FidelityJudgeState
from personas.personas import persona_list
from personas.therapists.baseline_v1 import base_app
from langchain_core.messages import HumanMessage
import uuid
import pandas as pd
import json

DEBUG = True

def run_conversation(persona: tuple[CasePersona, WarningSigns], therapist, max_turns: int = 10, streaming: bool=False):
    run_id = uuid.uuid4().hex[:8]

    persona_config = {"configurable": {"thread_id": f"persona-{run_id}"}}
    therapist_config = {"configurable": {"thread_id": f"therapist-{run_id}"}}
    readable_convo = []

    #Set the flag to fill out the form before
    persona_state: PersonaAgent = {
        "warning_signs": persona[1],
        "persona": persona[0],
        "flag": "initial_survey"
    }
    patient_response = persona_agent.invoke(persona_state, persona_config)
    if streaming:
        print("="*10 + "initial survery" + "="*10)
        print(f"{patient_response['intake_SAD_total']}/40")
        print("="*35)
        print("="*10 + "conversation" + "="*10)
    
    #start conversation set starting messages in initial state
    first_message = "How are you?"
    readable_convo.append(f"Therapist: {first_message}")

    patient_response = persona_agent.invoke({
        "messages": [HumanMessage("How are you?")],
        "flag": "conversation"}, persona_config)
    readable_convo.append(f"Patient: {patient_response['messages'][-1].content}")
    
    if streaming:
        print(readable_convo[-1])
    
    therapist_response = therapist.invoke({
        "messages": patient_response['messages'],
        "flag": "conversation"
    }, therapist_config)
    readable_convo.append(f"Therapist: {therapist_response['messages'][-1].content}")
    
    if streaming:
        print(readable_convo[-1])
    
    #start conversation
    for _ in range(max_turns):
        patient_response = persona_agent.invoke({
            "messages": [therapist_response['messages'][-1]],
            "flag": "conversation"}, persona_config)
        readable_convo.append(f"Patient: {patient_response['messages'][-1].content}")
        
        if streaming:
            print(readable_convo[-1])
        
        therapist_response = therapist.invoke({
            "messages": [patient_response['messages'][-1]],
            "flag": "conversation"
        }, therapist_config)
        readable_convo.append(f"Therapist: {therapist_response['messages'][-1].content}")
        
        if streaming:
            print(readable_convo[-1])
    
    patient_response = persona_agent.invoke({"flag": "exit_survey"}, persona_config)
    if streaming:
        print("="*35)
        print("="*10 + "initial survery" + "="*10)
        print(f"{patient_response['intake_SAD_total']}/40")
        print("="*35)

    therapist_response = therapist.invoke({"flag": "conceptualize_case"}, therapist_config)
    
    if streaming:    
        print("="*10 + "Case Conceptualization" + "="*10)
        print(therapist_response['case'])
        print("="*35)
    
    fidelity_judge_state: FidelityJudgeState = {
        "conversation": readable_convo
    }

    fidelity_judge_response = fidelity_judge_app.invoke(fidelity_judge_state)

    if streaming:
        print(fidelity_judge_response['rating'].display_summary())
    
    print(therapist_response['case'])
    print(readable_convo)
    print(patient_response['persona'])

def test_crisis_categorization(Therapist):

    #get the reddit cases
    with open("acute_crisis/crisis_messages.json", "r") as f:
        crisis_cases = json.load(f)

    results = []
    for case in crisis_cases["crisis_dataset"][15:25]:
        run_id = uuid.uuid4().hex[:8]
        therapist_config = {"configurable": {"thread_id": f"therapist-{run_id}"}}

        response = base_app.invoke({
            "messages":[HumanMessage(content=case["message"])],
            "flag": "crisis_categorize"
        }, therapist_config)

        results.append({
            "case_id": case["id"],
            "expected": case["crisis_type"],
            "actual": response["crisis_classification"].classification,
            "reasoning": response["crisis_classification"].reasoning
        })

    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    #run_conversation(persona_list[0], base_app, 5, True)
    test_crisis_categorization(base_app)
