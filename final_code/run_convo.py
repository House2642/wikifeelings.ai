from personas.persona_agent import persona_agent, CasePersona, WarningSigns, PersonaAgent
from fidelity_judge import fidelity_judge_app, FidelityJudgeState
from crisis_judge import crisis_judge_app, CrisisJudgeState
from case_concept_judge import case_judge_app, CaseJudgeState
from crisis_patient import simulate_crisis_patient
from monitor_agent.monitor_agent import monitor_app
from personas.personas import persona_list
from personas.therapists.baseline_v1 import base_app
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import pandas as pd
import json
import csv
from tqdm import tqdm
from datetime import datetime

DEBUG = True

def run_conversation(run_id: int, persona_id: int, persona: tuple[CasePersona, WarningSigns],therapist, max_turns: int = 10, streaming: bool=False):
    instance_id = uuid.uuid4().hex[:8]

    persona_config = {"configurable": {"thread_id": f"persona-{instance_id}"}}
    therapist_config = {"configurable": {"thread_id": f"therapist-{instance_id}"}}
    readable_convo = []
    therapist_reasoning_traces = []

    #start conversation set starting messages in initial state
    first_message = "How are you feeling today?"
    readable_convo.append(f"Therapist: {first_message}")

    patient_response = persona_agent.invoke({
        "messages": [HumanMessage("How are you?")],
        "persona": persona[0],
        "flag": "conversation"}, persona_config)
    readable_convo.append(f"Patient: {patient_response['messages'][-1].content}")
    
    if streaming:
        print(readable_convo[-1])
    
    therapist_response = therapist.invoke({
    "messages": [AIMessage(first_message), HumanMessage(content=patient_response['messages'][-1].content)],
    "flag": "conversation"
    }, therapist_config)
    readable_convo.append(f"Therapist: {therapist_response['messages'][-1].content}")
    therapist_reasoning_traces.extend(therapist_response.get("reasoning_traces", []))
    
    if streaming:
        print(readable_convo[-1])
    
    #start conversation
    for _ in range(max_turns):
        patient_response = persona_agent.invoke({
            "messages": [HumanMessage(content=therapist_response['messages'][-1].content)],
            "flag": "conversation"
        }, persona_config)

        readable_convo.append(f"Patient: {patient_response['messages'][-1].content}")
        
        if streaming:
            print(readable_convo[-1])
        
        therapist_response = therapist.invoke({
            "messages": [HumanMessage(content=patient_response['messages'][-1].content)],
            "flag": "conversation"
        }, therapist_config)
        readable_convo.append(f"Therapist: {therapist_response['messages'][-1].content}")
        therapist_reasoning_traces.extend(therapist_response.get("reasoning_traces", []))
        
        if streaming:
            print(readable_convo[-1])
    

    therapist_response = therapist.invoke({"flag": "conceptualize_case"}, therapist_config)
    
    if streaming:    
        print("="*10 + "Case Conceptualization" + "="*10)
        print(therapist_response['case'])
        print("="*35)
    
    #conversation
    write_row("runs/conversations.csv", {
        "run_id": run_id,
        "persona_id": persona_id,
        "readable_conversation": readable_convo,
        "persona_reasoning_trace": patient_response["CoT_traces"],
        "therapist_reasoning_traces": therapist_reasoning_traces,
    })
    
    #Judge the outputs
    #judge the fidelity
    fidelity_judge_state: FidelityJudgeState = {
        "conversation": readable_convo
    }

    fidelity_judge_response = fidelity_judge_app.invoke(fidelity_judge_state)

    if streaming:
        print(fidelity_judge_response['rating'].display_summary())
    
    #case_conceptualization Judge
    concept_eval = {
        "conversation": readable_convo, 
        "ground_truth_case": patient_response['persona'].model_dump(),
        "conceptualized_case": therapist_response['case'].model_dump()
    }
    case_judge_response = case_judge_app.invoke(concept_eval)

    write_row("runs/fidelity.csv", {
        "run_id": run_id,
        "persona_id": persona_id,
        "ctrs_total": fidelity_judge_response['rating'].total_score,
        "ctrs_part_i": fidelity_judge_response['rating'].part_i_score,
        "ctrs_part_ii": fidelity_judge_response['rating'].part_ii_score,
        "agenda": fidelity_judge_response['rating'].agenda,
        "agenda_reasoning": fidelity_judge_response['rating'].agenda_reasoning,
        "feedback": fidelity_judge_response['rating'].feedback,
        "feedback_reasoning": fidelity_judge_response['rating'].feedback_reasoning,
        "understanding": fidelity_judge_response['rating'].understanding,
        "understanding_reasoning": fidelity_judge_response['rating'].understanding_reasoning,
        "interpersonal_effectiveness": fidelity_judge_response['rating'].interpersonal_effectiveness,
        "interpersonal_effectiveness_reasoning": fidelity_judge_response['rating'].interpersonal_effectiveness_reasoning,
        "collaboration": fidelity_judge_response['rating'].collaboration,
        "collaboration_reasoning": fidelity_judge_response['rating'].collaboration_reasoning,
        "pacing_and_time_use": fidelity_judge_response['rating'].pacing_and_time_use,
        "pacing_and_time_use_reasoning": fidelity_judge_response['rating'].pacing_and_time_use_reasoning,
        "guided_discovery": fidelity_judge_response['rating'].guided_discovery,
        "guided_discovery_reasoning": fidelity_judge_response['rating'].guided_discovery_reasoning,
        "focusing_on_key_cognitions": fidelity_judge_response['rating'].focusing_on_key_cognitions,
        "focusing_on_key_cognitions_reasoning": fidelity_judge_response['rating'].focusing_on_key_cognitions_reasoning,
        "strategy_for_change": fidelity_judge_response['rating'].strategy_for_change,
        "strategy_for_change_reasoning": fidelity_judge_response['rating'].strategy_for_change_reasoning,
        "application_of_techniques": fidelity_judge_response['rating'].application_of_techniques,
        "application_of_techniques_reasoning": fidelity_judge_response['rating'].application_of_techniques_reasoning,
        "homework": fidelity_judge_response['rating'].homework,
        "homework_reasoning": fidelity_judge_response['rating'].homework_reasoning,
    })

    #case_conceptualization
    write_row("runs/conceptualization.csv", {
        "run_id": run_id,
        "persona_id": persona_id,
        "ground_truth_distortions": patient_response['persona'].cognitive_distortions,
        "conceptualized_distortions": therapist_response['case'].cognitive_distortions,
        "distorition_jaccard": jaccard_similarity(patient_response['persona'].cognitive_distortions, therapist_response['case'].cognitive_distortions),
        "concept_total": case_judge_response['rating'].total,
        "situation": case_judge_response['rating'].situation,
        "situation_reasoning": case_judge_response['rating'].situation_reasoning,
        "automatic_thoughts": case_judge_response['rating'].automatic_thoughts,
        "automatic_thoughts_reasoning": case_judge_response['rating'].automatic_thoughts_reasoning,
        "cognitive_distortions": case_judge_response['rating'].cognitive_distortions,
        "cognitive_distortions_reasoning": case_judge_response['rating'].cognitive_distortions_reasoning,
        "emotions": case_judge_response['rating'].emotions,
        "emotions_reasoning": case_judge_response['rating'].emotions_reasoning,
        "behaviors": case_judge_response['rating'].behaviors,
        "behaviors_reasoning": case_judge_response['rating'].behaviors_reasoning,
    })

def jaccard_similarity(ground_truth: list[str], predicted: list[str]) -> float:
    gt_set = set(ground_truth)
    pred_set = set(predicted)
    
    intersection = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    
    return intersection / union if union > 0 else 1.0


def test_crisis_categorization(run_id: int, therapist, max_crisis_turns: int = 3):
    """
    Multi-turn crisis evaluation.

    For each case classified as harm_to_self, runs exactly max_crisis_turns
    therapist turns interleaved with simulated patient replies.  All turns are
    always completed so the judge sees the full transcript and can credit each
    protocol step (ASSESS, DE-ESCALATE, RECOMMEND, REQUEST_HUMAN_CONSULTATION)
    wherever it appears across the conversation.
    """
    with open("acute_crisis/crisis_messages.json", "r") as f:
        crisis_cases = json.load(f)

    for case in tqdm(crisis_cases["crisis_dataset"], desc="Running crisis evals"):
        instance_id = uuid.uuid4().hex[:8]
        therapist_config = {"configurable": {"thread_id": f"therapist-{instance_id}"}}

        classification = therapist.invoke({
            "messages": [HumanMessage(content=case["message"])],
            "flag": "crisis_categorize"
        }, therapist_config)

        if classification["crisis_classification"].classification == "harm_to_self":
            readable_convo = [f"Patient: {case['message']}"]

            if DEBUG:
                print(f"\n{'='*60}")
                print(f"Case {case['id']}")
                print(f"{'='*60}")
                print(f"Patient: {case['message']}\n")

            for turn in range(max_crisis_turns):
                # Therapist turn
                last_patient_msg = readable_convo[-1][len("Patient: "):]
                therapist_response = therapist.invoke({
                    "messages": [HumanMessage(content=last_patient_msg)],
                    "flag": "conversation"
                }, therapist_config)
                therapist_msg = therapist_response["messages"][-1].content
                readable_convo.append(f"Therapist: {therapist_msg}")

                if DEBUG:
                    print(f"Therapist: {therapist_msg}\n")

                # Patient turn (skip after the last therapist turn)
                if turn < max_crisis_turns - 1:
                    patient_reply = simulate_crisis_patient(case["message"], readable_convo)
                    readable_convo.append(f"Patient: {patient_reply}")

                    if DEBUG:
                        print(f"Patient: {patient_reply}\n")

            evaluation = crisis_judge_app.invoke({
                "conversation": readable_convo
            })
            write_row("runs/crisis_conversations.csv", {
                "run_id": run_id,
                "case_id": case["id"],
                "readable_conversation": readable_convo,
            })
            write_row("runs/crisis.csv", {
                "run_id": run_id,
                "case_id": case["id"],
                "expected": case["crisis_type"],
                "actual": classification["crisis_classification"].classification,
                "reasoning": classification["crisis_classification"].reasoning,
                "crisis_conversation": readable_convo,
                "eval_score": evaluation["total"],
                "assess": evaluation["rating"].assess,
                "assess_reasoning": evaluation["rating"].assess_reasoning,
                "de_escalate": evaluation["rating"].de_escalate,
                "de_escalate_reasoning": evaluation["rating"].de_escalate_reasoning,
                "recommend_emergency_services": evaluation["rating"].recommend_emergency_services,
                "recommend_emergency_services_reasoning": evaluation["rating"].recommend_emergency_services_reasoning,
                "request_human_consultation": evaluation["rating"].request_human_consultation,
                "request_human_consultation_reasoning": evaluation["rating"].request_human_consultation_reasoning
            })
        else:
            write_row("runs/crisis.csv", {
                "run_id": run_id,
                "case_id": case["id"],
                "expected": case["crisis_type"],
                "actual": classification["crisis_classification"].classification,
                "reasoning": classification["crisis_classification"].reasoning,
            })

def write_row(filename: str, row: dict):
    cleaned = {k: v.replace('\n', ' ').strip() if isinstance(v, str) else v for k, v in row.items()}
    with open(filename, "r") as f:
        fieldnames = f.readline().strip().split(",")
    
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writerow(cleaned)

if __name__ == "__main__":
    run_id = uuid.uuid4().hex[:8]
    num_rounds = 10
    therapist = "monitor_agent - Haiku"
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_row("runs/run_meta_data.csv", {
        "run_id": run_id,
        "num_rounds": num_rounds,
        "therapist": therapist,
        "run_date_time": date_time,
    })

    #for i, persona in tqdm(enumerate(persona_list), desc="evaluating personas"):
        #run_conversation(run_id, i, persona, monitor_app, 20, False)
    test_crisis_categorization(run_id, monitor_app)
