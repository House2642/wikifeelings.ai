from langchain_core.messages import SystemMessage, HumanMessage
from risk_persona import persona_app, build_prompt, persona
from baseline_v1 import base_app
from judge import judge_app
import uuid
import pandas as pd
from tqdm import tqdm
from test_cases import TEST_CASES

CONVO_PRINT = False

def eval_run(persona: persona, max_turns: int = 10):
    run_id = uuid.uuid4().hex[:8]

    persona_config = {"configurable": {"thread_id": f"persona-{run_id}"}}
    therapist_config = {"configurable": {"thread_id": f"therapist-{run_id}"}}
    convo = ["Therapist: How are you?"] 
    system = build_prompt(persona)

    persona_state = {
        "system_message": SystemMessage(system),
        "messages": [HumanMessage("Therapist: How are you today?")]
    }
    
    patient_response = persona_app.invoke(persona_state, persona_config)
    convo.append(patient_response['messages'][-1].content)
    if CONVO_PRINT:
        print("Therapist: How are you?")
        print(f"Persona: {patient_response['messages'][-1].content}")

    therapist_state = {
        "messages": [HumanMessage("Therapist: How are you?"), patient_response['messages'][-1]],
        "reasoning_traces": []
    }
    
    therapist_response = base_app.invoke(therapist_state, therapist_config)
    convo.append(therapist_response['messages'][-1].content)

    if CONVO_PRINT:
        print(f"Therapist: {therapist_response['messages'][-1].content}")
    
    num_rounds = 0
    while num_rounds < max_turns:
        patient_response = persona_app.invoke({"messages": [HumanMessage(therapist_response['messages'][-1].content)]}, persona_config)
        convo.append(patient_response['messages'][-1].content)
        
        if CONVO_PRINT:
            print(f"Persona: {patient_response['messages'][-1].content}")
        therapist_response = base_app.invoke({"messages": [HumanMessage(patient_response['messages'][-1].content)]}, therapist_config)
        convo.append(therapist_response['messages'][-1].content)
        
        if CONVO_PRINT:
            print(f"Therapist: {therapist_response['messages'][-1].content}")
        num_rounds += 1
    
    judge_state = {
        "conversation": '\n'.join(convo)
    }

    evaluation = judge_app.invoke(judge_state)

    return {
        "eval_results": evaluation["eval_results"],
        "conversation": convo
    }

def display_results(results: list[dict]):
    rows = []
    for i, r in enumerate(results):
        gt = r["ground_truth"]
        scores = r["eval_results"]
        rows.append({
            "case": i+1,
            "ideation": gt["ideation"],
            "plan": gt["plan"],
            "intent": gt["intent"],
            "behavior": gt["behavior"], 
            "honesty": gt["honesty"],
            "expressiveness": gt["expressiveness"],

            #llm as judge
            "signal_detection": scores["signal_detection"],
            "risk_assessment": scores["risk_assessment"],
            "harm_avoidance": scores["harm_avoidance"],
            "therapeutic_alliance": scores["therapeutic_alliance"],
            "appropriate_escalation": scores["appropriate_escalation"]
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

def main():
    
    results = []
    for i, persona in enumerate(tqdm(TEST_CASES[:2], desc="Running evals")):
        tqdm.write(f"Case {i+1}: ideation={persona['ideation']}, expr={persona['expressiveness']}")
        result = eval_run(persona, max_turns=10)
        result["ground_truth"] = persona
        results.append(result)
    
    df = display_results(results)
    
    final_score = df[["signal_detection", "risk_assessment", "harm_avoidance", "therapeutic_alliance", "appropriate_escalation"]].mean().sum()
    print(final_score)
if __name__ == "__main__":
    main()