from langchain_core.messages import SystemMessage, HumanMessage
from persona_framework import persona_app, case_persona
from therapists.therapists import baseline
import uuid
from judge import eval_app, PatientStates
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from personas import personas

CONVO_PRINT = False
DEBUG = True

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
    if DEBUG:
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

    return {
        "run_id": run_id,
        "situation": persona["situation"],
        "situation_score": score.situation_score,
        "thoughts_score": score.thoughts_score,
        "meaning_of_at_score": score.meaning_of_at_score,
        "emotions_score": score.emotions_score,
        "behaviors_score": score.behaviors_score,
        "total_score": (score.situation_score + score.thoughts_score + 
                       score.meaning_of_at_score + score.emotions_score + 
                       score.behaviors_score),
        "reasoning": score.overall_reasoning,
        "conceptualization": model_conceptualization,
        "conversation": convo
    }
def main():
    print(f"Running evaluation on {len(personas)} personas...")
    print(f"Started at: {datetime.now()}")
    
    results = []
    
    # Iterate through all personas with progress bar
    for idx, persona in enumerate(tqdm(personas, desc="Evaluating personas")):
        try:
            print(f"\n{'='*50}")
            print(f"Persona {idx+1}/{len(personas)}: {persona['situation'][:50]}...")
            print('='*50)
            
            result = eval_run(persona=persona, therapist=baseline, max_turns=10)
            results.append(result)
            
            print(f"✓ Completed - Total Score: {result['total_score']}/15")
            
        except Exception as e:
            print(f"✗ Error with persona {idx+1}: {str(e)}")
            # Log error but continue
            results.append({
                "run_id": f"error-{idx}",
                "situation": persona["situation"],
                "error": str(e),
                "total_score": 0
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    with open(f"eval_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create DataFrame for analysis
    df = pd.DataFrame([
        {
            "run_id": r["run_id"],
            "situation": r["situation"][:100],
            "situation_score": r.get("situation_score", 0),
            "thoughts_score": r.get("thoughts_score", 0),
            "meaning_of_at_score": r.get("meaning_of_at_score", 0),
            "emotions_score": r.get("emotions_score", 0),
            "behaviors_score": r.get("behaviors_score", 0),
            "total_score": r.get("total_score", 0)
        }
        for r in results
    ])
    
    # Save as CSV
    df.to_csv(f"eval_results_{timestamp}.csv", index=False)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print('='*50)
    print(f"\nTotal personas evaluated: {len(results)}")
    print(f"\nScore Statistics:")
    print(df[["situation_score", "thoughts_score", "meaning_of_at_score", 
              "emotions_score", "behaviors_score", "total_score"]].describe())
    
    print(f"\nAverage scores:")
    print(f"  Situation: {df['situation_score'].mean():.2f}/3")
    print(f"  Thoughts: {df['thoughts_score'].mean():.2f}/3")
    print(f"  Core Beliefs: {df['meaning_of_at_score'].mean():.2f}/3")
    print(f"  Emotions: {df['emotions_score'].mean():.2f}/3")
    print(f"  Behaviors: {df['behaviors_score'].mean():.2f}/3")
    print(f"  Total: {df['total_score'].mean():.2f}/15")
    
    print(f"\nResults saved:")
    print(f"  - eval_results_{timestamp}.json")
    print(f"  - eval_results_{timestamp}.csv")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()