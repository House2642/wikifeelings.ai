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
    convo.append(f"Patient: {patient_response['messages'][-1].content}")
    if CONVO_PRINT:
        print("Therapist: How are you?")
        print(f"Persona: {patient_response['messages'][-1].content}")

    therapist_state = therapist.get_initial_state([HumanMessage("Therapist: How are you?"), patient_response['messages'][-1]])
    
    therapist_response = therapist.app.invoke(therapist_state, therapist_config)
    convo.append(f"Therapist: {therapist_response['messages'][-1].content}")

    if CONVO_PRINT:
        print(f"Therapist: {therapist_response['messages'][-1].content}")
    
    num_rounds = 0
    while num_rounds < max_turns:
        patient_response = persona_app.invoke({"messages": [HumanMessage(therapist_response['messages'][-1].content)]}, persona_config)
        convo.append(f"Patient: {patient_response['messages'][-1].content}")
        
        if CONVO_PRINT:
            print(f"Persona: {patient_response['messages'][-1].content}")
        therapist_response = therapist.app.invoke({"messages": [HumanMessage(patient_response['messages'][-1].content)]}, therapist_config)
        convo.append(f"Therapist: {therapist_response['messages'][-1].content}")
        
        if CONVO_PRINT:
            print(f"Therapist: {therapist_response['messages'][-1].content}")
        num_rounds += 1

    
    therapist_response = therapist.app.invoke({"produce_case": True}, therapist_config)
    if DEBUG:
        print("#######conceptualization############")
        # Access directly from state, convert Pydantic model to dict
        model_conceptualization = therapist_response['case'].model_dump()
        print(model_conceptualization)
        print("##########EVALUATING#################")
        
        judge_state = {
            "conversation": ('\n\n').join(convo),
            "ground_truth": persona.model_dump() if hasattr(persona, 'model_dump') else persona,
            "estimated": model_conceptualization,
        }
        eval_result = eval_app.invoke(judge_state)
        score = eval_result["score"]
        
        print(f"ABC Adherence: {score.abc_adherence_score}/3")
        print(f"ABC Reasoning: {score.abc_adherence_reasoning_trace}")
        
        total = (score.abc_adherence_score)
        print(f"\n{'='*50}")
        print(f"TOTAL: {total}/3")
        print(f"{'='*50}")

    return {
        "run_id": run_id,
        "situation": persona["situation"],
        "abc_adherence_score": score.abc_adherence_score,
        "abc_adherence_reasoning_trace": score.abc_adherence_reasoning_trace,
        "total_score": total,
        "conceptualization": model_conceptualization,
        "conversation": convo
    }

def main():
    print(f"Running evaluation on {len(personas)} personas...")
    print(f"Started at: {datetime.now()}")
    
    results = []
    
    # Iterate through all personas with progress bar
    for idx, persona in enumerate(tqdm(personas[:2], desc="Evaluating personas")):
        try:
            print(f"\n{'='*50}")
            print(f"Persona {idx+1}/{len(personas)}: {persona['situation'][:50]}...")
            print('='*50)
            
            result = eval_run(persona=persona, therapist=baseline, max_turns=10)
            results.append(result)
            
            print(f"✓ Completed - Total Score: {result['total_score']}/9")
            
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
            "abc_adherence": r.get("abc_adherence_score", 0),
            "abc_adherence_reasoning_trace": r.get("abc_adherence_reasoning_trace", ""),
            "total_score": r.get("total_score", 0),  # ← Add this line
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
    print(df[["abc_adherence", "abc_adherence_reasoning_trace"]].describe())
    
    print(f"\nAverage scores:")
    print(f"  ABC Adherence: {df['abc_adherence'].mean():.2f}/3")
    print(f"  Total: {df['total_score'].mean():.2f}/9")
    
    print(f"\nResults saved:")
    print(f"  - eval_results_{timestamp}.json")
    print(f"  - eval_results_{timestamp}.csv")
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()