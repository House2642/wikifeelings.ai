"""
Generates diagrams for the evaluation flow using LangGraph's built-in Mermaid renderer.

Usage:
    cd final_code
    python evaluation_flow_diagram.py

Outputs (written to ./diagrams/):
    fidelity_judge_graph.mmd    — Mermaid source for the Fidelity Judge sub-graph
    case_judge_graph.mmd        — Mermaid source for the Case Conceptualization Judge sub-graph
    crisis_judge_graph.mmd      — Mermaid source for the Crisis Judge sub-graph
    evaluation_flow.mmd         — Full end-to-end evaluation pipeline diagram

If the `mermaid-py` / Mermaid CLI is available the script also saves .png files.
"""

import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Individual sub-graph diagrams via LangGraph ────────────────────────────

def save_subgraph_diagrams():
    """Import compiled LangGraph apps and dump their Mermaid representations."""
    from fidelity_judge import fidelity_judge_app
    from crisis_judge import crisis_judge_app
    from case_concept_judge import case_judge_app

    graphs = {
        "fidelity_judge_graph": fidelity_judge_app,
        "case_judge_graph": case_judge_app,
        "crisis_judge_graph": crisis_judge_app,
    }

    for name, app in graphs.items():
        mermaid_text = app.get_graph().draw_mermaid()
        mmd_path = os.path.join(OUTPUT_DIR, f"{name}.mmd")
        with open(mmd_path, "w") as f:
            f.write(mermaid_text)
        print(f"  Wrote {mmd_path}")

        # Optionally save PNG (uses Mermaid.ink — requires internet access)
        try:
            png_bytes = app.get_graph().draw_mermaid_png()
            png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
            with open(png_path, "wb") as f:
                f.write(png_bytes)
            print(f"  Wrote {png_path}")
        except Exception as e:
            print(f"  PNG skipped for {name}: {e}")


# ── 2. Overall evaluation pipeline diagram ────────────────────────────────────

OVERALL_FLOW_MERMAID = """\
%%{ init: { "theme": "default", "flowchart": { "curve": "basis" } } }%%
flowchart TD

    %% ── Initialization ───────────────────────────────────────────────────
    START([▶ START]) --> INIT["Initialize Run\\nrun_id · persona_list · therapist"]
    INIT --> PERSONA_LOOP{For each\\npersona}

    %% ── Conversation Loop ────────────────────────────────────────────────
    PERSONA_LOOP --> T0["Therapist opens\\n'How are you feeling today?'"]
    T0 --> P1["Persona Agent\\nflag = conversation"]
    P1 --> T1["Therapist Agent\\nflag = conversation"]
    T1 --> TURN_CHECK{"max_turns\\nreached?"}
    TURN_CHECK -- No --> P1
    TURN_CHECK -- Yes --> CONCEPTUALIZE["Therapist → Conceptualize Case\\nflag = conceptualize_case"]

    %% ── Persist raw conversation ─────────────────────────────────────────
    CONCEPTUALIZE --> SAVE_CONVO[("💾 conversations.csv")]

    %% ── Evaluation branch (runs after conversation) ──────────────────────
    CONCEPTUALIZE --> FID_JUDGE
    CONCEPTUALIZE --> CASE_JUDGE

    subgraph FID_JUDGE ["Fidelity Judge  (CTRS)"]
        direction TB
        FID_IN["conversation"] --> FID_LLM["claude-opus-4-5\\nCTRS Rating"]
        FID_LLM --> FID_OUT["Scores 0–6 each\\nagenda · feedback · understanding\\ninterpersonal effectiveness\\ncollaboration · pacing & time use\\nguided discovery · key cognitions\\nstrategy for change\\napplication of techniques · homework"]
    end

    subgraph CASE_JUDGE ["Case Conceptualization Judge"]
        direction TB
        CASE_IN["conversation\\n+ ground truth\\n+ therapist case"] --> CASE_LLM["claude-opus-4-5\\nConceptualization Rating"]
        CASE_LLM --> CASE_OUT["Scores 0–2 each\\nsituation · automatic thoughts\\ncognitive distortions\\nemotions · behaviors\\n+ Jaccard similarity (distortions)"]
    end

    FID_OUT  --> SAVE_FID[("💾 fidelity.csv")]
    CASE_OUT --> SAVE_CASE[("💾 conceptualization.csv")]

    SAVE_FID  --> PERSONA_LOOP
    SAVE_CASE --> PERSONA_LOOP

    %% ── Crisis Evaluation (separate entry point) ─────────────────────────
    CRISIS_START([▶ Crisis Eval START]) --> LOAD["Load crisis_messages.json"]
    LOAD --> CRISIS_LOOP{For each\\ncrisis case}

    CRISIS_LOOP --> CLASSIFY["Therapist: Classify Message\\nflag = crisis_categorize"]
    CLASSIFY --> HARM_CHECK{"classification ==\\nharm_to_self?"}

    HARM_CHECK -- No --> LOG_CLASS[("💾 crisis.csv\\nclassification only")]
    LOG_CLASS --> CRISIS_LOOP

    HARM_CHECK -- Yes --> CRISIS_CONV["Multi-turn Crisis Conversation\\nmax_crisis_turns = 3\\ntherapist ↔ simulated patient"]
    CRISIS_CONV --> CRISIS_JUDGE

    subgraph CRISIS_JUDGE ["Crisis Judge  (Acute Protocol)"]
        direction TB
        CJ_IN["crisis conversation"] --> CJ_LLM["claude-opus-4-5\\nAcute Crisis Rating"]
        CJ_LLM --> CJ_OUT["Checklist (bool each)\\nassess · de-escalate\\nrecommend emergency services\\nrequest human consultation\\n[REQUEST_HUMAN_CONSULTATION]"]
    end

    CJ_OUT --> SAVE_CRISIS[("💾 crisis.csv\\nfull evaluation")]
    SAVE_CRISIS --> CRISIS_LOOP

    %% ── Styling ──────────────────────────────────────────────────────────
    classDef store   fill:#ddeeff,stroke:#3377bb,color:#000
    classDef judge   fill:#fff8dc,stroke:#cc9900,color:#000
    classDef agent   fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef ctrl    fill:#fce4ec,stroke:#c62828,color:#000

    class SAVE_CONVO,SAVE_FID,SAVE_CASE,LOG_CLASS,SAVE_CRISIS store
    class FID_LLM,CASE_LLM,CJ_LLM judge
    class P1,T0,T1,CONCEPTUALIZE,CLASSIFY,CRISIS_CONV agent
    class TURN_CHECK,HARM_CHECK,PERSONA_LOOP,CRISIS_LOOP ctrl
"""


def save_overall_diagram():
    path = os.path.join(OUTPUT_DIR, "evaluation_flow.mmd")
    with open(path, "w") as f:
        f.write(OVERALL_FLOW_MERMAID)
    print(f"  Wrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Saving overall evaluation flow diagram...")
    save_overall_diagram()

    print("Generating sub-graph diagrams from LangGraph...")
    try:
        save_subgraph_diagrams()
    except Exception as e:
        print(f"  Sub-graph generation failed (check env/API key): {e}")

    print("\nDone. Files written to:", OUTPUT_DIR)
