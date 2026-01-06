#!/usr/bin/env python3
"""
CBT Therapy Chatbot MVP

A verifiable CBT chatbot that builds a cognitive model of the user
and intervenes using Socratic questioning.

Architecture:
    User Input → LLM Extraction → State Update → Policy Decision → Response Generation → Validation → Output

The LLM is a sensor, not a decision-maker. All policy decisions are formal Python code.
"""

import os
from langchain_anthropic import ChatAnthropic

from schemas import CBTState, Extraction
from extraction import extract_from_message
from policy import decide_action, update_state
from response import generate_response
from validation import validate_output
from constants import CRISIS_RESPONSE


def print_debug(state: CBTState, extraction: Extraction, action: str):
    """Print debug information about the current state."""
    print("\n" + "="*70)
    print("DEBUG INFO")
    print("="*70)
    print(f"Extraction:")
    print(f"  Situation: {extraction.situation}")
    print(f"  Thought: {extraction.thought}")
    print(f"  Feeling: {extraction.feeling} @ {extraction.intensity}/10")
    print(f"  Distortions: {extraction.distortions}")
    print(f"  Behaviors: {extraction.behaviors}")
    print(f"  Crisis Level: {extraction.crisis_level}")
    print(f"\nCurrent State:")
    print(f"  Situation: {state.situation} (has: {state.has_situation})")
    print(f"  Thought: {state.thought} (has: {state.has_thought})")
    print(f"  Feeling: {state.feeling} @ {state.intensity}/10 (has: {state.has_feeling})")
    print(f"  Distortions: {state.distortions}")
    print(f"  Behaviors: {state.behaviors}")
    print(f"  Crisis Level: {state.crisis_level}")
    print(f"\nAction Decided: {action}")
    print("="*70 + "\n")


def run_conversation(debug: bool = False):
    """
    Main conversation loop.

    Args:
        debug: If True, print debug information after each turn
    """

    # Initialize the language model
    # Using Claude Haiku for efficiency (as specified in the doc)
    model = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.7)

    # Create extractor with structured output
    extractor = model.with_structured_output(Extraction)

    # Initialize state
    state = CBTState()

    print("\n" + "="*70)
    print("CBT THERAPY CHATBOT - MVP")
    print("="*70)
    print("\nTherapist: Hi, I'm here to help. How are you doing today?")
    print("\n(Type 'quit', 'exit', or 'bye' to end the conversation)")
    print("="*70 + "\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nTherapist: Take care of yourself. I'm here whenever you need to talk.\n")
            break

        try:
            # ==========================================
            # STEP 1: EXTRACT - LLM parses into structured format
            # ==========================================
            extraction = extract_from_message(extractor, user_input)

            # ==========================================
            # STEP 2: UPDATE STATE - Pure Python, no LLM
            # ==========================================
            state = update_state(state, extraction, user_input)

            # ==========================================
            # STEP 3: DECIDE ACTION - Pure Python, formal rules
            # ==========================================
            action = decide_action(state)
            state.current_action = action

            # ==========================================
            # STEP 4: GENERATE RESPONSE
            # ==========================================
            if action == "CRISIS_RESPONSE":
                # Crisis responses are HARDCODED, never LLM-generated
                response = CRISIS_RESPONSE
            else:
                # Use LLM to generate response within constraints
                response = generate_response(model, state, action, user_input)

            # ==========================================
            # STEP 5: VALIDATE OUTPUT - Hard safety gate
            # ==========================================
            is_valid, validated_response = validate_output(response, action, state)

            # ==========================================
            # STEP 6: Update state with assistant message and output
            # ==========================================
            state = update_state(state, Extraction(), "", validated_response)

            print(f"\nTherapist: {validated_response}\n")

            # Debug output (optional)
            if debug:
                print_debug(state, extraction, action)

        except KeyboardInterrupt:
            print("\n\nTherapist: Take care of yourself. I'm here whenever you need to talk.\n")
            break
        except Exception as e:
            print(f"\n[ERROR] Something went wrong: {e}")
            print("Therapist: I'm sorry, I had trouble processing that. Could you try again?\n")
            if debug:
                import traceback
                traceback.print_exc()


def main():
    """Entry point for the chatbot."""
    import argparse

    parser = argparse.ArgumentParser(description="CBT Therapy Chatbot MVP")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing extraction and state"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[ERROR] ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'\n")
        return

    run_conversation(debug=args.debug)


if __name__ == "__main__":
    main()
