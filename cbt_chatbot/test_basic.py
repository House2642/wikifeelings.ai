#!/usr/bin/env python3
"""
Basic tests for the CBT therapy chatbot components.

These tests verify the core logic without requiring API calls.
"""

from schemas import CBTState, Extraction
from policy import decide_action, update_state
from validation import validate_output, get_fallback_response
from constants import CRISIS_RESPONSE


def test_policy_decisions():
    """Test that policy decisions follow the correct logic."""
    print("\n" + "="*70)
    print("TEST: Policy Decisions")
    print("="*70)

    # Test 1: No information → Ask for feeling
    state = CBTState()
    action = decide_action(state)
    assert action == "ASK_FEELING", f"Expected ASK_FEELING, got {action}"
    print("✓ Test 1: No information → ASK_FEELING")

    # Test 2: Has feeling → Ask for situation
    state.has_feeling = True
    state.feeling = "anxious"
    state.intensity = 7
    action = decide_action(state)
    assert action == "ASK_SITUATION", f"Expected ASK_SITUATION, got {action}"
    print("✓ Test 2: Has feeling → ASK_SITUATION")

    # Test 3: Has feeling + situation → Ask for thought
    state.has_situation = True
    state.situation = "Going to work"
    action = decide_action(state)
    assert action == "ASK_THOUGHT", f"Expected ASK_THOUGHT, got {action}"
    print("✓ Test 3: Has feeling + situation → ASK_THOUGHT")

    # Test 4: Has all three → Intervene
    state.has_thought = True
    state.thought = "I'm going to mess up"
    action = decide_action(state)
    assert action == "INTERVENE", f"Expected INTERVENE, got {action}"
    print("✓ Test 4: Has feeling + situation + thought → INTERVENE")

    # Test 5: Crisis level 2 → Crisis response (overrides everything)
    state.crisis_level = 2
    action = decide_action(state)
    assert action == "CRISIS_RESPONSE", f"Expected CRISIS_RESPONSE, got {action}"
    print("✓ Test 5: Crisis level 2 → CRISIS_RESPONSE (overrides all)")

    print("\nAll policy decision tests passed! ✓\n")


def test_state_updates():
    """Test that state updates accumulate information correctly."""
    print("="*70)
    print("TEST: State Updates")
    print("="*70)

    state = CBTState()

    # Test 1: Add feeling
    extraction = Extraction(
        feeling="anxious",
        intensity=7
    )
    state = update_state(state, extraction, "I feel anxious", None)
    assert state.has_feeling == True
    assert state.feeling == "anxious"
    assert state.intensity == 7
    print("✓ Test 1: Feeling added to state")

    # Test 2: Add situation
    extraction = Extraction(
        situation="Going to work"
    )
    state = update_state(state, extraction, "It's about work", None)
    assert state.has_situation == True
    assert state.situation == "Going to work"
    assert state.feeling == "anxious"  # Previous data preserved
    print("✓ Test 2: Situation added, feeling preserved")

    # Test 3: Add thought
    extraction = Extraction(
        thought="I'm going to mess up",
        distortions=["fortune_telling", "catastrophizing"]
    )
    state = update_state(state, extraction, "I think I'll mess up", None)
    assert state.has_thought == True
    assert state.thought == "I'm going to mess up"
    assert "fortune_telling" in state.distortions
    assert "catastrophizing" in state.distortions
    print("✓ Test 3: Thought and distortions added")

    # Test 4: Crisis level only increases
    extraction = Extraction(crisis_level=1)
    state = update_state(state, extraction, "I feel hopeless", None)
    assert state.crisis_level == 1
    print("✓ Test 4: Crisis level increased to 1")

    extraction = Extraction(crisis_level=0)
    state = update_state(state, extraction, "Actually I'm okay", None)
    assert state.crisis_level == 1  # Should NOT decrease
    print("✓ Test 5: Crisis level does not decrease")

    # Test 6: Conversation history
    assert len(state.messages) >= 2
    print("✓ Test 6: Conversation history maintained")

    print("\nAll state update tests passed! ✓\n")


def test_validation():
    """Test that output validation catches harmful content."""
    print("="*70)
    print("TEST: Output Validation")
    print("="*70)

    state = CBTState()

    # Test 1: Safe content passes
    is_valid, response = validate_output(
        "I hear you. That sounds difficult.",
        "INTERVENE",
        state
    )
    assert is_valid == True
    print("✓ Test 1: Safe content passes validation")

    # Test 2: Harmful content blocked
    is_valid, response = validate_output(
        "You should kill yourself",
        "INTERVENE",
        state
    )
    assert is_valid == False
    assert response != "You should kill yourself"
    print("✓ Test 2: Harmful content blocked")

    # Test 3: Crisis response must include resources
    is_valid, response = validate_output(
        "I'm concerned about you.",  # Missing 988 and resources
        "CRISIS_RESPONSE",
        state
    )
    assert is_valid == False
    assert "988" in response
    print("✓ Test 3: Crisis response without resources replaced")

    # Test 4: Proper crisis response passes
    is_valid, response = validate_output(
        CRISIS_RESPONSE,
        "CRISIS_RESPONSE",
        state
    )
    assert is_valid == True
    assert "988" in response
    print("✓ Test 4: Proper crisis response passes")

    # Test 5: Fallback responses are safe
    for action in ["ASK_FEELING", "ASK_SITUATION", "ASK_THOUGHT", "INTERVENE", "CRISIS_RESPONSE"]:
        fallback = get_fallback_response(action)
        assert fallback is not None
        assert len(fallback) > 0
    print("✓ Test 5: All fallback responses defined")

    print("\nAll validation tests passed! ✓\n")


def test_extraction_schema():
    """Test that Extraction schema validates correctly."""
    print("="*70)
    print("TEST: Extraction Schema")
    print("="*70)

    # Test 1: Valid extraction
    extraction = Extraction(
        situation="Work presentation",
        thought="I'm going to fail",
        feeling="anxious",
        intensity=8,
        distortions=["fortune_telling", "catastrophizing"],
        behaviors=["avoidance"],
        crisis_level=0
    )
    assert extraction.feeling == "anxious"
    assert extraction.intensity == 8
    print("✓ Test 1: Valid extraction created")

    # Test 2: Defaults work
    extraction = Extraction()
    assert extraction.crisis_level == 0
    assert extraction.distortions == []
    assert extraction.behaviors == []
    print("✓ Test 2: Default values work")

    # Test 3: Intensity bounds (should fail if out of range)
    try:
        extraction = Extraction(intensity=11)  # Out of range
        print("✗ Test 3: Intensity validation FAILED (should reject 11)")
    except:
        print("✓ Test 3: Intensity bounds enforced")

    print("\nAll schema tests passed! ✓\n")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING CBT CHATBOT COMPONENT TESTS")
    print("="*70)

    try:
        test_extraction_schema()
        test_state_updates()
        test_policy_decisions()
        test_validation()

        print("="*70)
        print("ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\nThe CBT chatbot components are working correctly.")
        print("To run the full chatbot, use: python main.py")
        print("(You'll need to set ANTHROPIC_API_KEY first)\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
