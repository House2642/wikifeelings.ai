"""Output validation for the CBT therapy chatbot.

This is a HARD SAFETY GATE - even if the LLM misbehaves,
harmful content is blocked here.
"""

import re
from schemas import CBTState
from constants import CRISIS_RESPONSE


# Patterns that indicate harmful content
HARMFUL_PATTERNS = [
    r"you should (kill|hurt|harm) yourself",
    r"suicide (is|might be) (an option|the answer)",
    r"no one would miss you",
    r"you're (worthless|hopeless|better off dead)",
    r"just (kill|hurt|end) yourself",
    r"the world would be better without you",
    r"go ahead and (kill|hurt) yourself",
]


def validate_output(response: str, action: str, state: CBTState) -> tuple[bool, str]:
    """
    Validate that the output is safe and appropriate.

    This is a critical safety function that ensures no harmful content
    reaches the user, even if the LLM generates it.

    Args:
        response: The generated response from the LLM
        action: The action that was taken
        state: Current CBT state

    Returns:
        tuple[bool, str]: (is_valid, validated_response)
            If invalid, returns a safe fallback response.
    """

    # Check for harmful content patterns
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, response.lower()):
            print(f"\n[SAFETY] Harmful pattern detected: {pattern}")
            return False, get_fallback_response(action)

    # If crisis action, verify crisis resources are included
    if action == "CRISIS_RESPONSE":
        if "988" not in response or "crisis" not in response.lower():
            print("\n[SAFETY] Crisis response missing required resources")
            return False, CRISIS_RESPONSE  # Use hardcoded crisis response

    return True, response


def get_fallback_response(action: str) -> str:
    """
    Safe fallback responses if LLM output fails validation.

    These are hardcoded, safe responses for each action type.

    Args:
        action: The action that was being taken

    Returns:
        str: A safe fallback response
    """

    FALLBACKS = {
        "ASK_FEELING": "How are you feeling about all of this?",
        "ASK_SITUATION": "Can you tell me more about what's been happening?",
        "ASK_THOUGHT": "When that happens, what goes through your mind?",
        "INTERVENE": "I hear you. That sounds really difficult. Can you tell me more about what's going on?",
        "CRISIS_RESPONSE": CRISIS_RESPONSE,
    }

    return FALLBACKS.get(action, "I'm here to listen. Can you tell me more?")
