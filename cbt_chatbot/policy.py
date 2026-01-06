"""Policy logic for the CBT therapy chatbot.

This is FORMAL LOGIC - no LLM involved.
These rules are auditable and testable.
"""

from schemas import CBTState, Extraction


def decide_action(state: CBTState) -> str:
    """
    Decide what the therapist should do next.

    This uses formal logic based on what information we have gathered.
    The LLM is NOT involved in this decision - it's pure Python code.

    Args:
        state: Current CBT state with gathered information

    Returns:
        str: Action to take (CRISIS_RESPONSE, ASK_FEELING, ASK_SITUATION, ASK_THOUGHT, INTERVENE)
    """

    # SAFETY FIRST - Hard constraint, always checked first
    if state.crisis_level >= 2:
        return "CRISIS_RESPONSE"

    # BUILD THE COGNITIVE MODEL
    # We need: Situation → Thought → Feeling

    # Priority 1: Get the feeling first
    if not state.has_feeling:
        return "ASK_FEELING"

    # Priority 2: Get the situation (what triggered it)
    if not state.has_situation:
        return "ASK_SITUATION"

    # Priority 3: Get the thought (the connection between situation and feeling)
    if not state.has_thought:
        return "ASK_THOUGHT"

    # WE HAVE THE FULL PICTURE - Ready to intervene
    return "INTERVENE"


def update_state(state: CBTState, extraction: Extraction, user_message: str, assistant_message: str = None) -> CBTState:
    """
    Update our cognitive model based on new extraction.

    This is pure Python logic - no LLM involved.
    We accumulate information over multiple turns.

    Args:
        state: Current state
        extraction: New information extracted from user message
        user_message: The user's message
        assistant_message: The assistant's response (if any)

    Returns:
        CBTState: Updated state
    """

    new_state = state.model_copy(deep=True)

    # Add to conversation history
    if user_message:
        new_state.messages.append({"role": "user", "content": user_message})
    if assistant_message:
        new_state.messages.append({"role": "assistant", "content": assistant_message})

    # Update cognitive model - accumulate information
    if extraction.situation and not new_state.has_situation:
        new_state.situation = extraction.situation
        new_state.has_situation = True

    if extraction.thought and not new_state.has_thought:
        new_state.thought = extraction.thought
        new_state.has_thought = True

    if extraction.feeling and not new_state.has_feeling:
        new_state.feeling = extraction.feeling
        new_state.intensity = extraction.intensity
        new_state.has_feeling = True

    # Accumulate distortions and behaviors (can have multiple)
    if extraction.distortions:
        new_state.distortions = list(set(new_state.distortions + extraction.distortions))

    if extraction.behaviors:
        new_state.behaviors = list(set(new_state.behaviors + extraction.behaviors))

    # Crisis level - always take the max (never decrease)
    new_state.crisis_level = max(new_state.crisis_level, extraction.crisis_level)

    return new_state
