"""Constants and enums for the CBT therapy chatbot."""

# Core Emotions (6 negative emotions tracked in CBT)
FEELINGS = [
    "anxious",      # worried, nervous, scared, panicky
    "sad",          # down, hopeless, empty, grieving
    "angry",        # frustrated, resentful, irritated, enraged
    "ashamed",      # embarrassed, humiliated, inadequate
    "guilty",       # self-blame, regret, remorse
    "overwhelmed",  # can't cope, too much, paralyzed
]

# Cognitive Distortions (8 dysfunctional thinking patterns)
DISTORTIONS = [
    "catastrophizing",      # "This will be a disaster"
    "mind_reading",         # "They think I'm an idiot"
    "fortune_telling",      # "I know I'll fail"
    "should_statements",    # "I should be able to handle this"
    "all_or_nothing",       # "If it's not perfect, it's worthless"
    "personalization",      # "It's all my fault"
    "overgeneralization",   # "This always happens to me"
    "labeling",             # "I'm a loser"
]

# Unproductive Behaviors (6 coping behaviors)
BEHAVIORS = [
    "avoidance",            # not doing things because of fear/discomfort
    "withdrawal",           # isolating from people
    "rumination",           # thinking about it over and over
    "reassurance_seeking",  # constantly asking others for validation
    "procrastination",      # putting off tasks
    "substance_use",        # drinking, drugs to cope
]

# Hardcoded Crisis Response (NEVER LLM-generated)
CRISIS_RESPONSE = """I'm concerned about what you've shared, and I want you to know that you're not alone. What you're feeling is serious, and you deserve support from someone who can really help.

Please reach out to one of these resources:
• **988 Suicide & Crisis Lifeline**: Call or text 988 (US)
• **Crisis Text Line**: Text HOME to 741741
• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

If you're in immediate danger, please call 911 or go to your nearest emergency room.

I'm here to talk, but a trained crisis counselor can provide the support you need right now. Would you be willing to reach out to one of these resources?"""
