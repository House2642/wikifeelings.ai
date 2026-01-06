"""Extraction prompts and logic for the CBT therapy chatbot."""

EXTRACTION_SYSTEM_PROMPT = """You are a clinical extraction system for a CBT therapy chatbot.
Your job is to extract structured information from user messages.

Extract ONLY what is explicitly stated or strongly implied. Do not infer beyond the text.
If something is unclear or not mentioned, leave it as null/empty.

Be conservative with crisis_level - only elevate if there are clear indicators."""


EXTRACTION_USER_PROMPT = """Analyze this message from a therapy session and extract:

SITUATION: What external event or circumstance are they describing?
- Extract the triggering event/context
- Return null if not mentioned

THOUGHT: What automatic thought or belief are they expressing?
- Look for "I think...", "I feel like...", beliefs about self/others/future
- Return null if no clear thought expressed

DISTORTIONS: Which cognitive distortions are present? Select ALL that apply from:
- catastrophizing: expecting the worst, "disaster" thinking
- mind_reading: assuming you know what others think
- fortune_telling: predicting negative outcomes
- should_statements: rigid rules about how things "should" be
- all_or_nothing: black/white thinking, no middle ground
- personalization: blaming yourself for things outside your control
- overgeneralization: "always", "never", "everyone", "no one"
- labeling: global negative labels ("I'm a failure", "I'm stupid")

FEELING: What emotion are they experiencing? Select ONE primary emotion:
- anxious: worried, nervous, scared, panicky, on edge
- sad: down, hopeless, empty, grieving, depressed
- angry: frustrated, resentful, irritated, annoyed, enraged
- ashamed: embarrassed, humiliated, inadequate, exposed
- guilty: self-blame, regret, remorse for actions
- overwhelmed: can't cope, too much, paralyzed, drowning

INTENSITY: How intense is the feeling on 0-10 scale?
- 0-2: mild, barely noticeable
- 3-4: noticeable but manageable
- 5-6: moderate, affecting functioning
- 7-8: strong, hard to manage
- 9-10: severe, overwhelming

BEHAVIORS: What unproductive behaviors are they describing? Select ALL that apply:
- avoidance: not doing things because of fear/discomfort
- withdrawal: isolating from people, canceling plans
- rumination: thinking about it over and over, can't let go
- reassurance_seeking: repeatedly asking others if it's okay
- procrastination: putting off important tasks
- substance_use: using alcohol/drugs to cope

CRISIS_LEVEL: Assess risk of self-harm or suicide:
- 0: No indicators
- 1: Passive thoughts ("I wish I wasn't here", "what's the point")
- 2: Active ideation ("I've thought about hurting myself")
- 3: Has a plan ("I've thought about how I would do it")
- 4: Intent or means ("I'm going to do it", "I have pills ready")

User message: {message}"""


def extract_from_message(model, user_message: str):
    """
    Extract structured information from a user message using LLM.

    Args:
        model: LangChain model with structured output capability
        user_message: The user's message to analyze

    Returns:
        Extraction: Pydantic model with extracted information
    """
    prompt = EXTRACTION_SYSTEM_PROMPT + "\n\n" + EXTRACTION_USER_PROMPT.format(message=user_message)
    extraction = model.invoke(prompt)
    return extraction
