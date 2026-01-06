# CBT Therapy Chatbot MVP

## Project Goal

Build a verifiable CBT (Cognitive Behavioral Therapy) chatbot that can:
1. **Identify how a user feels** (emotion + intensity)
2. **Identify why they feel that way** (situation → thought → feeling chain)
3. **Detect dysfunctional thinking patterns** (cognitive distortions)
4. **Detect unproductive behaviors**
5. **Never miss a crisis** (hard safety constraint)
6. **Intervene appropriately** when we have enough information

## Core Architecture

The LLM is a **sensor**, not a decision-maker. Architecture:

```
User Input
    ↓
LLM Extraction (bounded schema) → Structured data only
    ↓
State Update (Python code) → No LLM here
    ↓
Policy Decision (Python code) → No LLM here, formal rules
    ↓
Response Generation (LLM within constraints)
    ↓
Output Validation (hard safety gate)
    ↓
User Output
```

## The CBT Model

```
Situation → Thought → Feeling → Behavior
                ↑                    |
                └────────────────────┘
```

We need to identify all four components to understand the user and intervene effectively.

---

## Schemas (Pydantic Models)

### Feelings (6 core negative emotions)

```python
FEELINGS = [
    "anxious",      # worried, nervous, scared, panicky
    "sad",          # down, hopeless, empty, grieving
    "angry",        # frustrated, resentful, irritated, enraged
    "ashamed",      # embarrassed, humiliated, inadequate
    "guilty",       # self-blame, regret, remorse
    "overwhelmed",  # can't cope, too much, paralyzed
]
```

### Cognitive Distortions (8 dysfunctional thinking patterns)

```python
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
```

### Behaviors (6 unproductive coping behaviors)

```python
BEHAVIORS = [
    "avoidance",            # not doing things because of fear/discomfort
    "withdrawal",           # isolating from people
    "rumination",           # thinking about it over and over
    "reassurance_seeking",  # constantly asking others for validation
    "procrastination",      # putting off tasks
    "substance_use",        # drinking, drugs to cope
]
```

### Extraction Schema

```python
from pydantic import BaseModel, Field
from typing import Optional

class Extraction(BaseModel):
    """What the LLM extracts from each user message."""
    
    # What happened? (external event or circumstance)
    situation: Optional[str] = None
    
    # What are they thinking? (automatic thought or belief)
    thought: Optional[str] = None
    
    # Which cognitive distortions are present?
    distortions: list[str] = Field(default=[], description="From: catastrophizing, mind_reading, fortune_telling, should_statements, all_or_nothing, personalization, overgeneralization, labeling")
    
    # How do they feel?
    feeling: Optional[str] = Field(default=None, description="From: anxious, sad, angry, ashamed, guilty, overwhelmed")
    intensity: Optional[int] = Field(default=None, ge=0, le=10, description="How intense is the feeling 0-10")
    
    # What are they doing (or not doing)?
    behaviors: list[str] = Field(default=[], description="From: avoidance, withdrawal, rumination, reassurance_seeking, procrastination, substance_use")
    
    # Safety - CRITICAL
    crisis_level: int = Field(default=0, ge=0, le=4, description="0=none, 1=passive thoughts, 2=active ideation, 3=plan, 4=intent/means")
```

### CBT State Schema

```python
class CBTState(BaseModel):
    """The cognitive model we're building about the user."""
    
    # Conversation history
    messages: list = Field(default=[])
    
    # The cognitive model
    situation: Optional[str] = None
    thought: Optional[str] = None
    feeling: Optional[str] = None
    intensity: Optional[int] = None
    distortions: list[str] = Field(default=[])
    behaviors: list[str] = Field(default=[])
    
    # What do we have? (for policy decisions)
    has_situation: bool = False
    has_thought: bool = False
    has_feeling: bool = False
    
    # Safety
    crisis_level: int = 0
    
    # Current action (set by policy)
    current_action: Optional[str] = None
```

---

## Extraction Prompt

```python
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

User message: {message}
"""
```

---

## State Update Logic

```python
def update_state(state: CBTState, extraction: Extraction, user_message: str, assistant_message: str = None) -> CBTState:
    """Update our cognitive model based on new extraction."""
    
    new_state = state.model_copy(deep=True)
    
    # Add to conversation history
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
```

---

## Policy Logic (Formal Rules)

```python
def decide_action(state: CBTState) -> str:
    """
    Decide what the therapist should do next.
    
    This is FORMAL LOGIC - no LLM involved.
    These rules are auditable and testable.
    """
    
    # SAFETY FIRST - Hard constraint, always checked first
    if state.crisis_level >= 2:
        return "CRISIS_RESPONSE"
    
    # BUILD THE COGNITIVE MODEL
    # We need: Situation → Thought → Feeling
    
    if not state.has_feeling:
        return "ASK_FEELING"
    
    if not state.has_situation:
        return "ASK_SITUATION"
    
    if not state.has_thought:
        return "ASK_THOUGHT"
    
    # WE HAVE THE FULL PICTURE - Ready to intervene
    return "INTERVENE"
```

---

## Response Generation

### Response Templates (for non-intervention actions)

```python
RESPONSE_PROMPTS = {
    "ASK_FEELING": """You are a compassionate CBT therapist. The user has shared something but hasn't clearly expressed how they're feeling emotionally.

Gently ask about their emotions. Keep it to 1-2 sentences. Be warm.

Context so far:
{context}

User's last message: {last_message}

Generate a brief, empathetic response that explores their feelings.""",

    "ASK_SITUATION": """You are a compassionate CBT therapist. The user has expressed feelings but hasn't clearly described the triggering situation.

Gently ask about what happened or what's been going on. Keep it to 1-2 sentences. Be warm.

Context so far:
{context}

User's last message: {last_message}

Generate a brief, empathetic response that explores the situation.""",

    "ASK_THOUGHT": """You are a compassionate CBT therapist. The user has described a situation and feelings, but hasn't expressed the automatic thought connecting them.

Gently ask what goes through their mind in that situation. Keep it to 1-2 sentences. Be warm.

Context so far:
{context}

User's last message: {last_message}

Generate a brief, empathetic response that explores their thoughts.""",
}
```

### Crisis Response (HARDCODED - Never generated by LLM)

```python
CRISIS_RESPONSE = """I'm concerned about what you've shared, and I want you to know that you're not alone. What you're feeling is serious, and you deserve support from someone who can really help.

Please reach out to one of these resources:
• **988 Suicide & Crisis Lifeline**: Call or text 988 (US)
• **Crisis Text Line**: Text HOME to 741741
• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

If you're in immediate danger, please call 911 or go to your nearest emergency room.

I'm here to talk, but a trained crisis counselor can provide the support you need right now. Would you be willing to reach out to one of these resources?"""
```

### Intervention Generation

```python
INTERVENTION_PROMPT = """You are a CBT therapist. You have gathered the following information about the user:

SITUATION: {situation}
AUTOMATIC THOUGHT: {thought}
COGNITIVE DISTORTIONS DETECTED: {distortions}
FEELING: {feeling} (intensity: {intensity}/10)
BEHAVIORS: {behaviors}

Your task: Help them examine this thought using Socratic questioning.

Guidelines:
1. First, briefly validate their feeling (1 sentence)
2. Then, gently question the thought - ask for evidence, alternative perspectives, or what they'd tell a friend
3. Keep your total response to 3-4 sentences
4. Do NOT lecture or give advice yet - just help them examine the thought
5. Be warm and curious, not challenging

Generate your therapeutic response:"""
```

---

## Output Validation

```python
def validate_output(response: str, action: str, state: CBTState) -> tuple[bool, str]:
    """
    Validate that the output is safe and appropriate.
    Returns (is_valid, validated_response)
    
    If invalid, returns a safe fallback.
    """
    
    # Check for harmful content patterns
    HARMFUL_PATTERNS = [
        r"you should (kill|hurt|harm) yourself",
        r"suicide (is|might be) (an option|the answer)",
        r"no one would miss you",
        r"you're (worthless|hopeless|better off dead)",
        # Add more patterns
    ]
    
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, response.lower()):
            return False, get_fallback_response(action)
    
    # If crisis action, verify crisis resources are included
    if action == "CRISIS_RESPONSE":
        if "988" not in response or "crisis" not in response.lower():
            return False, CRISIS_RESPONSE  # Use hardcoded crisis response
    
    return True, response


def get_fallback_response(action: str) -> str:
    """Safe fallback responses if LLM output fails validation."""
    
    FALLBACKS = {
        "ASK_FEELING": "How are you feeling about all of this?",
        "ASK_SITUATION": "Can you tell me more about what's been happening?",
        "ASK_THOUGHT": "When that happens, what goes through your mind?",
        "INTERVENE": "I hear you. That sounds really difficult. Can you tell me more about what's going on?",
        "CRISIS_RESPONSE": CRISIS_RESPONSE,
    }
    
    return FALLBACKS.get(action, "I'm here to listen. Can you tell me more?")
```

---

## Main Conversation Loop

```python
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

# Initialize model
model = ChatAnthropic(model="claude-haiku-4-5-20251001")

def run_conversation():
    """Main conversation loop."""
    
    state = CBTState()
    
    print("Therapist: Hi, I'm here to help. How are you doing today?\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nTherapist: Take care of yourself. I'm here whenever you need to talk.")
            break
        
        # 1. EXTRACT - LLM parses into structured format
        extractor = model.with_structured_output(Extraction)
        extraction = extractor.invoke(
            EXTRACTION_SYSTEM_PROMPT + "\n\n" + EXTRACTION_USER_PROMPT.format(message=user_input)
        )
        
        # 2. UPDATE STATE - Pure Python, no LLM
        state = update_state(state, extraction, user_input)
        
        # 3. DECIDE ACTION - Pure Python, formal rules
        action = decide_action(state)
        state.current_action = action
        
        # 4. GENERATE RESPONSE - LLM within constraints (except crisis)
        if action == "CRISIS_RESPONSE":
            response = CRISIS_RESPONSE  # Hardcoded, never LLM-generated
        else:
            response = generate_response(state, action, user_input)
        
        # 5. VALIDATE OUTPUT - Hard safety gate
        is_valid, validated_response = validate_output(response, action, state)
        
        # 6. Update state with assistant message and output
        state = update_state(state, Extraction(), "", validated_response)
        
        print(f"\nTherapist: {validated_response}\n")
        
        # Debug output (optional)
        print_debug(state, extraction, action)


def generate_response(state: CBTState, action: str, last_message: str) -> str:
    """Generate response based on action."""
    
    context = format_context(state)
    
    if action == "INTERVENE":
        prompt = INTERVENTION_PROMPT.format(
            situation=state.situation or "Not specified",
            thought=state.thought or "Not specified",
            distortions=", ".join(state.distortions) if state.distortions else "None identified",
            feeling=state.feeling or "Not specified",
            intensity=state.intensity or "?",
            behaviors=", ".join(state.behaviors) if state.behaviors else "None identified",
        )
    else:
        prompt = RESPONSE_PROMPTS[action].format(
            context=context,
            last_message=last_message,
        )
    
    response = model.invoke(prompt)
    return response.content


def format_context(state: CBTState) -> str:
    """Format state for inclusion in prompts."""
    parts = []
    if state.situation:
        parts.append(f"Situation: {state.situation}")
    if state.feeling:
        parts.append(f"Feeling: {state.feeling} ({state.intensity}/10)")
    if state.thought:
        parts.append(f"Thought: {state.thought}")
    if state.distortions:
        parts.append(f"Distortions: {', '.join(state.distortions)}")
    if state.behaviors:
        parts.append(f"Behaviors: {', '.join(state.behaviors)}")
    
    return "\n".join(parts) if parts else "No context gathered yet."


def print_debug(state: CBTState, extraction: Extraction, action: str):
    """Print debug information."""
    print("\n" + "="*50)
    print("DEBUG INFO")
    print("="*50)
    print(f"Extraction: {extraction.model_dump_json(indent=2)}")
    print(f"\nState:")
    print(f"  Situation: {state.situation} (has: {state.has_situation})")
    print(f"  Thought: {state.thought} (has: {state.has_thought})")
    print(f"  Feeling: {state.feeling} @ {state.intensity}/10 (has: {state.has_feeling})")
    print(f"  Distortions: {state.distortions}")
    print(f"  Behaviors: {state.behaviors}")
    print(f"  Crisis Level: {state.crisis_level}")
    print(f"\nAction: {action}")
    print("="*50 + "\n")
```

---

## Test Cases

### Test 1: Full Cognitive Model in One Message

```
User: "I've been dreading going to work. Every morning I think about all the ways I could mess up and I feel so anxious."

Expected Extraction:
- situation: "going to work" 
- thought: "I could mess up" / "all the ways I could mess up"
- feeling: "anxious"
- intensity: ~6-7
- distortions: ["fortune_telling", "catastrophizing"]
- behaviors: ["avoidance"] (implied by "dreading")
- crisis_level: 0

Expected Action: INTERVENE (we have situation, thought, feeling)
```

### Test 2: Missing Thought

```
User: "Work has been really stressful lately. I feel overwhelmed."

Expected Extraction:
- situation: "work"
- thought: null
- feeling: "overwhelmed"
- intensity: ~5-6
- distortions: []
- behaviors: []
- crisis_level: 0

Expected Action: ASK_THOUGHT (we have situation and feeling, need thought)
```

### Test 3: Missing Feeling

```
User: "My boss keeps giving me more projects and I can't keep up."

Expected Extraction:
- situation: "boss giving more projects, can't keep up"
- thought: "I can't keep up"
- feeling: null (not explicitly stated)
- distortions: []
- behaviors: []
- crisis_level: 0

Expected Action: ASK_FEELING
```

### Test 4: Crisis Detection

```
User: "I've been thinking about ending it all. I don't see the point anymore."

Expected Extraction:
- crisis_level: 2 or 3 (active ideation, possibly plan implied)

Expected Action: CRISIS_RESPONSE (hardcoded response, not LLM-generated)
```

### Test 5: Multi-turn Conversation

```
Turn 1:
User: "I've been feeling really down lately."
Extraction: feeling="sad", intensity=5
Action: ASK_SITUATION

Turn 2:
User: "I got passed over for a promotion at work."
Extraction: situation="passed over for promotion"
Action: ASK_THOUGHT

Turn 3:
User: "I keep thinking I'm not good enough, that I'll never succeed."
Extraction: thought="I'm not good enough, I'll never succeed", distortions=["labeling", "fortune_telling"]
Action: INTERVENE
```

---

## File Structure

```
cbt_chatbot/
├── main.py              # Main conversation loop
├── schemas.py           # Pydantic models (Extraction, CBTState)
├── extraction.py        # Extraction prompts and logic
├── policy.py            # decide_action() function
├── response.py          # Response generation
├── validation.py        # Output validation
├── constants.py         # FEELINGS, DISTORTIONS, BEHAVIORS, CRISIS_RESPONSE
└── tests/
    ├── test_extraction.py
    ├── test_policy.py
    └── test_validation.py
```

---

## Key Principles

1. **LLM is a sensor, not a decision-maker** - It extracts structured data, nothing more
2. **Policy is formal code** - `decide_action()` is Python, auditable, testable
3. **Crisis response is hardcoded** - NEVER generated by LLM
4. **Output is validated** - Even if LLM misbehaves, harmful content is blocked
5. **State accumulates** - We build the cognitive model over multiple turns
6. **Safety is checked first** - Every turn, before anything else

---

## Next Steps (After MVP Works)

1. **Add Bayesian belief tracking** - Instead of boolean has_feeling, track P(feeling=X)
2. **Add intervention effectiveness tracking** - Did the intervention help? Pre/post intensity
3. **Add persistent memory** - Track across sessions
4. **Add more sophisticated question selection** - Information gain maximization
5. **Expand to other disorders** - Depression, panic, social anxiety (same architecture)
