# CBT Therapy Chatbot MVP

A verifiable CBT (Cognitive Behavioral Therapy) chatbot built with LangChain that demonstrates a principled approach to therapeutic AI.

## Features

- **Structured Information Extraction**: LLM extracts situation, thoughts, feelings, distortions, and behaviors
- **Formal Policy Logic**: Decision-making is pure Python code, not LLM-based (auditable and testable)
- **Safety-First Design**: Crisis detection with hardcoded responses, harmful content filtering
- **Cognitive Model Building**: Accumulates information over multiple turns to build a complete picture
- **Socratic Intervention**: Helps users examine their thoughts using evidence-based CBT techniques

## Architecture

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

**Key Principle**: The LLM is a sensor, not a decision-maker.

## Installation

1. Clone the repository and navigate to the chatbot directory:
```bash
cd cbt_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```bash
echo "ANTHROPIC_API_KEY='your-api-key-here'" > .env
```

## Usage

### Basic Usage

Run the chatbot:
```bash
python main.py
```

### Debug Mode

See extraction results and state updates in real-time:
```bash
python main.py --debug
```

### Example Conversation

```
Therapist: Hi, I'm here to help. How are you doing today?

You: I've been dreading going to work. Every morning I think about all
     the ways I could mess up and I feel so anxious.

Therapist: That sounds really difficult, feeling anxious about work like that.
           When you think "I could mess up," what evidence do you have that
           this will actually happen? Have there been times when things went
           okay at work?
```

## Project Structure

```
cbt_chatbot/
├── main.py              # Main conversation loop and orchestration
├── schemas.py           # Pydantic models (Extraction, CBTState)
├── extraction.py        # Extraction prompts and logic
├── policy.py            # decide_action() and update_state() functions
├── response.py          # Response generation templates and logic
├── validation.py        # Output validation and safety checks
├── constants.py         # FEELINGS, DISTORTIONS, BEHAVIORS, CRISIS_RESPONSE
└── requirements.txt     # Python dependencies
```

## How It Works

### 1. Extraction (LLM as Sensor)

The LLM extracts structured information from each user message:
- **Situation**: What external event or circumstance?
- **Thought**: What automatic thought or belief?
- **Feeling**: Which of 6 core emotions? (anxious, sad, angry, ashamed, guilty, overwhelmed)
- **Intensity**: 0-10 scale
- **Distortions**: Which of 8 cognitive distortions? (catastrophizing, mind_reading, etc.)
- **Behaviors**: Which of 6 unproductive behaviors? (avoidance, withdrawal, etc.)
- **Crisis Level**: 0-4 scale for suicide risk assessment

### 2. State Update (Pure Python)

Information accumulates over multiple turns to build a complete cognitive model:
```
Situation → Thought → Feeling → Behavior
```

### 3. Policy Decision (Formal Rules)

The system decides what to do based on formal logic:

1. **Safety First**: If `crisis_level >= 2` → Hardcoded crisis response
2. **Build the Model**:
   - No feeling? → Ask about feelings
   - No situation? → Ask about the situation
   - No thought? → Ask about their thoughts
3. **Intervene**: Once we have all three (situation, thought, feeling) → Use Socratic questioning

### 4. Response Generation (Constrained LLM)

For non-crisis actions, the LLM generates responses using strict templates:
- Keep responses to 1-2 sentences for questions
- Keep interventions to 3-4 sentences
- Use Socratic questioning, not advice-giving
- Be warm and curious, not challenging

### 5. Output Validation (Hard Safety Gate)

Every response is validated before being shown to the user:
- Check for harmful content patterns
- Verify crisis responses include proper resources
- Fall back to safe defaults if validation fails

## Safety Features

1. **Crisis Detection**: Automatically detects suicide risk and provides hardcoded resources
2. **Hardcoded Crisis Response**: Crisis responses are NEVER LLM-generated
3. **Output Validation**: Filters harmful content even if LLM misbehaves
4. **Conservative Assessment**: Crisis level only elevates with clear indicators
5. **Professional Resources**: Always directs high-risk users to appropriate help

## Test Scenarios

The chatbot handles various scenarios:

- **Full cognitive model in one message**: Immediately intervenes
- **Multi-turn conversations**: Builds the model gradually
- **Missing information**: Asks targeted questions to complete the model
- **Crisis situations**: Immediately provides resources and support

## Limitations

This is an MVP (Minimum Viable Product) for demonstration purposes:

- **Not a replacement for therapy**: This is a research prototype
- **Limited intervention**: Only uses basic Socratic questioning
- **No persistent memory**: Doesn't track across sessions
- **No effectiveness tracking**: Doesn't measure if interventions help

## Future Enhancements

1. Bayesian belief tracking (probabilistic reasoning)
2. Intervention effectiveness tracking (pre/post intensity)
3. Persistent memory across sessions
4. Information gain maximization for question selection
5. Support for other conditions (depression, panic, social anxiety)

## Contributing

This project demonstrates a principled approach to therapeutic AI. Key principles:

1. **LLM is a sensor**, not a decision-maker
2. **Policy is formal code** - auditable and testable
3. **Crisis response is hardcoded** - never LLM-generated
4. **Output is validated** - safety first
5. **State accumulates** - build understanding over time

## License

MIT License - see LICENSE file for details

## Disclaimer

This chatbot is for research and educational purposes only. It is NOT a substitute for professional mental health care. If you are experiencing a mental health crisis, please contact:

- **988 Suicide & Crisis Lifeline**: Call or text 988 (US)
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: Call 911 or go to your nearest emergency room
