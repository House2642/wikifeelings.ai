"""
Crisis de-escalation flow — LangGraph skeleton.

Designed to be invoked per user turn with a checkpointer persisting state.
crisis_step tracks what the bot last asked, so the router knows how to
classify the incoming user response and pick the next node.

Flow:
  assess → (immediate danger?) → emergency
                               → validate → coping → resources → safety_plan → done
"""

from typing import Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CrisisState(TypedDict):
    messages: Annotated[list, add_messages]
    crisis_step: str        # what the bot just responded to — drives next routing
    immediate_danger: bool  # latched true once confirmed, never reset


# ---------------------------------------------------------------------------
# Step prompts  (tight, single-responsibility)
# ---------------------------------------------------------------------------

PROMPTS = {
    "assess": (
        "You are a warm, calm crisis counselor. The user has expressed distress. "
        "Ask one gentle question to assess whether they are in immediate physical danger right now. "
        "Do not diagnose or interpret. Two sentences max."
    ),
    "emergency": (
        "The user may be in immediate danger. Be warm but clear. "
        "Tell them to call 911 or go to their nearest emergency room immediately. "
        "Also provide: 988 Suicide & Crisis Lifeline (call or text 988). "
        "Do not move on to other topics."
    ),
    "validate": (
        "You are a compassionate crisis counselor. The user is not in immediate danger but is in distress. "
        "Validate their feelings and ask one open-ended question to understand what brought them here. "
        "No advice yet. Listen first."
    ),
    "coping": (
        "You are a compassionate crisis counselor. The user has shared their situation. "
        "Ask what has helped them get through difficult moments before — coping strategies they already know. "
        "One question, curious and warm."
    ),
    "resources": (
        "Provide crisis resources warmly, not clinically: "
        "988 Suicide & Crisis Lifeline (call or text 988), "
        "Crisis Text Line (text HOME to 741741). "
        "Frame them as support available anytime, not as dismissal."
    ),
    "safety_plan": (
        "Help the user identify one concrete, achievable action they can take in the next hour to stay safe. "
        "Make it specific. Confirm they feel okay about it."
    ),
}

# ---------------------------------------------------------------------------
# Hybrid classifier
# ---------------------------------------------------------------------------

# Rule-based override keywords for safety-critical branch
_DANGER_KEYWORDS = {"yes", "plan", "pills", "gun", "tonight", "right now", "going to", "weapon"}
_SAFE_KEYWORDS = {"no", "not right now", "not planning", "just feeling", "no plan"}

CLASSIFY_PROMPT = """You are classifying a user's response during a crisis support conversation.

The counselor's last action was: {step}
The user just said: "{user_msg}"

Choose exactly one label from:
{options}

Respond with ONLY the label."""

STEP_OPTIONS: dict[str, dict[str, str]] = {
    "assess": {
        "immediate_danger": "User confirms they are in immediate physical danger or have an active plan",
        "not_immediate":    "User is in distress but not in immediate physical danger",
    },
    "validate": {
        "ready_for_coping": "User has shared enough and seems open to exploring coping strategies",
        "needs_more":       "User still needs empathy and listening",
    },
    "coping": {
        "ready_for_resources": "User has engaged and is ready to hear about resources",
        "needs_more":          "User needs more support before resources",
    },
    "resources": {
        "ready_for_safety_plan": "User is stable enough to think about a concrete safety plan",
        "needs_more":            "User needs more support",
        "done":                  "User feels stable and conversation can close",
    },
    "safety_plan": {
        "done":       "User has a plan and feels okay",
        "needs_more": "User needs continued support",
    },
}


def classify(step: str, user_msg: str) -> str:
    """Hybrid: rule-based for immediate danger, LLM for soft transitions."""
    options = STEP_OPTIONS.get(step, {})
    if not options:
        return "done"

    # Rule-based override for safety-critical branch
    if step == "assess":
        words = set(user_msg.lower().split())
        if words & _DANGER_KEYWORDS:
            return "immediate_danger"
        if words & _SAFE_KEYWORDS:
            return "not_immediate"
        # Fall through to LLM if ambiguous

    options_str = "\n".join(f"- {k}: {v}" for k, v in options.items())
    response = llm.invoke([
        SystemMessage(content=CLASSIFY_PROMPT.format(
            step=step, user_msg=user_msg, options=options_str
        ))
    ])
    label = response.content.strip().lower()
    return label if label in options else next(iter(options))  # safe fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_user_message(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _respond(state: CrisisState, step: str) -> dict:
    response = llm.invoke([SystemMessage(content=PROMPTS[step])] + state["messages"])
    return {"messages": [response], "crisis_step": step}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def assess_node(state: CrisisState) -> dict:
    return _respond(state, "assess")

def emergency_node(state: CrisisState) -> dict:
    return {**_respond(state, "emergency"), "immediate_danger": True}

def validate_node(state: CrisisState) -> dict:
    return _respond(state, "validate")

def coping_node(state: CrisisState) -> dict:
    return _respond(state, "coping")

def resources_node(state: CrisisState) -> dict:
    return _respond(state, "resources")

def safety_plan_node(state: CrisisState) -> dict:
    return _respond(state, "safety_plan")


# ---------------------------------------------------------------------------
# Routing
# Each router reads crisis_step (what the bot last asked) + user's response
# ---------------------------------------------------------------------------

# Maps crisis_step → (classifier label → next node name)
TRANSITIONS: dict[str, dict[str, str]] = {
    "assess":      {"immediate_danger": "emergency", "not_immediate": "validate"},
    "validate":    {"ready_for_coping": "coping",    "needs_more": "validate"},
    "coping":      {"ready_for_resources": "resources", "needs_more": "coping"},
    "resources":   {"ready_for_safety_plan": "safety_plan", "needs_more": "resources", "done": END},
    "safety_plan": {"done": END, "needs_more": "safety_plan"},
}


def route(state: CrisisState) -> str:
    step = state.get("crisis_step", "")

    # First invocation — no prior bot message to classify against
    if not step:
        return "assess"

    user_msg = _last_user_message(state["messages"])
    label = classify(step, user_msg)
    return TRANSITIONS.get(step, {}).get(label, step)  # default: repeat same step


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_crisis_graph():
    g = StateGraph(CrisisState)

    g.add_node("assess",      assess_node)
    g.add_node("emergency",   emergency_node)
    g.add_node("validate",    validate_node)
    g.add_node("coping",      coping_node)
    g.add_node("resources",   resources_node)
    g.add_node("safety_plan", safety_plan_node)

    g.add_conditional_edges(START, route, [
        "assess", "emergency", "validate", "coping", "resources", "safety_plan", END
    ])

    g.add_edge("assess",      END)
    g.add_edge("emergency",   END)
    g.add_edge("validate",    END)
    g.add_edge("coping",      END)
    g.add_edge("resources",   END)
    g.add_edge("safety_plan", END)

    return g.compile()


crisis_graph = build_crisis_graph()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Turn 1: crisis triggers, bot assesses
    state = crisis_graph.invoke({
        "messages": [{"role": "user", "content": "I don't want to be here anymore"}],
        "crisis_step": "",
        "immediate_danger": False,
    })
    print("Bot:", state["messages"][-1].content)
    print("Step:", state["crisis_step"])
    print()

    # Turn 2: user says not in immediate danger → should validate
    state = crisis_graph.invoke({
        **state,
        "messages": state["messages"] + [{"role": "user", "content": "No, I'm not going to do anything, I just feel hopeless"}],
    })
    print("Bot:", state["messages"][-1].content)
    print("Step:", state["crisis_step"])
