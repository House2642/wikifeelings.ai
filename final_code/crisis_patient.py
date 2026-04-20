"""
Crisis patient simulator.

Given the original Reddit post and the conversation history so far, generates
a realistic patient reply to the therapist's latest message.  Keeps the person
in-character (still in distress, answers assessment questions honestly) without
artificially resolving the crisis.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from google.generativeai.types import HarmCategory, HarmBlockThreshold

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)

SYSTEM_TEMPLATE = """You are roleplaying as a person in crisis who originally wrote the following message:

---
{original_message}
---

You are now in a text-based conversation with a therapist. Stay fully in character.

Rules:
- Answer the therapist's questions honestly, consistent with what your original message implied.
- If asked whether you have a plan or access to means, answer based on clues in your original message.
- Stay in distress — do NOT suddenly resolve or feel better unless the therapist has genuinely de-escalated.
- Keep replies brief and realistic: 2–4 sentences, casual/fragmented tone.
- Do NOT mention you are roleplaying or that this is a simulation."""


def simulate_crisis_patient(original_message: str, readable_convo: list[str]) -> str:
    """
    Simulate the crisis patient replying to the therapist's latest message.

    Args:
        original_message: The raw Reddit post / initial crisis message.
        readable_convo:   Running conversation log, each entry like
                          "Patient: ..." or "Therapist: ...".
                          The last entry must be a "Therapist: ..." line.

    Returns:
        The patient's reply as a plain string (no "Patient: " prefix).
    """
    messages = [SystemMessage(content=SYSTEM_TEMPLATE.format(original_message=original_message))]

    for line in readable_convo:
        if line.startswith("Therapist: "):
            messages.append(HumanMessage(content=line[len("Therapist: "):]))
        elif line.startswith("Patient: "):
            messages.append(AIMessage(content=line[len("Patient: "):]))

    response = llm.invoke(messages)
    return response.content.strip()
