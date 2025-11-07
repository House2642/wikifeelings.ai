from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from datetime import datetime
import operator
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()

from debug import pretty_print_situation

model = init_chat_model("gpt-4o")
#source:https://beckinstitute.org/wp-content/uploads/2021/08/Traditional-CCD.pdf
#What really is meaning_of_at
class Situation(BaseModel):
    when: datetime = Field(default_factory=datetime.now, description="When this situation occurred")
    messages: Annotated[List[AnyMessage], operator.add] = Field(default=[], description="Conversation messages")
    next_question: str = Field(default="", description="question to ask user")
    situation: str = Field(default="", description="What is currently going on")
    automatic_thoughts: Annotated[List[str], operator.add] = Field(default=[], description="List of automatic thoughts")
    meaning_of_at: Optional[str] = Field(default=None, description="What these automatic thoughts mean to the person")
    emotions: Annotated[List[str], operator.add] = Field(default=[], description="Emotions experienced in this situation")
    behavior: Annotated[List[str], operator.add] = Field(default=[], description="Behaviors or actions taken in response")
    num_follow_ups: int = Field(default=0)

class extractSitData(BaseModel):
    situation: str = Field(default="", description="What is currently going on")
    automatic_thoughts: List[str] = Field(default=[], description="List of automatic thoughts")
    meaning_of_at: Optional[str] = Field(default=None, description="What these automatic thoughts mean to the person")
    emotions: List[str] = Field(default=[], description="Emotions experienced in this situation")
    behavior: List[str] = Field(default=[], description="Behaviors or actions taken in response")

class FollowUpDecision(BaseModel):
    should_follow_up: bool = Field(default=False, description="Whether a follow-up question is needed (True/False).")
    reasoning: str = Field(default="",description="Brief explanation of why this follow-up question is needed.")

class FollowUpQuestion(BaseModel):
    follow_up_question: str = Field(default="",description="The specific follow-up question to ask the client.")

structuredModel = model.with_structured_output(extractSitData)

def ask(state: Situation):
    user_input = input("Agent: "+ state.next_question + "\nUser:")
    msg = structuredModel.invoke([SystemMessage(f"""
Act as a licensed CBT therapist conducting a clinical assessment. Extract CBT data using the following therapeutic definitions:

**SITUATION**: The specific, objective facts about what happened. Focus on:
- Observable events, times, places, people involved
- Factual circumstances that triggered the emotional response
- Avoid interpretations or subjective judgments
- Example: "During my presentation to the board at 2 PM" not "I gave a terrible presentation"

**AUTOMATIC THOUGHTS**: Spontaneous thoughts, images, or memories that occur in response to the situation:
- Stream-of-consciousness thoughts that "pop into mind"
- Self-talk, predictions, interpretations, or judgments
- Can be words, images, or memories
- Often brief and may seem realistic in the moment
- Example: "I'm going to mess this up" or "Everyone thinks I'm incompetent"

**MEANING OF AUTOMATIC THOUGHTS**: The deeper significance or core beliefs these thoughts reflect:
- What these thoughts suggest about self, others, world, or future
- Underlying schemas or core beliefs being activated
- Personal significance: "What does this mean about me as a person?"
- Example: If thought is "I'll fail the test" → meaning might be "I'm not smart enough" or "I'm inadequate"

**EMOTIONS**: Specific feeling states (not thoughts disguised as feelings):
- Primary emotions: sad, anxious, angry, guilty, ashamed, hurt, happy, excited
- Rate intensity if possible (mild, moderate, severe)
- Avoid cognitive words like "feel like" - focus on pure emotional states
- Example: "anxious and ashamed" not "I feel like I'm stupid"

**BEHAVIORS**: Observable actions, responses, or coping strategies:
- What the person did or didn't do in response to thoughts/emotions
- Safety behaviors, avoidance, escape, rituals
- Physical responses, communication patterns
- Include both helpful and unhelpful coping strategies
- Example: "avoided eye contact, left meeting early, called in sick the next day"

Extract only what you can confidently identify from the client's description. Leave fields blank if information is unclear or not provided - you'll have opportunities for follow-up questions to gather missing data.

Current therapeutic context: Initial assessment session focusing on identifying the CBT triangle (thoughts-feelings-behaviors) connected to a specific triggering situation.
"""), HumanMessage(f"""How user is feeling today: {user_input}""")])

    return {"situation": msg.situation, 
            "automatic_thoughts": msg.automatic_thoughts,
            "emotions": msg.emotions, 
            "behavior": msg.behavior, 
            "meaning_of_at": msg.meaning_of_at,
            "messages": [AIMessage("Agent: " + state.next_question), HumanMessage("User: " + user_input)]}

def determine_follow_up(state: Situation):
    if state.num_follow_ups > 5:
        return False
    follow_up_model = model.with_structured_output(FollowUpDecision)
    response = follow_up_model.invoke([
        SystemMessage(f"""
            ROLE: You are a licensed CBT therapist determining if a follow-up question is needed.

            CONTEXT SUMMARY:
            Recent conversation:
            {state.messages}

            Latest extracted data:
            - Situation: {state.situation}
            - Automatic thoughts: {state.automatic_thoughts}
            - Meaning of AT: {state.meaning_of_at}
            - Emotions: {state.emotions}
            - Behaviors: {state.behavior}
            - Last question asked: {state.next_question}

            OBJECTIVE:

            Determine whether another follow-up question is required to deepen understanding or ensure safety.
            Generate reasoning on why a follow up question might need to be asked
        """
        )
    ])
    return response.should_follow_up

def generate_follow_up(state: Situation):
    follow_up_model = model.with_structured_output(FollowUpQuestion)
    response = follow_up_model.invoke([
        SystemMessage(f"""
                ROLE: You are a licensed Cognitive Behavioral Therapist (CBT) continuing a clinical assessment session. 
                Your goal is to decide the **next single follow-up question** that best helps the client clarify their 
                thoughts, feelings, behaviors, or deeper meanings.

                CONTEXT:
                Below is the current summary of the therapeutic interaction.

                RECENT CONVERSATION:
                {state.messages}

                CURRENT WORKING DATA:
                - Situation: {state.situation}
                - Automatic Thoughts: {state.automatic_thoughts}
                - Meaning of Automatic Thoughts: {state.meaning_of_at}
                - Emotions: {state.emotions}
                - Behaviors: {state.behavior}
                - Last question asked: {state.next_question}

                THERAPEUTIC OBJECTIVE:
                Ask **one open-ended, empathic question** that helps move the client deeper into reflection or fill in 
                missing parts of the CBT triangle (Situation – Thoughts – Emotions – Behaviors – Meaning). 
                You are guiding the client through gentle discovery, not interrogation.

                DECISION RULES:
                1. **Safety first.** If there is *any* hint of self-harm, hopelessness, or inability to cope,
                ask: "Are you having thoughts about harming yourself or ending your life right now?"  
                (Do not ask anything else.)

                2. **If no safety concern**, identify which area is most unclear:
                - Situation unclear → clarify *who/what/when/where* of the triggering event.
                - Automatic thoughts unclear → clarify *exact words or images* that went through their mind.
                - Emotions unclear → clarify *which feelings* and their *intensity*.
                - Behaviors unclear → clarify *what they did or didn’t do* in response.
                - Meaning unclear → clarify *what that thought means about them, others, or the future*.

                3. **Ask one question only.**
                - Do not chain multiple questions with “and” or “or.”
                - Keep ≤ 20 words unless safety question.
                - Use warm, natural tone, like a therapist mid-session.
                """
        )
    ])
    return {"next_question": response.follow_up_question, "num_follow_ups": state.num_follow_ups +1}




situation = StateGraph(Situation)
situation.add_node("ask", ask)
situation.add_node("genQ", generate_follow_up)

situation.add_edge(START, "ask")
situation.add_conditional_edges("ask", determine_follow_up, {True: "genQ", False: END})
situation.add_edge("genQ", "ask")
chain = situation.compile()

openQ =  "How are you feeling today?"
initial_state = Situation(next_question=openQ)
final_state_dict = chain.invoke(initial_state)

final_state = Situation(
    when=final_state_dict.get('when', datetime.now()),
    situation=final_state_dict.get('situation', ''),
    automatic_thoughts=final_state_dict.get('automatic_thoughts', []),
    meaning_of_at=final_state_dict.get('meaning_of_at'),
    emotions=final_state_dict.get('emotions', []),
    behavior=final_state_dict.get('behavior', []),
    messages=[],
    next_question = ""
)
pretty_print_situation(final_state)