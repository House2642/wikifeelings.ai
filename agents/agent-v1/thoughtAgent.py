from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
from datetime import datetime, timezone
from enum import Enum

load_dotenv()

model = init_chat_model("gpt-4o")


class TriggerType(Enum):
    EVENT = "event"
    UNHELPFUL_BEHAVIOR = "unhelpful_behavior"
    UNCLEAR = "unclear"


class ThoughtRecordState(TypedDict):
    user_input: str
    # session_id: int
    date_time: datetime
    situation_type: TriggerType
    situation_details: str
    automatic_thoughts: list

# tools
def identify_situation(state: ThoughtRecordState):
    trigger = model.invoke([SystemMessage("""
        Analyze this situation for a thought record exercise.
        
        Classify the trigger type as ONE of:
        - "event_external" - External event or situation that happened and is associated with a negative emotion (conversation, missing saying hi to someone, ))
        - "event_internal" - Internal situation (an intense emotion, a painful sensation, an image, daydream, flashback or stream of
            thoughts—e.g., thinking about your future)
        - "unhelpful_behavior" - Action or behavior the USER engaged in (procrastinating, avoiding, lying, ruminating etc.) that is leading to an emotional response
        
        Respond with ONLY the classification word: event_external, event_internal, or unhelpful_behavior
        """),HumanMessage(state["user_input"])])
    situation_details = model.invoke([SystemMessage(f"""
        This situation has been classified as: {trigger.content.strip()}
        Now provide a detailed analysis:
        1. Summarize what happened in 1-2 clear sentences
        Format your response as:
        SUMMARY: [clear summary of the situation]
        """),HumanMessage(f"""User's situation: {state["user_input"]}
        Trigger classification: {trigger.content.strip()}""")])
    return {"situation_type": trigger.content.strip(), "situation_details": situation_details.content}

def label_automatic_thoughts(state: ThoughtRecordState):
    follow_up = model.invoke([SystemMessage(f"""Be empathetic and like a firend. Based on the situation, generate the following question to better understand or
    identify the automatic thought. Use the situation context to frame the questions in context:
        1. What thought(s) and/or image(s) went through your mind (before, during and after the {state['situation_type']})?
        situation type: {state['situation_type']}
        situation: {state['situation_details']}
        Respond with follow up question
        """)])
    users_thoughts = input(follow_up.content+" \nUser:")

    thoughts = model.invoke([SystemMessage(f"""Based on the thoughts the user is having analyze and extract their automatic thoughts.
    Automatic Thoughts are the thoughts that automatically arise in our minds all throughout the day. Often, we are completely unaware we are even having thoughts, but with a little instruction and practice, you can learn to easily identify them, and, as a result, get a better handle on your mood and behavior. 
    Automatic Thoughts Examples
        - Today is going to be terrible. 
        - No one is interested in what I’m saying.
        - I only have bad dates. 
        - It’s all my fault. 
        - She is a bad person. 
        - I’ll never find a good job. 
        - I didn’t deserve the promotion.
        - I should be more motivated. 
        Thoughts = {users_thoughts}
        Output the identified automatic thoughts in a python list = ["Automatic Thought 1", "Automatic Thought 2"]""")])
    
    return {"automatic_thoughts": eval(thoughts.content.strip())}



thought_record = StateGraph(ThoughtRecordState)
thought_record.add_node("identify", identify_situation)
thought_record.add_node("automatic", label_automatic_thoughts)

# Edges:
thought_record.add_edge(START, "identify")
thought_record.add_edge("identify", "automatic")
thought_record.add_edge("automatic", END)

chain = thought_record.compile()
user_input = input("Agent: Describe your situation \nUser:")
state = chain.invoke({'user_input': user_input})
print("Analysis")
print(state["situation_type"])
print(state["situation_details"])
print(state["automatic_thoughts"])
print("\n--- --- ---\n")