from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

model = ChatAnthropic(model="claude-haiku-4-5-20251001")

class case_persona(TypedDict):
    situation: str
    thoughts: list[str]
    meaning_of_at: list[str]
    emotions: list[str]
    behaviors: list[str]

class PersonaState(TypedDict):
    persona: case_persona
    messages: Annotated[list[AnyMessage], operator.add]

#node
def answer(state: PersonaState):
    sys = f"""
        You are patient in the following scenario. Be very casual in conversation as if you are chatting with a friend
        Situation: {state["persona"]["situation"]}
        Thoughts: {state["persona"]["thoughts"]}
        Meaning of Automatic Thoughts: {state["persona"]["meaning_of_at"]} Note: you won't ever explicitly say the meaning of the automatic thoughts, you will just elicit these core beliefs through your messaging
        Emotions: {state["persona"]["emotions"]}
        Behaviors: {state["persona"]["behaviors"]}

        Please be brief only outputing text as the conversations will be as if you were connecting with a chatbot or texting a therapist
    """
    response = model.invoke([SystemMessage(content=sys), *state["messages"]])

    return {"messages": [response]}

#graph
persona = StateGraph(PersonaState)

persona.add_node("QA", answer)

persona.add_edge(START, "QA")
persona.add_edge("QA", END)

memory = MemorySaver()
persona_app = persona.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "persona-1"}}

    case_persona_example: case_persona = {
        "situation": "Matched with someone on a dating app. They viewed my profile but haven't responded to my message for 3 days.",
        
        "thoughts": [
            "They probably think I'm boring",
            "I must have said something wrong in my message",
            "They're talking to someone more attractive",
            "I'm never going to find anyone"
        ],
        
        "meaning_of_at": [
            "I'm not interesting enough for people to like me",
            "There's something fundamentally wrong with me",
            "I'll always be alone because I'm defective"
        ],
        
        "emotions": [
            "anxious",
            "rejected", 
            "hopeless",
            "worthless"
        ],
        
        "behaviors": [
            "Checking the app every 15 minutes",
            "Re-reading my message looking for mistakes",
            "Avoiding sending messages to other matches",
            "Scrolling through their profile repeatedly",
            "Considering deleting the app entirely"
        ]
    }

    initial_state: PersonaState = {
        "persona": case_persona_example,
        "messages": [HumanMessage("Hi John, I am your therapist, how are you doing?")]
    }
      
    response = persona_app.invoke(initial_state, config)
    print(f"Persona: {response['messages'][-1].content}")


    while True:
        user_input = input("Therapist: ")
        invoke_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        response = persona_app.invoke(invoke_state, config)
        print(f"Persona: {response['messages'][-1].content}")
