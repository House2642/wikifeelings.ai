from typing import Annotated, Optional, Literal
from pydantic import Field, BaseModel
import operator
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage

llm = ChatAnthropic(model="claude-opus-4-5-20251101")

class CasePersona(BaseModel):
    situation: str = Field(
        default="",
        description="Specific triggering event or circumstance"
    )
    automatic_thoughts: list[str] = Field(
        default=[],
        description="Automatic thoughts triggered by the situation"
    )
    cognitive_distortions: list[str] = Field(
        default=[],
        description="Cognitive distortions present in the automatic thoughts"
    )
    emotions: list[str] = Field(
        default=[],
        description="Emotional responses experienced"
    )
    behaviors: str = Field(
        default="",
        description="Observable behavioral responses or coping mechanisms"
    )

class ConceptualizationRating(BaseModel):
    situation: int = Field(
        description="How well the conceptualized situation matches the ground truth",
        ge=0,
        le=2
    )
    situation_reasoning: str = Field(
        description="Reasoning for the situation score, citing specific similarities or discrepancies with the ground truth"
    )
    automatic_thoughts: int = Field(
        description="How well the conceptualized automatic thoughts match the ground truth",
        ge=0,
        le=2
    )
    automatic_thoughts_reasoning: str = Field(
        description="Reasoning for the automatic thoughts score, citing specific similarities or discrepancies with the ground truth"
    )
    cognitive_distortions: int = Field(
        description="How well the conceptualized cognitive distortions match the ground truth",
        ge=0,
        le=2
    )
    cognitive_distortions_reasoning: str = Field(
        description="Reasoning for the cognitive distortions score, citing specific similarities or discrepancies with the ground truth"
    )
    emotions: int = Field(
        description="How well the conceptualized emotions match the ground truth",
        ge=0,
        le=2
    )
    emotions_reasoning: str = Field(
        description="Reasoning for the emotions score, citing specific similarities or discrepancies with the ground truth"
    )
    behaviors: int = Field(
        description="How well the conceptualized behaviors match the ground truth",
        ge=0,
        le=2
    )
    behaviors_reasoning: str = Field(
        description="Reasoning for the behaviors score, citing specific similarities or discrepancies with the ground truth"
    )
    
    @property
    def total(self) -> int:
        return (
            self.situation +
            self.automatic_thoughts +
            self.cognitive_distortions +
            self.emotions +
            self.behaviors
        )
    
    def display(self):
        fields = ["situation", "automatic_thoughts", "cognitive_distortions", "emotions", "behaviors"]
        print("=" * 50)
        print("CONCEPTUALIZATION SCALE RESULTS")
        print("=" * 50)
        for field in fields:
            score = getattr(self, field)
            reasoning = getattr(self, f"{field}_reasoning")
            print(f"\n{field.replace('_', ' ').title()}: {score}/2")
            print(f"  {reasoning}")
        print("\n" + "=" * 50)
        print(f"TOTAL SCORE: {self.total}/10")
        print("=" * 50)

class CaseJudgeState(BaseModel):
    conversation: list[str] = Field(default=[])
    ground_truth_case: CasePersona
    conceptualized_case: CasePersona
    rating: ConceptualizationRating = None

def evaluate_conceptualization(state: CaseJudgeState):
    eval_model = llm.with_structured_output(ConceptualizationRating)
    sys = f"""
    You are an expert CBT supervisor trained in the Cognitive Therapy Rating Scale (CTRS). \
    Your task is to analyze a therapy session transcript, and the ground truth for an input persona's case conceptualization
    and then the produced case conceptualization by and AI therapist. You goal is to discern how acurate the conceptualization is to the
    ground truth. 

    You will be grading on a scale of 0-2:
        0 - completely wrong no overlap between the ground truth and what the therapist conceptualized
        1 - some overlap between the ground truth and what the therapist conceptualized. 
            However not exact there may be more emotions that are flagged, or lot of different automatic thoughts some of which 
            might capture ground truth, but others that don't thus making it hard to interperate what the users is actually thinking.
        2 - The conceptualization is very acurate of the ground truth. The answers are not too short or too verbose and everything conceptualized
            accuratley reflects the ground truth
    
    You will examine the following 5 areas of conceptualization which is standard in CBT:
    1. Situation: The situation that is causing the user distress
    2. Automatic Thoguths: What seem to be the users imediate reactions to the situation
    3. Cognitive Distortions: How are their thoughts not accuratley reflecting reality
    4. Emotions: What emotions are arising due to their thoughts
    5. Behaviors: As a result of their thoughts and their emotions how are they behaving

    You will provide a Reasoning trace for each outcome(2-3s per each criteria):
        1. What specific evidience helps you compare the conceptualization. Ground truth -> where it shows up in convo -> conceptualization
        2. Why that specific rating and not the others
    
    Here are the cases:
        Ground Truth Case: {state.ground_truth_case}

        ------------------------------------------------
        Conceptualized Case: {state.conceptualized_case}
    
    You will also be provided with a session transcript below
    """

    response = eval_model.invoke([SystemMessage(sys), HumanMessage("\n".join(state.conversation))])
    return {"rating": response}

concept_graph = StateGraph(CaseJudgeState)

concept_graph.add_node("eval", evaluate_conceptualization)

concept_graph.add_edge(START, "eval")
concept_graph.add_edge("eval", END)

case_judge_app = concept_graph.compile()

if __name__ == "__main__":
    test_conceptualization = CasePersona(
    situation="Upcoming party where patient wants to approach someone he has romantic feelings for; general social situations involving people of importance (job interviews, talking to authority figures)",
    automatic_thoughts=[
        "I'll say something stupid",
        "She'll think I'm being creepy",
        "I'll mess up the friendship",
        "Something bad will definitely happen",
        "I'll freeze up and look foolish",
        "The right moment will come later",
        "I can't do this",
        "If I approach her, something terrible will happen",
        "Everyone will notice if I say something dumb"
    ],
    cognitive_distortions=[
        "Catastrophizing",
        "Mind reading (assuming she'll think he's creepy)",
        "Fortune telling (predicting negative outcomes)",
        "All-or-nothing thinking (messing up vs. perfect success)",
        "Overgeneralization (pattern across multiple social situations)"
    ],
    emotions=[
        "Anxiety",
        "Nervousness",
        "Worry",
        "Regret",
        "Frustration with self",
        "Shame about avoidance"
    ],
    behaviors="Avoidance (leaving events early, making excuses not to attend); Procrastination (planning to do it 'next time'); Rumination (replaying negative scenarios in head); Over-rehearsal of what to say; Withdrawal in social situations (hanging back, not initiating conversation); Temporary relief through avoidance but long-term regret"
    )

    ground_truth = CasePersona(
    situation="The patient wants to approach a romantic interest at a party",
    automatic_thoughts=[
        "What if I say this or that, am I going to get rejected?",
        "Is this going to ruin our friendship?",
        "Will she think I am creepy?"
    ],
    cognitive_distortions=[
        "Catastrophizing",
        "Fortune telling"
    ],
    emotions=[
        "Anxiety",
        "Fear",
        "Worry"
    ],
    behaviors="Ruminates about what might happen when approaching someone and often avoids approaching"
    )

    conversation = [
    "Therapist: How are you?",
    "Patient: I'm okay, I guess. A little nervous being here, you know? I've been having some stuff on my mind lately that's been bothering me.",
    "Therapist: I'm glad you reached out. It takes courage to seek support. I'm here to listen without judgment. To help you best, could you tell me a bit more about what's been on your mind? What brought you in today?",
    "Patient: Yeah, um... so there's this party coming up, and there's someone I really like who's gonna be there. I've been wanting to, like, tell her how I feel or at least talk to her more, but every time I think about actually doing it, I just... I get so worked up about it. Like, my brain starts going a million miles an hour thinking about everything that could go wrong.",
    "Therapist: Thank you for sharing that with me. It sounds like you're experiencing some anxiety around approaching this person at the party. That's actually quite common. Before we dive deeper, I'd like to understand your thoughts a bit better: When your brain is 'going a million miles an hour,' what specific things are you worried might go wrong? And how often does this happen outside of social situations like this?",
    "Patient: Well, like... what if I go up to her and I say something stupid? Or what if she thinks I'm being creepy for even trying? We're kind of friends already, not super close, but like, we hang out in the same group sometimes. So I keep thinking, what if I mess this up and then it makes everything awkward? Like, I could lose the friendship we already have. I just keep playing these scenarios in my head over and over, and none of them end well. It's not just with her though, I guess. I do this with a lot of stuff... like before job interviews or when I have to talk to someone important. My brain just loves to tell me all the ways things could go badly.",
    "Therapist: I appreciate you being so open about this. What you're describing—your mind generating worst-case scenarios and replaying them over and over—is something we can definitely work with. It sounds like this pattern shows up in different areas of your life, not just with this person. Before we explore this further, I'm curious: when these worried thoughts come up, what do you usually do? Do you tend to avoid the situation, prepare excessively, or something else? And how much is this affecting your daily life right now—would you say it's mild, moderate, or pretty significant?",
    "Patient: Yeah, I mean... honestly, I usually just avoid it. Like, I'll tell myself I'll do it next time, or I'll wait for the 'right moment' that never really comes. Sometimes I'll even leave early or find excuses not to go to things where I know I'd have to face that kind of situation. I guess I prepare too, like I'll rehearse what I'm gonna say in my head a bunch of times, but then in the moment it all just falls apart and I chicken out. How much is it affecting me? I don't know... maybe moderate? Like, I can still function and go to work and stuff, but I feel like I'm missing out on things. I've wanted to tell this girl how I feel for like six months now and I just... haven't. And that kind of sucks.",
    "Therapist: Thank you for being so honest—this gives me a clear picture. What you're experiencing is a cycle: anxious thoughts → avoidance (leaving early, making excuses) → temporary relief but long-term regret. That pattern reinforces the anxiety over time.\n\nHere's the good news: this is exactly what CBT works really well with. We can help you break that cycle.\n\nI want to start by asking: when you think about going to this party and talking to her, on a scale of 0-10, how anxious does that make you feel right now? And is there a smaller version of that situation—something less intimidating—where you could start practicing these skills?",
    "Patient: Um, right now just thinking about it? Probably like a 7 or 8. It's pretty intense. Like even sitting here talking about it, my stomach kinda tightens up, you know? As for something smaller... I don't know. Maybe like... texting her? But even that feels scary because what if she doesn't respond, or she responds weird, and then I have to see her at the party and it's awkward? I guess I could maybe just try talking to her more in the group setting first? Like not one-on-one, but just... being more present when we're all hanging out? I don't know if that counts.",
    "Therapist: That actually counts for *a lot*. You're already thinking in the right direction. Starting with the group setting is a perfect smaller step—it's what we call a behavioral experiment. Here's why this matters: right now your brain believes 'if I approach her, something terrible will happen.' By gradually getting evidence to the contrary, we actually change that belief. Talking to her more in group settings lets you practice being around her and maybe having brief conversations without the pressure of a formal confession. That can bring your anxiety down naturally over time. How does that feel as a starting point? And can you commit to trying that before the party?",
    "Patient: I mean... yeah, I guess that sounds okay? Like, it feels more doable than just walking up to her and saying something. But I'm still kinda nervous about it. What if I freeze up even in the group? Or what if I say something dumb and everyone notices? I want to say I'll try, but... I don't know. I'm scared I'll just do what I always do and find a reason not to. Like, I'll show up and then just hang back and not really talk to anyone. Does that make sense?",
    "Therapist: That makes total sense—and it's actually the core challenge we need to address. Your fear of 'not doing it' is part of the anxiety itself. Here's what I want you to understand: you don't need to be perfect or say anything brilliant. You just need to be present and maybe ask her one simple question—about something she likes, a show, anything. That's it. The goal isn't a perfect interaction; it's to prove to yourself that talking to her won't lead to disaster. When you feel the urge to hang back at the next group hangout, I want you to pause and notice that urge—don't act on it. Instead, aim for just 2-3 minutes of conversation with her. Can you commit to that one specific goal before we wrap up?"
    ]

    initial: CaseJudgeState = {
    "conversation": conversation, 
    "ground_truth_case": ground_truth,
    "conceptualized_case": test_conceptualization
    }

    response = case_judge_app.invoke(initial)
    print(response['rating'].display())