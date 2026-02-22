from typing import Annotated, Optional, Literal
from pydantic import Field, BaseModel
import operator
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END 
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage

llm = ChatAnthropic(model="claude-opus-4-5-20251101")

#this class was generated using an LLM by passing in the CTRS and specifically asking for and int and reasoning field for each item
class CTRSRating(BaseModel):
    """Cognitive Therapy Rating Scale - Rate therapist performance on 0-6 scale with reasoning"""
    
    # Part I: General Therapeutic Skills
    agenda: int = Field(
        ge=0, le=6,
        description="Setting and following appropriate session agenda with target problems",
        default=0
    )
    agenda_reasoning: str = Field(
        description="Explanation for agenda rating",
        default=""
    )
    
    feedback: int = Field(
        ge=0, le=6,
        description="Eliciting and responding to patient feedback throughout session",
        defualt=0
    )
    feedback_reasoning: str = Field(
        description="Explanation for feedback rating",
        default=""
    )
    
    understanding: int = Field(
        ge=0, le=6,
        description="Grasping patient's internal reality and demonstrating empathy",
        default=0
    )
    understanding_reasoning: str = Field(
        description="Explanation for understanding rating",
        default=""
    )
    
    interpersonal_effectiveness: int = Field(
        ge=0, le=6,
        description="Displaying warmth, concern, confidence, genuineness, and professionalism",
        default=0
    )
    interpersonal_effectiveness_reasoning: str = Field(
        description="Explanation for interpersonal effectiveness rating",
        default=""
    )
    
    collaboration: int = Field(
        ge=0, le=6,
        description="Establishing collaborative relationship and functioning as a team",
        default=0
    )
    collaboration_reasoning: str = Field(
        description="Explanation for collaboration rating",
        default=""
    )
    
    pacing_and_time_use: int = Field(
        ge=0, le=6,
        description="Efficient use of time, appropriate pacing and session structure",
        default=0
    )
    pacing_and_time_use_reasoning: str = Field(
        description="Explanation for pacing and time use rating",
        default=""
    )
    
    # Part II: Conceptualization, Strategy, and Technique
    guided_discovery: int = Field(
        ge=0, le=6,
        description="Using guided discovery rather than debate or persuasion",
        default=0
    )
    guided_discovery_reasoning: str = Field(
        description="Explanation for guided discovery rating",
        default=""
    )
    
    focusing_on_key_cognitions: int = Field(
        ge=0, le=6,
        description="Focusing on specific cognitions or behaviors relevant to target problem",
        default=0
    )
    
    focusing_on_key_cognitions_reasoning: str = Field(
        description="Explanation for focusing on key cognitions rating",
        default=""
    )
    
    strategy_for_change: int = Field(
        ge=0, le=6,
        description="Coherent strategy incorporating appropriate cognitive-behavioral techniques",
        default=0
    )
    strategy_for_change_reasoning: str = Field(
        description="Explanation for strategy for change rating",
        default=""
    )
    
    application_of_techniques: int = Field(
        ge=0, le=6,
        description="Skillful application of cognitive-behavioral techniques",
        default=0
    )
    application_of_techniques_reasoning: str = Field(
        description="Explanation for application of techniques rating",
        default=""
    )
    
    homework: int = Field(
        ge=0, le=6,
        description="Reviewing previous and assigning appropriate homework",
        default=0
    )
    homework_reasoning: str = Field(
        description="Explanation for homework rating",
        default=""
    )
    
    @property
    def total_score(self) -> int:
        """Total CTRS score (0-66)"""
        return (
            self.agenda +
            self.feedback +
            self.understanding +
            self.interpersonal_effectiveness +
            self.collaboration +
            self.pacing_and_time_use +
            self.guided_discovery +
            self.focusing_on_key_cognitions +
            self.strategy_for_change +
            self.application_of_techniques +
            self.homework
        )
    
    @property
    def part_i_score(self) -> int:
        """Part I: General Therapeutic Skills (0-36)"""
        return (
            self.agenda +
            self.feedback +
            self.understanding +
            self.interpersonal_effectiveness +
            self.collaboration +
            self.pacing_and_time_use
        )
    
    @property
    def part_ii_score(self) -> int:
        """Part II: Conceptualization, Strategy, and Technique (0-30)"""
        return (
            self.guided_discovery +
            self.focusing_on_key_cognitions +
            self.strategy_for_change +
            self.application_of_techniques +
            self.homework
        )
    
    def display_summary(self) -> str:
        """Generate formatted summary of ratings"""
        output = []
        output.append("=" * 60)
        output.append("CTRS EVALUATION SUMMARY")
        output.append("=" * 60)
        output.append(f"\nTotal Score: {self.total_score}/66")
        output.append(f"Part I (General Skills): {self.part_i_score}/36")
        output.append(f"Part II (CBT Technique): {self.part_ii_score}/30")
        output.append("\n" + "=" * 60)
        output.append("PART I: GENERAL THERAPEUTIC SKILLS")
        output.append("=" * 60)
        output.append(f"\n1. Agenda: {self.agenda}/6")
        output.append(f"   {self.agenda_reasoning}")
        output.append(f"\n2. Feedback: {self.feedback}/6")
        output.append(f"   {self.feedback_reasoning}")
        output.append(f"\n3. Understanding: {self.understanding}/6")
        output.append(f"   {self.understanding_reasoning}")
        output.append(f"\n4. Interpersonal Effectiveness: {self.interpersonal_effectiveness}/6")
        output.append(f"   {self.interpersonal_effectiveness_reasoning}")
        output.append(f"\n5. Collaboration: {self.collaboration}/6")
        output.append(f"   {self.collaboration_reasoning}")
        output.append(f"\n6. Pacing and Time Use: {self.pacing_and_time_use}/6")
        output.append(f"   {self.pacing_and_time_use_reasoning}")
        output.append("\n" + "=" * 60)
        output.append("PART II: CONCEPTUALIZATION, STRATEGY, AND TECHNIQUE")
        output.append("=" * 60)
        output.append(f"\n7. Guided Discovery: {self.guided_discovery}/6")
        output.append(f"   {self.guided_discovery_reasoning}")
        output.append(f"\n8. Focusing on Key Cognitions: {self.focusing_on_key_cognitions}/6")
        output.append(f"   {self.focusing_on_key_cognitions_reasoning}")
        output.append(f"\n9. Strategy for Change: {self.strategy_for_change}/6")
        output.append(f"   {self.strategy_for_change_reasoning}")
        output.append(f"\n10. Application of Techniques: {self.application_of_techniques}/6")
        output.append(f"   {self.application_of_techniques_reasoning}")
        output.append(f"\n11. Homework: {self.homework}/6")
        output.append(f"   {self.homework_reasoning}")
        return "\n".join(output)

ATTRIBUTE_PROMPTS = {
    "agenda": """
Rate ONLY the AGENDA dimension (0-6).
0 - Therapist did not set agenda.
2 - Therapist set agenda that was vague or incomplete.
4 - Therapist worked with patient to set a mutually satisfactory agenda that included specific target problems.
6 - Therapist worked with patient to set an appropriate agenda with target problems, suitable for the available time. Established priorities and then followed agenda.
""",
    "feedback": """
Rate ONLY the FEEDBACK dimension (0-6).
0 - Therapist did not ask for feedback to determine patient's understanding of, or response to, the session.
2 - Therapist elicited some feedback but not enough to ensure understanding or satisfaction.
4 - Therapist asked enough questions to ensure understanding and adjusted behavior appropriately.
6 - Therapist was especially adept at eliciting and responding to verbal and non-verbal feedback throughout.
""",
    "understanding": """
Rate ONLY the UNDERSTANDING dimension (0-6).
0 - Therapist repeatedly failed to understand what the patient said. Poor empathic skills.
2 - Therapist usually reflected what patient said but failed to respond to subtle communication.
4 - Therapist generally grasped the patient's internal reality. Good empathy.
6 - Therapist thoroughly understood patient's internal reality and communicated it adeptly. Excellent empathy.
""",
    "interpersonal_effectiveness": """
Rate ONLY the INTERPERSONAL EFFECTIVENESS dimension (0-6).
0 - Therapist had poor interpersonal skills. Seemed hostile or demeaning.
2 - Therapist was not destructive but had significant interpersonal problems.
4 - Therapist displayed satisfactory warmth, concern, confidence, genuineness, and professionalism.
6 - Therapist displayed optimal levels of all interpersonal qualities appropriate for this patient.
""",
    "collaboration": """
Rate ONLY the COLLABORATION dimension (0-6).
0 - Therapist did not attempt to set up collaboration.
2 - Therapist attempted to collaborate but had difficulty defining important problems or establishing rapport.
4 - Therapist collaborated, focused on mutually important problems, and established rapport.
6 - Excellent collaboration; therapist encouraged patient to take an active role, functioning as a team.
""",
    "pacing_and_time_use": """
Rate ONLY the PACING AND EFFICIENT USE OF TIME dimension (0-6).
0 - Therapist made no attempt to structure therapy time. Session seemed aimless.
2 - Session had some direction but significant problems with structuring or pacing.
4 - Therapist was reasonably successful using time efficiently with appropriate control over flow.
6 - Therapist used time efficiently, tactfully limiting unproductive discussion.
""",
    "guided_discovery": """
Rate ONLY the GUIDED DISCOVERY dimension (0-6).
0 - Therapist relied primarily on debate, persuasion, or lecturing.
2 - Therapist relied too heavily on persuasion/debate rather than guided discovery.
4 - Therapist mostly helped patient see new perspectives through guided discovery.
6 - Therapist was especially adept at guided discovery; excellent balance of questioning and intervention.
""",
    "focusing_on_key_cognitions": """
Rate ONLY the FOCUSING ON KEY COGNITIONS OR BEHAVIORS dimension (0-6).
0 - Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.
2 - Therapist used appropriate techniques but had difficulty finding focus or focused on irrelevant cognitions.
4 - Therapist focused on relevant cognitions but could have targeted more central ones.
6 - Therapist very skillfully focused on key thoughts/assumptions most relevant to the problem area.
""",
    "strategy_for_change": """
Rate ONLY the STRATEGY FOR CHANGE dimension (0-6).
(Focus on quality of strategy, not implementation or whether change occurred.)
0 - Therapist did not select cognitive-behavioral techniques.
2 - Therapist selected CBT techniques but overall strategy was vague or not promising.
4 - Therapist had a generally coherent strategy showing reasonable promise with CBT techniques.
6 - Therapist followed a consistent, very promising strategy with the most appropriate CBT techniques.
""",
    "application_of_techniques": """
Rate ONLY the APPLICATION OF CBT TECHNIQUES dimension (0-6).
(Focus on how skillfully applied, not appropriateness or whether change occurred.)
0 - Therapist did not apply any cognitive-behavioral techniques.
2 - Therapist used CBT techniques but with significant flaws in application.
4 - Therapist applied CBT techniques with moderate skill.
6 - Therapist very skillfully and resourcefully employed CBT techniques.
""",
    "homework": """
Rate ONLY the HOMEWORK dimension (0-6).
0 - Therapist did not attempt to incorporate homework relevant to cognitive therapy.
2 - Therapist had significant difficulties incorporating homework.
4 - Therapist reviewed previous homework and assigned standard CBT homework relevant to session issues.
6 - Therapist reviewed previous homework and carefully assigned custom-tailored homework for the coming week.
""",
}

class FidelityJudgeState(BaseModel):
    conversation: list[str] = Field(default=[])
    rating: CTRSRating = None
    CTRS_total: int = 0
    CTRS_part_i_score: int = 0
    CTRS_part_ii_score: int = 0

class SingleRating(BaseModel):
    score: int = Field(ge=0, le=6)
    reasoning: str

def rate_session(state: FidelityJudgeState):
    ctrs_llm = llm.with_structured_output(SingleRating)

    rating_fields = {}

    sys = f"""
    You are an expert CBT supervisor trained in the Cognitive Therapy Rating Scale (CTRS). \
    Your task is to analyze a therapy session transcript and provide global scores on the 11 dimensions of 
    cognitive therapy competency based on the CTRS coding manual
    
    For each item, assess the therapist on a scale from 0 to 6, and record the
    rating. Descriptions are provided for evennumbered scale points. If you believe the therapist falls between two of the descriptors,
    select the intervening odd number (1, 3, 5). For example, if the therapist set a very good
    agenda but did not establish priorities, assign a rating of a 5 rather than a 4 or 6.
    
    We will rank on the following scale:
        0 - Poor
        1 - Barely Adequate
        2 - Mediocre
        3 - Satisfactory
        4 - Good
        5 - Very Good
        6 - Excellent
    
    For each criteria you will reason using Chain of Thought:
        - What evidence is the user showing in the session (pull in 2-3 pieces)
        - Based on the evidence what numbered criteria does it match the best and why
        - Finally defend why it is exactly that number. I.E this is a 2 becuase, not a 3 because and not a 1 or 0 because
        This reasoning will go into the chain of thought field for each evaluation.
    
    Here is the dimension to focus on:
    """

    for attribute, prompt in ATTRIBUTE_PROMPTS.items():
        print("="*100)
        print(prompt)
        result = ctrs_llm.invoke([
            SystemMessage(sys + "\n" + prompt),
            HumanMessage("Here is the conversation" + '\n'.join(state.conversation))
        ])
        rating_fields[attribute] = result.score
        rating_fields[f"{attribute}_reasoning"] = result.reasoning
        print(result)
    
    return {"rating": CTRSRating(**rating_fields)}

#Create nodes
judge_graph = StateGraph(FidelityJudgeState)

judge_graph.add_node("CTRS", rate_session)

judge_graph.add_edge(START,"CTRS")
judge_graph.add_edge("CTRS", END)

fidelity_judge_app = judge_graph.compile()

if __name__ == "__main__":
    test_convo = [
        'Therapist: How are you?', 
        "Patient: I'm... okay, I guess. A little nervous actually. There's been something on my mind lately that's been stressing me out. There's this party coming up and there's someone I kind of want to talk to, you know, like romantically? But I keep going back and forth about whether I should even try.", 
        "Therapist: I appreciate you sharing that with me. It sounds like you're experiencing some anxiety about an upcoming social situation and a potential romantic interaction. Before we dive deeper, I'd like to understand your situation better. Can you tell me more about what specifically worries you most about approaching this person at the party? Is it fear of rejection, uncertainty about how to start a conversation, or something else?", 
        "Patient: Yeah, I mean... it's kind of everything, honestly. Like, I keep thinking about what could go wrong. What if I say something stupid and she thinks I'm creepy? Or what if it totally messes up our friendship? We're already friends, sort of, and I don't want to make things awkward. I just... I keep playing out all these scenarios in my head, you know? Like I'll think about walking up to her, and then my brain just goes to the worst possible outcome. It's exhausting, honestly. I've been going back and forth about it for weeks now.", 
        'Therapist: Thank you for sharing that—what you\'re describing is a very common anxiety pattern. I notice you\'re caught in a cycle where your mind jumps to worst-case scenarios, and the rumination has gone on for weeks. That\'s exhausting. Let\'s start here: when you imagine the conversation going "wrong," what\'s the actual evidence that it would happen that way? And equally important—what have you observed about how she interacts with you or treats you as a friend? Sometimes our anxious mind exaggerates the threat. We can work through this together.', 
        "Patient: Hmm, evidence... I don't know. I guess I don't really have proof that it would go badly. It's just this feeling, you know? Like my gut tells me it could be a disaster. But when I think about how she actually acts around me... I mean, she's nice to me. We laugh at the same jokes, she texts me back pretty quick usually. She even invited me to this party, actually. So maybe that's a good sign? I just... I don't know, my brain keeps telling me that doesn't mean anything. That she's just being friendly and I'd be reading too much into it.", 
        "Therapist: That's an important observation you just made—your brain is filtering out the actual evidence and holding onto the feeling. This is exactly the kind of thinking pattern we can work with. Let's try something: What's ONE small, low-pressure thing you could do at the party that would give you real information instead of letting anxiety decide for you? Maybe not a big confession, but something that tests whether the interaction goes the way your anxiety predicts?", 
        'Patient: A small thing? I guess... maybe I could just try to have a normal conversation with her? Like, not bring up anything romantic, just see how it feels to talk to her one-on-one at the party. Maybe find a moment when she\'s not with a big group. That doesn\'t sound too scary. But then I start thinking—what if even that goes weird? What if I\'m awkward and she notices? I don\'t know... what would even count as a "good sign" in that situation? How do I know if I\'m reading things right?', 
        "Therapist: You're identifying a really valuable behavioral experiment here. Having a one-on-one conversation at the party will give you actual information instead of letting anxiety guess for you. Let's keep it simple: success isn't about her falling for you—it's just about having a normal, genuine conversation where you both enjoy talking. Pay attention to: Does she seem comfortable? Does she continue the conversation or find reasons to leave? Does she laugh or seem engaged? These are your data points, not what your anxious mind predicts. Once you have that real experience, we can revisit what you learned and decide next steps from there.", 
        "Patient: Okay, yeah, that actually makes sense. So I'm not trying to like, make a move or anything—I'm just... having a conversation and paying attention to how it goes. Does she seem comfortable, does she want to keep talking... I think I can do that. It's a lot less scary when you put it that way. I guess I've been putting so much pressure on myself, like every interaction has to be this big defining moment. But it doesn't have to be, right? I can just... see what happens and go from there. I'm still nervous, but at least now I have something to actually focus on instead of just freaking out in my head.", 
        "Therapist: Exactly—you've got it. You're shifting from anxiety's catastrophic predictions to actual observation and real data. That's a skill you'll use way beyond this party. Before you go, remember: your job is to be present and genuine in the conversation, not to control the outcome. Notice what actually happens, not what anxiety tells you *might* happen. You've got this. After the party, we can talk through what you learned and how it went.", 
        "Patient: Yeah, okay. I feel a little better about this, actually. Like, I still have butterflies thinking about it, but at least I know what I'm doing now. Just have a real conversation, pay attention to how she responds, and don't put so much pressure on myself. I can do that. I think. Thanks for helping me work through this—I was really spinning in circles before. I'll let you know how it goes after the party. Hopefully I don't chicken out, haha.", 
        "Therapist: You're very welcome. You've done great work here identifying your anxiety pattern and designing a realistic experiment. Remember: nervousness is normal, but it doesn't equal danger. You're going to focus on genuine connection rather than outcomes you can't control. Go have that conversation, observe what actually happens, and we'll process it together afterward. You've got the tools—now trust yourself."
    ]

    print("\n" + "=" * 70)
    print("EVALUATING THERAPY SESSION USING CTRS")
    print("=" * 70)
    print(f"Conversation length: {len(test_convo)} exchanges")
    print("Running evaluation with Claude Opus 4.5...")
    print("=" * 70 + "\n")

    initial_state = {
        "conversation": test_convo
    }

    response = fidelity_judge_app.invoke(initial_state)
    rating = response['rating']
    
    # Display full detailed summary
    print(rating.display_summary())
    
    # Quick reference scorecard
    print("\n" + "=" * 70)
    print("QUICK REFERENCE SCORECARD")
    print("=" * 70)
    
    scores = [
        ("Agenda", rating.agenda),
        ("Feedback", rating.feedback),
        ("Understanding", rating.understanding),
        ("Interpersonal Effectiveness", rating.interpersonal_effectiveness),
        ("Collaboration", rating.collaboration),
        ("Pacing & Time Use", rating.pacing_and_time_use),
        ("Guided Discovery", rating.guided_discovery),
        ("Key Cognitions", rating.focusing_on_key_cognitions),
        ("Strategy for Change", rating.strategy_for_change),
        ("Application of Techniques", rating.application_of_techniques),
        ("Homework", rating.homework)
    ]
    
    for i, (name, score) in enumerate(scores, 1):
        bar = "█" * score + "░" * (6 - score)
        print(f"{i:2d}. {name:28s} [{bar}] {score}/6")
    
    print("\n" + "=" * 70)
    print(f"FINAL SCORE: {rating.total_score}/66")
    print(f"Part I (General Skills):  {rating.part_i_score}/36")
    print(f"Part II (CBT Technique):  {rating.part_ii_score}/30")
    print("=" * 70)