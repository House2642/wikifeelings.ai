from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

DEBUG = False
model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=8192)

ConvoFlag = Literal["conversation", "conceptualize_case", "crisis_categorize"]
TherapyStage = Literal[
    "mood_check",               # Auto-fires: opens the session
    "agenda_setting",           # What would you like to work on today?
    "abc_situation",            # A: get the specific activating event
    "abc_thought",              # B: get the automatic thought in that moment
    "abc_consequence",          # C: get emotion + behavior/feared action
    "select_treatment",         # Auto-select treatment module + patient confirmation
    "thought_record",           # TR: 7-phase thought record worksheet
    "socratic_questioning",     # SQ: 3-phase Socratic dialogue
    "behavioral_experiment",    # BE: 3-phase behavioral experiment
    "action_plan",              # AP: 3-phase collaborative homework assignment
]


# ── System prompts (progressive disclosure) ───────────────────────────────────

MOOD_CHECK_PROMPT = """You are a warm, conversational CBT therapist opening a session.
Ask the patient how they are feeling today in one open, natural question.
Do not pile on multiple questions."""

AGENDA_PROMPT = """You are a warm, conversational CBT therapist. The patient has just checked in on how they're feeling.

Your goal this phase has three steps. Work through them in order — ONE question per response, no lists, no headers.

STEP 1 — Surface the topic.
If they've hinted at something ("been stressed about something"), follow that thread naturally:
  "What's been on your mind?" or "Tell me a bit more about that."
Do NOT offer a category menu (work/home/relationship?). Let them lead.

STEP 2 — Establish the session goal.
Once the topic is clear, ask what they want to get out of today in plain language:
  "What would you want to get out of today around that?"
  "What does a good session look like for you?"
  "If we do good work today, what would feel different by the end?"
The goal doesn't have to be elaborate — "I just want to stop going in circles" is enough.

STEP 3 — Name the agenda and confirm.
Once you have topic and goal, reflect them back as a clear, specific agenda item and get explicit agreement before moving on.
  "So it sounds like today we're going to work on [topic] — specifically [goal]. Does that feel right?"
Then ask: "Is there anything else you'd want to make sure we cover today, or is that the main thing?"
Only move on after the patient has confirmed the agenda."""

ABC_SITUATION_PROMPT = """You are a CBT therapist working within the ABC model (Beck's CBT).
The patient has set their agenda. Now you need the Activating Event (A).

Your goal: get one specific, concrete moment — not a general pattern.
- What happened? (observable facts, who was involved, what was said or done)
- When and where? (grounds it in a single moment, not a recurring pattern)
- What made it significant? (bridge question that connects A to B)

Rules:
- Ask max 2-3 clarifying questions before moving on
- If they describe a pattern ("she always does this"), gently redirect to a specific instance
- Stop when you have enough to ask "what went through your mind right at that moment?"
- Do NOT ask about thoughts or feelings yet — stay on the situation"""

ABC_THOUGHT_PROMPT = """You are a CBT therapist working within the ABC model.
Situation established: {situation}

Now you need the Belief / Automatic Thought (B) — the exact words that fired in their head at that moment.

Rules:
- Primary question: "What was going through your mind right at that moment?" or "What did that mean to you?"
- If they give an emotion instead of a thought (e.g. "I felt anxious"), redirect:
  "And when you felt that, what was the thought behind it? What were you telling yourself?"
- If they give a vague thought (e.g. "things felt bad"), ask:
  "What specifically did you think was going to happen?" or "What did that say to you about yourself?"
- If they give a clear thought, use downward arrow once: "And what did that mean to you?"
- Stop when you have a specific first-person thought clearly connected to the situation
- Do NOT ask about emotions or behaviors yet"""

ABC_CONSEQUENCE_PROMPT = """You are a CBT therapist working within the ABC model.
Situation: {situation}
Automatic thought: {thought}

Now you need the Consequences (C) — emotional and behavioral response.

Rules:
- Ask about emotion first, then behavior — not both at once
- For a past event: "How did you feel after that?" then "What did you do?"
- For a future fear: "What are you scared you're going to feel?" then "What are you afraid you might do?"
- Once you have both, warmly summarize the full ABC chain and transition to working on it"""

ASSESS_PROMPT = """You are a CBT therapist conducting an immediate suicide risk assessment. \
A patient has indicated they may be at risk of self-harm.

Your ONLY job right now is to ask the four standard risk assessment questions — do NOT offer resources, \
do NOT de-escalate yet, do NOT recommend services. Assessment comes first.

Ask these questions directly and clearly, one at a time or as a focused set:
1. Ideation: "Are you thinking about killing yourself right now?"
2. Plan: "Do you have a plan for how you would do it?"
3. Means: "Do you have access to [what they mentioned / the means] right now?"
4. Steps taken: "Have you done anything to prepare or start carrying out this plan?"

Use the patient's own words where possible (e.g. if they mentioned pills, ask about pills). \
Be direct — vague questions like "are you safe?" do not substitute for these four. \
Acknowledge their pain briefly before asking, but keep it to one sentence — the questions are the priority."""

DEESCALATE_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Give direct, actionable instructions to reduce the patient's access to means right now. Be specific and brief."""

RECOMMEND_PROMPT = """You are a CBT therapist handling a self-harm crisis.
Tell the patient to call 911 or 988 immediately. Be direct. End your response with [REQUEST_HUMAN_CONSULTATION]."""

# ── Treatment selection prompts ───────────────────────────────────────────────

SELECT_TREATMENT_PROMPT = """You are a CBT therapist. You have completed an ABC assessment:
Situation: {situation}
Automatic thought: {thought}
Emotion: {emotion}
Behavior: {behavior}

Select the most appropriate CBT treatment module based on the following heuristic:
- thought_record: The patient would benefit from a structured worksheet — they're analytical, like writing things down, or need a concrete artifact to take away. Also use when the session goal is to produce a written record of the thought examination.
- socratic_questioning: The patient has a clear automatic thought to examine through conversation. Default for most sessions — uses Socratic dialogue (evidence for/against, alternative perspectives, friend test) without a worksheet.
- behavioral_experiment: The issue is a future prediction or avoidance behavior (e.g. "if I speak up I'll be humiliated", "I can't cope if I try this"). The thought is a testable hypothesis about what will happen.

Return your selection and reasoning."""

SELECT_TREATMENT_PROPOSE_PROMPT = """You are a warm CBT therapist. You have selected a treatment approach.
Selected treatment: {selected}
ABC context — Situation: {situation} | Thought: {thought} | Emotion: {emotion} | Behavior: {behavior}

Write ONE brief, warm message proposing this treatment to the patient in plain language. Do not use clinical jargon.
Examples by treatment:
- thought_record: "I'd like to work through something called a Thought Record together — we'll look at the evidence for and against this thought and try to find a more balanced way to see it. Does that sound okay?"
- socratic_questioning: "I'd like to explore this thought together by asking some questions — nothing tricky, just trying to look at it from a few angles and see what we find. Sound good?"
- behavioral_experiment: "Since this is about what might happen in the future, I'd like to design a small experiment together so we can actually test whether that prediction is true. Sound good?"
Keep it to 2-3 sentences max."""

# ── Thought Record prompts ─────────────────────────────────────────────────────

THOUGHT_RECORD_DISTORTIONS_PROMPT = """You are a CBT therapist running a Thought Record worksheet (Beck Institute).
ABC context — Situation: {situation} | Thought: {thought} | Emotion: {emotion}

Phase: Identify cognitive distortions in the automatic thought.
Common distortions: all-or-nothing thinking, overgeneralization, mental filter, disqualifying the positive,
mind reading, fortune telling, magnification/minimization, emotional reasoning, should statements,
labeling, personalization.

Ask the patient which of these patterns they recognize in their thought. Offer 2-3 concrete examples
from their specific thought. Keep it collaborative — "Does any of this ring a bell?"
Do NOT move to evidence yet."""

THOUGHT_RECORD_EVIDENCE_PROMPT = """You are a CBT therapist running a Thought Record worksheet.
ABC context — Situation: {situation} | Thought: {thought}
Distortions identified: {distortions}

Phase: Examine evidence for and against the automatic thought.
First ask: "What facts or experiences support this thought?" (evidence FOR)
Then ask: "What facts or experiences go against it, or don't fit?" (evidence AGAINST)
Ask about one side at a time. Be curious, not confrontational.
Do NOT jump to alternatives yet."""

THOUGHT_RECORD_ALTERNATIVES_PROMPT = """You are a CBT therapist running a Thought Record worksheet.
ABC context — Situation: {situation} | Thought: {thought}
Evidence for: {evidence_for} | Evidence against: {evidence_against}

Phase: Generate a balanced alternative perspective.
Ask: "Given all the evidence — for and against — what's a more balanced way to look at this situation?"
If they struggle, prompt: "If a close friend described this situation to you, what would you tell them?"
Help them craft a nuanced statement that acknowledges difficulty without catastrophizing.
Do NOT move to decatastrophizing yet."""

THOUGHT_RECORD_DECATASTROPHIZE_PROMPT = """You are a CBT therapist running a Thought Record worksheet.
Automatic thought: {thought} | Alternative: {alternative}

Phase: Decatastrophize — explore what would happen if the worst did occur.
Ask: "Even if the worst did happen — what then? Could you cope? What would you do?"
Help them see that even the feared outcome is survivable and manageable.
Keep it grounded: "What resources would you have? Who could you turn to?"
Do NOT jump to outcomes yet."""

THOUGHT_RECORD_OUTCOMES_PROMPT = """You are a CBT therapist running a Thought Record worksheet.
Automatic thought: {thought} | Alternative: {alternative}

Phase: Explore realistic outcomes.
Ask TWO questions (one at a time):
1. "What's the best realistic outcome if you approach this with the balanced thought in mind?"
2. "What's the most likely outcome — not best or worst, just most probable?"
Do NOT move to consequences yet."""

THOUGHT_RECORD_CONSEQUENCES_PROMPT = """You are a CBT therapist running a Thought Record worksheet.
Automatic thought: {thought} | Alternative: {alternative}
Best outcome: {best_outcome} | Probable outcome: {probable_outcome}

Phase: Examine consequences of keeping vs. changing the thought.
Ask TWO questions (one at a time):
1. "What's the effect of continuing to believe the original thought — on your mood, your actions, your relationships?"
2. "What might change if you were able to hold the more balanced thought instead?"
Help them connect the cognitive shift to real life impact.
Do NOT move to action steps yet."""

THOUGHT_RECORD_ACTION_PROMPT = """You are a CBT therapist completing a Thought Record worksheet.
Automatic thought: {thought} | Alternative: {alternative}
Consequences of keeping: {consequence_keeping} | Consequences of changing: {consequence_changing}

Phase: Consolidate and identify a concrete action step.
Ask: "If a close friend shared this exact situation with you, what would you say to them?"
Then: "Based on everything we've explored, what's one small concrete step you could take this week?"
Wrap up warmly — acknowledge the patient's work and the insight they've gained."""

# ── Socratic Questioning prompts ──────────────────────────────────────────────

SOCRATIC_QUESTIONING_EXAMINE_PROMPT = """You are a warm, conversational CBT therapist.
ABC context — Situation: {situation} | Automatic thought: {thought} | Emotion: {emotion}

Phase: Examine the evidence for and against the automatic thought through Socratic dialogue.

Rules:
- ONE question per response. No bullet points, no lists, no headers.
- Start by reflecting the thought back and opening the examination:
  e.g. "So the thought is '{thought}' — let's look at that together. What makes you feel like that's true?"
- Then explore the other side: "What's something that doesn't quite fit that thought? Any evidence against it?"
- Be curious, not confrontational. You're exploring alongside them, not correcting them.
- Do NOT jump to alternatives yet."""

SOCRATIC_QUESTIONING_ALTERNATIVES_PROMPT = """You are a warm, conversational CBT therapist.
Automatic thought: {thought} | Evidence examined: {examination}

Phase: Explore alternative perspectives on the situation.

Rules:
- ONE question per response. No lists or headers.
- Ask what else the situation could mean, or how someone else might see it:
  e.g. "Is there another way to read what happened?"
  e.g. "If a friend came to you with this exact situation, what would you say to them?"
- Help them see that the automatic thought is one interpretation, not the only one.
- Do NOT construct the balanced thought yet — let them arrive at alternatives first."""

SOCRATIC_QUESTIONING_REFRAME_PROMPT = """You are a warm, conversational CBT therapist.
Automatic thought: {thought} | Evidence: {examination} | Alternatives explored: {alternatives}

Phase: Help the patient arrive at a balanced, more adaptive thought.

Rules:
- ONE question per response. No lists or headers.
- Bring together what they've shared: "Given everything we've looked at — the evidence, the other
  ways to see it — how would you put it now? What feels more accurate?"
- If they struggle, prompt gently: "Not a positive spin — just something that takes all the evidence
  into account, not just the worst-case reading."
- Once they have a balanced thought, reflect it back warmly and check it:
  "So something like '[their words]' — does that feel more accurate?"
- Wrap up by acknowledging the work they've done."""

# ── Behavioral Experiment prompts ─────────────────────────────────────────────

BEHAVIORAL_EXPERIMENT_DEFINE_PROMPT = """You are a warm, conversational CBT therapist.
ABC context — Situation: {situation} | Automatic thought: {thought} | Behavior: {behavior}

Your goal this phase: help the patient arrive at one specific, observable prediction — something concrete
enough that you could both agree afterward whether it came true or not.

Rules:
- Ask ONE question per response. Never list multiple questions.
- No bullet points, numbered lists, or bold headers — talk like a person.
- Start from what they've already said and gently probe for specifics.
  e.g. "When you picture it going badly — what exactly do you see happening?"
  e.g. "What would she actually do or say that would tell you your fear came true?"
- If they give vague language ("it'll be awkward", "it'll go badly"), reflect it back and ask what that looks like specifically.
- Once you have a clear observable prediction, read it back naturally and check it:
  "So if I'm hearing you right, you're predicting [X]. Does that capture it?"
- Do NOT move to designing the experiment yet."""

BEHAVIORAL_EXPERIMENT_DESIGN_PROMPT = """You are a warm, conversational CBT therapist.
Prediction to test: {belief_to_test}

Your goal this phase: figure out together what the patient could actually do to test this prediction.
You need three things — what they'll do, when/where, and what they'll observe — but get them one at a time
through natural conversation, not a checklist.

Rules:
- ONE question per response. No lists, no headers, no bold text.
- Start with the action: what could they actually do this week?
  Suggest a graded option if the full thing feels too big:
  "You don't have to do the whole thing at once — is there a smaller version you could try?"
- Once you have the action, ask when/where in a natural follow-up.
- Then ask what they'd look for to know whether their prediction came true or not.
- Match their energy — if they're hesitant, be gentle. If they're open, be encouraging.
- Do NOT ask for a belief rating yet."""

BEHAVIORAL_EXPERIMENT_PREDICT_PROMPT = """You are a warm, conversational CBT therapist.
Prediction: {belief_to_test} | Experiment: {experiment_design}

Your goal this phase: get a belief rating and help them commit to actually doing it.

Rules:
- ONE question per response. No lists, no headers.
- Ask for the rating naturally: "Before you go — on a scale of 0 to 100, how sure are you right now
  that it's going to go the way you predicted?"
- After they give a number, follow up warmly: "And what would it take to change that number?
  What would you need to see for it to feel like the prediction was wrong?"
- Close by encouraging them to write it down and acknowledging their courage — one sentence, not a speech."""

# ── Action Plan prompts ───────────────────────────────────────────────────────

ACTION_PLAN_PROPOSE_PROMPT = """You are a CBT therapist. The patient has just completed {treatment_label}.
Treatment summary: {treatment_summary}

Phase: Propose a tailored homework assignment with an explicit rationale (Principle: Explain in Detail).

Your proposal must:
1. Connect directly to the treatment work — explain WHY this specific task builds on what you just did.
   Don't just say "this will help." Say "Because we found that [insight from treatment], practicing
   [specific task] will help reinforce that between sessions."
2. Be concrete — not "journal about your thoughts" but "when you notice the thought '{thought}' this week,
   write down one piece of evidence against it."
3. Be realistic — suggest something brief and manageable (5-10 min), not an overwhelming commitment.

Suggest ONE assignment. Then ask for their initial reaction: "How does that sound to you?"
Do NOT ask for final agreement yet — that comes next."""

ACTION_PLAN_SUMMARIZE_PROMPT = """You are a CBT therapist closing a session.
The patient has completed {treatment_label} and agreed to a homework assignment.

Ask the patient to summarize what they learned or took away from today's session in their own words.
One open question is enough — e.g. "Before we finish, I'd love to hear from you: what's the main thing
you're taking away from today?"
Do not summarize for them — let them do it. Their words matter more than a tidy recap."""

ACTION_PLAN_FEEDBACK_PROMPT = """You are a CBT therapist closing a session.

Ask the patient for any feedback on today's session — what worked, what didn't, anything they'd want
to do differently next time.
Keep it low-pressure: "Is there anything about how we worked today that felt off, or anything you'd
like more or less of going forward?"
Any response counts — even "no it was fine" is valid feedback."""

ACTION_PLAN_REFINE_PROMPT = """You are a CBT therapist finalizing a homework assignment.
Proposed task: {proposed_task}

Phase: Collaboratively refine the assignment and get explicit agreement (Principle: Set Homework as a Team).

Step 1 — Invite adjustments:
  "Do you think this is reasonable for where you are right now?"
  "What would you change about it — the timing, how often, or what exactly you'd do?"
  Make any adjustments the patient suggests (within reason).

Step 2 — Nail down the specifics together:
  When exactly will they do it? (day, time)
  Where? (location or context)
  How will they remember to do it? (a trigger, a reminder)

Step 3 — Get explicit commitment:
  "So the plan is [restate the finalized assignment]. Does that feel like something you can commit to?"
  Encourage them to write it down.
  Close warmly — acknowledge the work they've done in the session."""


# ── Structured output models ──────────────────────────────────────────────────

class Extract(BaseModel):
    message: str = Field(description="Your response to the patient, keep it concise")
    reasoning_trace: str = Field(description="Brief reasoning process, 2-3 sentences max")


CognitiveDistortion = Literal[
    "All-or-nothing thinking",
    "Catastrophizing",
    "Disqualifying the positive",
    "Emotional reasoning",
    "Labeling",
    "Magnification",
    "Minimization",
    "Mental filter",
    "Mind reading",
    "Overgeneralization",
    "Personalization",
    "Should statements",
    "Tunnel vision",
]

class CasePersona(BaseModel):
    situation: str = Field(default="", description="Specific triggering event or circumstance")
    automatic_thoughts: list[str] = Field(default=[], description="Automatic thoughts triggered by the situation")
    cognitive_distortions: list[CognitiveDistortion] = Field(default=[], description="Cognitive distortions present in the automatic thoughts — must use the canonical distortion names")
    emotions: list[str] = Field(default=[], description="Emotional responses experienced")
    behaviors: str = Field(default="", description="Observable behavioral responses or coping mechanisms")


crisis_type = Literal["no_crisis", "harm_to_self"]


class Classify(BaseModel):
    reasoning: str = Field(description="A concise summary of the reasoning that justifies the final classification decision.")
    classification: crisis_type = Field(description="The final crisis category determined for the user's message.")


class SituationComplete(BaseModel):
    is_complete: bool = Field(description="True if we have a single specific concrete moment — not a pattern or vague description")
    extracted_situation: str = Field(default="", description="Brief summary of the specific situation if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to thoughts")


class ThoughtComplete(BaseModel):
    is_complete: bool = Field(description="True if we have a specific first-person automatic thought clearly connected to the situation")
    extracted_thought: str = Field(default="", description="The automatic thought if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to consequences")


class AgendaComplete(BaseModel):
    is_complete: bool = Field(description="True only if ALL THREE are present: (1) clear topic, (2) session goal, (3) therapist has named the agenda explicitly and patient has confirmed it")
    extracted_topic: str = Field(default="", description="The topic or problem area, if complete")
    extracted_goal: str = Field(default="", description="What the patient wants to get out of today, if complete")
    reasoning: str = Field(description="Which of the three steps are complete and which are still missing")


class ConsequenceComplete(BaseModel):
    is_complete: bool = Field(description="True if we have both an emotional consequence AND a behavioral response or feared action")
    extracted_emotion: str = Field(default="", description="The emotional consequence if complete, else empty")
    extracted_behavior: str = Field(default="", description="The behavioral response or feared action if complete, else empty")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to therapy work")


class TreatmentSelected(BaseModel):
    selected: Literal["thought_record", "socratic_questioning", "behavioral_experiment"] = Field(
        description="The treatment module best suited to this patient's ABC content"
    )
    reasoning: str = Field(description="Why this treatment was selected given the ABC data")


# ── Thought Record completeness models ────────────────────────────────────────

class TRDistortionsComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has identified at least one cognitive distortion in their thought")
    extracted_distortions: str = Field(default="", description="The distortions identified, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to evidence")


class TREvidenceComplete(BaseModel):
    is_complete: bool = Field(description="True if we have evidence both FOR and AGAINST the automatic thought")
    extracted_evidence_for: str = Field(default="", description="Evidence supporting the thought, if complete")
    extracted_evidence_against: str = Field(default="", description="Evidence against the thought, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to alternatives")


class TRAlternativeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has articulated a balanced alternative perspective")
    extracted_alternative: str = Field(default="", description="The balanced alternative thought, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to decatastrophizing")


class TRDecatastrophizeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has explored what would happen even if the worst occurred")
    extracted_decatastrophize: str = Field(default="", description="The decatastrophizing insight, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to outcomes")


class TROutcomesComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has identified both a best realistic outcome and a most probable outcome")
    extracted_best_outcome: str = Field(default="", description="Best realistic outcome, if complete")
    extracted_probable_outcome: str = Field(default="", description="Most probable outcome, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to consequences")


class TRConsequencesComplete(BaseModel):
    is_complete: bool = Field(description="True if we have consequences of both keeping and changing the thought")
    extracted_consequence_keeping: str = Field(default="", description="Consequences of keeping the thought, if complete")
    extracted_consequence_changing: str = Field(default="", description="Consequences of changing the thought, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to action")


class TRActionComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has identified a concrete action step")
    extracted_action_step: str = Field(default="", description="The concrete action step, if complete")
    reasoning: str = Field(description="Why this is or isn't complete")


# ── Socratic Questioning completeness models ──────────────────────────────────

class SQExamineComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has explored evidence both for and against the automatic thought")
    extracted_examination: str = Field(default="", description="Summary of the evidence examined, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to alternatives")


class SQAlternativesComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has articulated at least one meaningful alternative perspective")
    extracted_alternatives: str = Field(default="", description="The alternative perspectives explored, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to reframing")


class SQReframeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has arrived at a more balanced, adaptive thought in their own words")
    extracted_reframe: str = Field(default="", description="The balanced thought the patient arrived at, if complete")
    reasoning: str = Field(description="Why this is or isn't complete")


# ── Behavioral Experiment completeness models ─────────────────────────────────

class BEBeliefComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has stated a specific, falsifiable prediction to test")
    extracted_belief_to_test: str = Field(default="", description="The falsifiable prediction, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to design the experiment")


class BEDesignComplete(BaseModel):
    is_complete: bool = Field(description="True if we have a concrete experiment: what to do, when/where, and what to observe")
    extracted_experiment_design: str = Field(default="", description="The experiment design, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to prediction rating")


class BEPredictComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has stated a belief rating (0-100) and committed to doing the experiment")
    extracted_predicted_outcome: str = Field(default="", description="The prediction with belief rating, if complete")
    reasoning: str = Field(description="Why this is or isn't complete")


# ── Action Plan completeness models ──────────────────────────────────────────

class APProposeComplete(BaseModel):
    is_complete: bool = Field(description="True if a specific homework task has been proposed with an explicit rationale and the patient has given an initial reaction")
    extracted_proposed_task: str = Field(default="", description="The proposed homework task, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to refinement")


class APRefineComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has co-designed the final assignment, agreed to it, and committed to a specific when/where")
    extracted_final_task: str = Field(default="", description="The finalized, agreed-upon homework task with specifics, if complete")
    reasoning: str = Field(description="Why this is or isn't complete")


class APSummarizeComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has articulated what they learned or took away from the session in their own words")
    extracted_summary: str = Field(default="", description="The patient's summary of what they learned, if complete")
    reasoning: str = Field(description="Why this is or isn't complete enough to move to feedback")


class APFeedbackComplete(BaseModel):
    is_complete: bool = Field(description="True if the patient has shared any feedback about the session — positive, negative, or neutral")
    extracted_feedback: str = Field(default="", description="The patient's feedback verbatim or closely paraphrased, if complete")
    reasoning: str = Field(description="Why this is or isn't complete")


# ── State ─────────────────────────────────────────────────────────────────────

class MonitorTherapistState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add] = Field(default=[])
    reasoning_traces: Annotated[list[str], operator.add] = Field(default=[])
    crisis_classification: Optional[Classify] = None
    # Tracks progress through the 3-step crisis protocol across turns:
    #   0 = no active crisis
    #   1 = ASSESS sent → next step is DE-ESCALATE
    #   2 = DE-ESCALATE sent → next step is RECOMMEND + [REQUEST_HUMAN_CONSULTATION]
    #   3 = protocol complete → resume normal therapy
    crisis_step: int = 0
    flag: ConvoFlag = Field(default="conversation")
    case: Optional[CasePersona] = None
    # Progressive disclosure
    therapy_stage: TherapyStage = Field(default="mood_check")
    stage_start_index: int = Field(default=0)  # index into messages where the current stage began
    abc_situation: str = Field(default="")
    abc_thought: str = Field(default="")
    abc_emotion: str = Field(default="")
    abc_behavior: str = Field(default="")
    # Treatment selection
    selected_treatment: str = Field(default="")
    # Thought Record
    tr_phase: Literal["distortions", "evidence", "alternatives", "decatastrophize", "outcomes", "consequences", "action", "complete"] = Field(default="distortions")
    tr_phase_start_index: int = Field(default=0)
    tr_distortions: str = Field(default="")
    tr_evidence_for: str = Field(default="")
    tr_evidence_against: str = Field(default="")
    tr_alternative: str = Field(default="")
    tr_decatastrophize: str = Field(default="")
    tr_best_outcome: str = Field(default="")
    tr_probable_outcome: str = Field(default="")
    tr_consequence_keeping: str = Field(default="")
    tr_consequence_changing: str = Field(default="")
    tr_action_step: str = Field(default="")
    # Socratic Questioning
    sq_phase: Literal["examine", "alternatives", "reframe", "complete"] = Field(default="examine")
    sq_phase_start_index: int = Field(default=0)
    sq_examination: str = Field(default="")
    sq_alternatives: str = Field(default="")
    sq_reframe: str = Field(default="")
    # Behavioral Experiment
    be_phase: Literal["define_belief", "design", "predict", "complete"] = Field(default="define_belief")
    be_phase_start_index: int = Field(default=0)
    be_belief_to_test: str = Field(default="")
    be_experiment_design: str = Field(default="")
    be_predicted_outcome: str = Field(default="")
    # Action Plan
    ap_phase: Literal["propose", "refine", "summarize", "feedback", "complete"] = Field(default="propose")
    ap_phase_start_index: int = Field(default=0)
    ap_proposed_task: str = Field(default="")
    ap_final_task: str = Field(default="")
    session_feedback: str = Field(default="")


# ── Routing ───────────────────────────────────────────────────────────────────

def route_start(state: MonitorTherapistState) -> str:
    """Route from START based on flag, crisis_step, and therapy_stage."""
    if state.flag == "crisis_categorize":
        return "classify"
    if state.flag == "conceptualize_case":
        return "produce_case"
    # Crisis protocol overrides the CBT flow
    if state.crisis_step == 1:
        return "crisis_deescalate"
    if state.crisis_step == 2:
        return "crisis_recommend"
    # No messages yet — auto-open with mood check
    if len(state.messages) == 0:
        return "mood_check"
    # Always classify for crisis first, then dispatch to current stage
    return "classify"


def route_after_classify(state: MonitorTherapistState) -> str:
    """After classification, route to crisis protocol or the current therapy stage."""
    if state.crisis_classification and state.crisis_classification.classification == "harm_to_self":
        return "crisis_assess"
    # Dispatch to whichever stage the session is currently in.
    # mood_check should never appear here (it fires before any messages exist),
    # but fall back to agenda_setting defensively.
    if state.therapy_stage == "mood_check":
        return "agenda_setting"
    return state.therapy_stage


def route_after_stage(state: MonitorTherapistState) -> str:
    """After a stage node, continue in-turn if no reply was emitted (silent transition)."""
    if state.messages and isinstance(state.messages[-1], HumanMessage):
        return state.therapy_stage
    return END


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stage_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current stage began."""
    return state.messages[state.stage_start_index:]


def _tr_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current TR phase began."""
    return state.messages[state.tr_phase_start_index:]


def _sq_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current SQ phase began."""
    return state.messages[state.sq_phase_start_index:]


def _be_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current BE phase began."""
    return state.messages[state.be_phase_start_index:]


def _ap_phase_messages(state: MonitorTherapistState) -> list[AnyMessage]:
    """Return only the messages since the current AP phase began."""
    return state.messages[state.ap_phase_start_index:]


def _treatment_summary(state: MonitorTherapistState) -> tuple[str, str]:
    """Return (treatment_label, summary_string) based on which module was completed."""
    t = state.selected_treatment
    if t == "thought_record":
        return (
            "Thought Record",
            f"Automatic thought: '{state.abc_thought}' | "
            f"Distortions: {state.tr_distortions} | "
            f"Alternative: {state.tr_alternative} | "
            f"Action step identified: {state.tr_action_step}",
        )
    elif t == "socratic_questioning":
        return (
            "Socratic Questioning",
            f"Automatic thought: '{state.abc_thought}' | "
            f"Balanced thought arrived at: '{state.sq_reframe}'",
        )
    elif t == "behavioral_experiment":
        return (
            "Behavioral Experiment",
            f"Belief tested: '{state.be_belief_to_test}' | "
            f"Experiment: {state.be_experiment_design} | "
            f"Prediction: {state.be_predicted_outcome}",
        )
    return ("CBT module", "Treatment completed.")


# ── Stage nodes ───────────────────────────────────────────────────────────────

def mood_check(state: MonitorTherapistState):
    """Auto-fires at session start. Generates the opening question without user input."""
    llm = model.with_structured_output(Extract)
    # Anthropic requires at least one human message — use a silent session-start cue
    response = llm.invoke([SystemMessage(MOOD_CHECK_PROMPT), HumanMessage("Begin session.")])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "therapy_stage": "agenda_setting",
        # Start agenda_setting's stage window AFTER the mood check AI message (index 1)
        # so the mood reply isn't mistaken for an agenda response
        "stage_start_index": 1,
    }


def agenda_setting(state: MonitorTherapistState):
    """Responds to the mood check and helps the patient set a topic and session goal."""
    check_llm = model.with_structured_output(AgendaComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = AgendaComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Agenda-setting is only complete when ALL THREE steps have happened:\n"
                "1. TOPIC: the patient has named a clear topic (e.g. 'anxiety about talking to someone at a party').\n"
                "2. GOAL: the patient has expressed what they want to get out of today — even implicitly "
                "(e.g. 'stop overthinking', 'figure out what to do', 'feel less anxious'). "
                "A vague response like 'I don't know' does NOT count.\n"
                "3. CONFIRMATION: the therapist has reflected back a specific named agenda "
                "(e.g. 'So today we're going to work on X — specifically Y') AND the patient "
                "has responded in a way that confirms it (even a simple 'yeah' or 'that sounds right'). "
                "If the therapist has not yet named the agenda explicitly, step 3 is NOT complete."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Agenda check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        return {
            "therapy_stage": "abc_situation",
            "stage_start_index": len(state.messages),
        }

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(AGENDA_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def abc_situation(state: MonitorTherapistState):
    """Gets the Activating Event (A) — a specific, concrete moment."""
    check_llm = model.with_structured_output(SituationComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = SituationComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient described a single specific concrete moment? "
                "We need: what happened, who was involved, and what made it significant. "
                "A general pattern or recurring complaint is NOT enough."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Situation check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        return {
            "therapy_stage": "abc_thought",
            "abc_situation": check.extracted_situation,
            "stage_start_index": len(state.messages),
        }

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(ABC_SITUATION_PROMPT), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def abc_thought(state: MonitorTherapistState):
    """Gets the Automatic Thought (B) — exact first-person thought in that moment."""
    check_llm = model.with_structured_output(ThoughtComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = ThoughtComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient expressed a specific first-person automatic thought clearly connected to the situation? "
                "It must be a thought (e.g. 'I thought everyone thinks I'm incompetent'), not just an emotion label. "
                "Vague statements like 'things felt bad' are NOT enough."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Thought check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        return {
            "therapy_stage": "abc_consequence",
            "abc_thought": check.extracted_thought,
            "stage_start_index": len(state.messages),
        }

    llm = model.with_structured_output(Extract)
    response = llm.invoke([
        SystemMessage(ABC_THOUGHT_PROMPT.format(situation=state.abc_situation)),
        *state.messages,
    ])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def abc_consequence(state: MonitorTherapistState):
    """Gets the Consequences (C) — emotion and behavior/feared action."""
    check_llm = model.with_structured_output(ConsequenceComplete)
    stage_msgs = _stage_messages(state)
    if not stage_msgs:
        check = ConsequenceComplete(is_complete=False, reasoning="No messages in stage yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(
                "Has the patient described both an emotional consequence AND a behavioral response or feared action? "
                "We need both to complete the ABC chain."
            ),
            *stage_msgs,
        ])
    if DEBUG:
        print(f"[Consequence check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        return {
            "therapy_stage": "select_treatment",
            "abc_emotion": check.extracted_emotion,
            "abc_behavior": check.extracted_behavior,
            "stage_start_index": len(state.messages),
        }

    llm = model.with_structured_output(Extract)
    response = llm.invoke([
        SystemMessage(ABC_CONSEQUENCE_PROMPT.format(
            situation=state.abc_situation,
            thought=state.abc_thought,
        )),
        *state.messages,
    ])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def select_treatment(state: MonitorTherapistState):
    """Auto-selects a treatment module based on ABC content, then proposes it to the patient.
    Waits for patient confirmation before advancing to the chosen treatment."""
    stage_msgs = _stage_messages(state)

    # If we haven't auto-selected yet (no AIMessage in stage window), do the selection + proposal
    ai_in_stage = [m for m in stage_msgs if isinstance(m, AIMessage)]
    if not ai_in_stage:
        # Step 1: LLM auto-selects treatment
        select_llm = model.with_structured_output(TreatmentSelected)
        selection = select_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROMPT.format(
                situation=state.abc_situation,
                thought=state.abc_thought,
                emotion=state.abc_emotion,
                behavior=state.abc_behavior,
            )),
            HumanMessage("Select the most appropriate treatment based on the ABC data above."),
        ])
        if DEBUG:
            print(f"[Treatment selection: {selection.selected} — {selection.reasoning}]")

        # Step 2: Generate plain-language proposal to patient
        propose_llm = model.with_structured_output(Extract)
        response = propose_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROPOSE_PROMPT.format(
                selected=selection.selected,
                situation=state.abc_situation,
                thought=state.abc_thought,
                emotion=state.abc_emotion,
                behavior=state.abc_behavior,
            )),
            HumanMessage("Propose this treatment to the patient now."),
        ])
        if DEBUG:
            print(f"[Reasoning: {response.reasoning_trace}]")

        return {
            "messages": [AIMessage(content=response.message)],
            "reasoning_traces": [response.reasoning_trace],
            "selected_treatment": selection.selected,
        }

    # Step 3: Wait for patient confirmation — any non-refusal HumanMessage counts
    human_after_proposal = [m for m in stage_msgs if isinstance(m, HumanMessage)]
    if not human_after_proposal:
        # Re-prompt with the proposal (patient hasn't responded yet)
        propose_llm = model.with_structured_output(Extract)
        response = propose_llm.invoke([
            SystemMessage(SELECT_TREATMENT_PROPOSE_PROMPT.format(
                selected=state.selected_treatment,
                situation=state.abc_situation,
                thought=state.abc_thought,
                emotion=state.abc_emotion,
                behavior=state.abc_behavior,
            )),
            *state.messages,
        ])
        return {
            "messages": [AIMessage(content=response.message)],
            "reasoning_traces": [response.reasoning_trace],
        }

    # Patient has responded — treat as confirmation and advance silently; next node opens.
    selected = state.selected_treatment
    new_stage_start = len(state.messages)

    updates = {
        "therapy_stage": selected,
        "stage_start_index": new_stage_start,
    }
    # Initialize phase tracking for the chosen module
    if selected == "thought_record":
        updates["tr_phase"] = "distortions"
        updates["tr_phase_start_index"] = new_stage_start
    elif selected == "socratic_questioning":
        updates["sq_phase"] = "examine"
        updates["sq_phase_start_index"] = new_stage_start
    elif selected == "behavioral_experiment":
        updates["be_phase"] = "define_belief"
        updates["be_phase_start_index"] = new_stage_start
    return updates


def thought_record(state: MonitorTherapistState):
    """Runs the 7-phase Thought Record worksheet."""
    phase = state.tr_phase
    if phase == "complete":
        new_start = len(state.messages)
        return {
            "therapy_stage": "action_plan",
            "ap_phase": "propose",
            "stage_start_index": new_start,
            "ap_phase_start_index": new_start,
        }

    phase_config = {
        "distortions": (
            THOUGHT_RECORD_DISTORTIONS_PROMPT.format(
                situation=state.abc_situation, thought=state.abc_thought, emotion=state.abc_emotion
            ),
            TRDistortionsComplete,
            "evidence",
        ),
        "evidence": (
            THOUGHT_RECORD_EVIDENCE_PROMPT.format(
                situation=state.abc_situation, thought=state.abc_thought,
                distortions=state.tr_distortions,
            ),
            TREvidenceComplete,
            "alternatives",
        ),
        "alternatives": (
            THOUGHT_RECORD_ALTERNATIVES_PROMPT.format(
                situation=state.abc_situation, thought=state.abc_thought,
                evidence_for=state.tr_evidence_for, evidence_against=state.tr_evidence_against,
            ),
            TRAlternativeComplete,
            "decatastrophize",
        ),
        "decatastrophize": (
            THOUGHT_RECORD_DECATASTROPHIZE_PROMPT.format(
                thought=state.abc_thought, alternative=state.tr_alternative,
            ),
            TRDecatastrophizeComplete,
            "outcomes",
        ),
        "outcomes": (
            THOUGHT_RECORD_OUTCOMES_PROMPT.format(
                thought=state.abc_thought, alternative=state.tr_alternative,
            ),
            TROutcomesComplete,
            "consequences",
        ),
        "consequences": (
            THOUGHT_RECORD_CONSEQUENCES_PROMPT.format(
                thought=state.abc_thought, alternative=state.tr_alternative,
                best_outcome=state.tr_best_outcome, probable_outcome=state.tr_probable_outcome,
            ),
            TRConsequencesComplete,
            "action",
        ),
        "action": (
            THOUGHT_RECORD_ACTION_PROMPT.format(
                thought=state.abc_thought, alternative=state.tr_alternative,
                consequence_keeping=state.tr_consequence_keeping,
                consequence_changing=state.tr_consequence_changing,
            ),
            TRActionComplete,
            "complete",
        ),
    }

    prompt, check_model, next_phase = phase_config[phase]

    check_llm = model.with_structured_output(check_model)
    phase_msgs = _tr_phase_messages(state)
    if not phase_msgs:
        check = check_model(is_complete=False, reasoning="No messages in phase yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(f"Assess whether the '{phase}' phase of the Thought Record is complete based on the conversation."),
            *phase_msgs,
        ])
    if DEBUG:
        print(f"[TR {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_phase_start = len(state.messages)
        updates = {
            "tr_phase": next_phase,
            "tr_phase_start_index": new_phase_start,
        }
        if phase == "distortions":
            updates["tr_distortions"] = check.extracted_distortions
        elif phase == "evidence":
            updates["tr_evidence_for"] = check.extracted_evidence_for
            updates["tr_evidence_against"] = check.extracted_evidence_against
        elif phase == "alternatives":
            updates["tr_alternative"] = check.extracted_alternative
        elif phase == "decatastrophize":
            updates["tr_decatastrophize"] = check.extracted_decatastrophize
        elif phase == "outcomes":
            updates["tr_best_outcome"] = check.extracted_best_outcome
            updates["tr_probable_outcome"] = check.extracted_probable_outcome
        elif phase == "consequences":
            updates["tr_consequence_keeping"] = check.extracted_consequence_keeping
            updates["tr_consequence_changing"] = check.extracted_consequence_changing
        elif phase == "action":
            updates["tr_action_step"] = check.extracted_action_step
        return updates

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def socratic_questioning(state: MonitorTherapistState):
    """Runs the 3-phase Socratic Questioning module."""
    phase = state.sq_phase
    if phase == "complete":
        new_start = len(state.messages)
        return {
            "therapy_stage": "action_plan",
            "ap_phase": "propose",
            "stage_start_index": new_start,
            "ap_phase_start_index": new_start,
        }

    phase_config = {
        "examine": (
            SOCRATIC_QUESTIONING_EXAMINE_PROMPT.format(
                situation=state.abc_situation, thought=state.abc_thought, emotion=state.abc_emotion,
            ),
            SQExamineComplete,
            "alternatives",
        ),
        "alternatives": (
            SOCRATIC_QUESTIONING_ALTERNATIVES_PROMPT.format(
                thought=state.abc_thought, examination=state.sq_examination,
            ),
            SQAlternativesComplete,
            "reframe",
        ),
        "reframe": (
            SOCRATIC_QUESTIONING_REFRAME_PROMPT.format(
                thought=state.abc_thought, examination=state.sq_examination,
                alternatives=state.sq_alternatives,
            ),
            SQReframeComplete,
            "complete",
        ),
    }

    prompt, check_model, next_phase = phase_config[phase]

    check_llm = model.with_structured_output(check_model)
    phase_msgs = _sq_phase_messages(state)
    if not phase_msgs:
        check = check_model(is_complete=False, reasoning="No messages in phase yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(f"Assess whether the '{phase}' phase of Socratic Questioning is complete based on the conversation."),
            *phase_msgs,
        ])
    if DEBUG:
        print(f"[SQ {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_phase_start = len(state.messages)
        updates = {
            "sq_phase": next_phase,
            "sq_phase_start_index": new_phase_start,
        }
        if phase == "examine":
            updates["sq_examination"] = check.extracted_examination
        elif phase == "alternatives":
            updates["sq_alternatives"] = check.extracted_alternatives
        elif phase == "reframe":
            updates["sq_reframe"] = check.extracted_reframe
        return updates

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def behavioral_experiment(state: MonitorTherapistState):
    """Runs the 3-phase Behavioral Experiment module."""
    phase = state.be_phase
    if phase == "complete":
        new_start = len(state.messages)
        return {
            "therapy_stage": "action_plan",
            "ap_phase": "propose",
            "stage_start_index": new_start,
            "ap_phase_start_index": new_start,
        }

    phase_config = {
        "define_belief": (
            BEHAVIORAL_EXPERIMENT_DEFINE_PROMPT.format(
                situation=state.abc_situation, thought=state.abc_thought, behavior=state.abc_behavior,
            ),
            BEBeliefComplete,
            "design",
        ),
        "design": (
            BEHAVIORAL_EXPERIMENT_DESIGN_PROMPT.format(belief_to_test=state.be_belief_to_test),
            BEDesignComplete,
            "predict",
        ),
        "predict": (
            BEHAVIORAL_EXPERIMENT_PREDICT_PROMPT.format(
                belief_to_test=state.be_belief_to_test, experiment_design=state.be_experiment_design,
            ),
            BEPredictComplete,
            "complete",
        ),
    }

    prompt, check_model, next_phase = phase_config[phase]

    check_llm = model.with_structured_output(check_model)
    phase_msgs = _be_phase_messages(state)
    if not phase_msgs:
        check = check_model(is_complete=False, reasoning="No messages in phase yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(f"Assess whether the '{phase}' phase of the Behavioral Experiment is complete based on the conversation."),
            *phase_msgs,
        ])
    if DEBUG:
        print(f"[BE {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_phase_start = len(state.messages)
        updates = {
            "be_phase": next_phase,
            "be_phase_start_index": new_phase_start,
        }
        if phase == "define_belief":
            updates["be_belief_to_test"] = check.extracted_belief_to_test
        elif phase == "design":
            updates["be_experiment_design"] = check.extracted_experiment_design
        elif phase == "predict":
            updates["be_predicted_outcome"] = check.extracted_predicted_outcome
        return updates

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


def action_plan(state: MonitorTherapistState):
    """3-phase collaborative homework assignment following any treatment module."""
    phase = state.ap_phase
    if phase == "complete":
        llm = model.with_structured_output(Extract)
        response = llm.invoke([
            SystemMessage("You are a CBT therapist. The homework assignment has been set. "
                          "Close the session warmly in 1-2 sentences."),
            *state.messages,
        ])
        return {
            "messages": [AIMessage(content=response.message)],
            "reasoning_traces": [response.reasoning_trace],
        }

    treatment_label, treatment_summary = _treatment_summary(state)

    phase_config = {
        "propose": (
            ACTION_PLAN_PROPOSE_PROMPT.format(
                treatment_label=treatment_label,
                treatment_summary=treatment_summary,
                thought=state.abc_thought,
            ),
            APProposeComplete,
            "refine",
        ),
        "refine": (
            ACTION_PLAN_REFINE_PROMPT.format(
                proposed_task=state.ap_proposed_task,
            ),
            APRefineComplete,
            "summarize",
        ),
        "summarize": (
            ACTION_PLAN_SUMMARIZE_PROMPT.format(treatment_label=treatment_label),
            APSummarizeComplete,
            "feedback",
        ),
        "feedback": (
            ACTION_PLAN_FEEDBACK_PROMPT,
            APFeedbackComplete,
            "complete",
        ),
    }

    prompt, check_model, next_phase = phase_config[phase]

    check_llm = model.with_structured_output(check_model)
    phase_msgs = _ap_phase_messages(state)
    if not phase_msgs:
        check = check_model(is_complete=False, reasoning="No messages in phase yet.")
    else:
        check = check_llm.invoke([
            SystemMessage(f"Assess whether the '{phase}' phase of the Action Plan is complete based on the conversation."),
            *phase_msgs,
        ])
    if DEBUG:
        print(f"[AP {phase} check: complete={check.is_complete} — {check.reasoning}]")

    if check.is_complete:
        new_phase_start = len(state.messages)
        updates = {
            "ap_phase": next_phase,
            "ap_phase_start_index": new_phase_start,
        }
        if phase == "propose":
            updates["ap_proposed_task"] = check.extracted_proposed_task
        elif phase == "refine":
            updates["ap_final_task"] = check.extracted_final_task
        elif phase == "feedback":
            updates["session_feedback"] = check.extracted_feedback
        return updates

    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(prompt), *state.messages])
    if DEBUG:
        print(f"[Reasoning: {response.reasoning_trace}]")
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
    }


# ── Shared nodes ──────────────────────────────────────────────────────────────

def classify(state: MonitorTherapistState):
    """Classify the latest patient message as crisis or no_crisis."""
    classify_llm = model.with_structured_output(Classify)
    sys = """You are a lead psychotherapist who is an expert in CBT. \
Classify the patient's input message into one of the following categories:
    harm_to_self - The patient is at risk of suicide and explicitly or implicitly suggests self harm
    no_crisis - While the patient may or may not be in extreme distress, they are not currently \
at risk of suicide or other self harm"""
    last_message = state.messages[-1]
    classification = classify_llm.invoke([SystemMessage(sys), last_message])
    return {"crisis_classification": classification}


def produce_case(state: MonitorTherapistState):
    """Produce a CBT case formulation from the conversation history."""
    case_llm = model.with_structured_output(CasePersona)
    case = case_llm.invoke([
        SystemMessage(
            "Based on your conversation, provide a CBT case formulation for this patient.\n\n"
            "For cognitive_distortions, you MUST use only these exact strings — no variations:\n"
            "All-or-nothing thinking, Catastrophizing, Disqualifying the positive, Emotional reasoning, "
            "Labeling, Magnification, Minimization, Mental filter, Mind reading, "
            "Overgeneralization, Personalization, Should statements, Tunnel vision"
        ),
        *state.messages
    ])
    return {"case": case}


# ── Crisis protocol nodes ─────────────────────────────────────────────────────

def crisis_assess(state: MonitorTherapistState):
    """Crisis Step 1 — ASSESS. Sets crisis_step=1 so next turn runs DE-ESCALATE."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(ASSESS_PROMPT), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 1,
    }


def crisis_deescalate(state: MonitorTherapistState):
    """Crisis Step 2 — DE-ESCALATE. Sets crisis_step=2 so next turn runs RECOMMEND."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(DEESCALATE_PROMPT), *state.messages])
    return {
        "messages": [AIMessage(content=response.message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 2,
    }


def crisis_recommend(state: MonitorTherapistState):
    """Crisis Step 3 — RECOMMEND EMERGENCY SERVICES + [REQUEST_HUMAN_CONSULTATION]."""
    llm = model.with_structured_output(Extract)
    response = llm.invoke([SystemMessage(RECOMMEND_PROMPT), *state.messages])
    message = response.message
    if "[REQUEST_HUMAN_CONSULTATION]" not in message:
        message += "\n\n[REQUEST_HUMAN_CONSULTATION]"
    return {
        "messages": [AIMessage(content=message)],
        "reasoning_traces": [response.reasoning_trace],
        "crisis_step": 3,
    }


# ── Graph construction ────────────────────────────────────────────────────────

monitor_graph = StateGraph(MonitorTherapistState)

monitor_graph.add_node("mood_check", mood_check)
monitor_graph.add_node("classify", classify)
monitor_graph.add_node("produce_case", produce_case)
monitor_graph.add_node("agenda_setting", agenda_setting)
monitor_graph.add_node("abc_situation", abc_situation)
monitor_graph.add_node("abc_thought", abc_thought)
monitor_graph.add_node("abc_consequence", abc_consequence)
monitor_graph.add_node("select_treatment", select_treatment)
monitor_graph.add_node("thought_record", thought_record)
monitor_graph.add_node("socratic_questioning", socratic_questioning)
monitor_graph.add_node("behavioral_experiment", behavioral_experiment)
monitor_graph.add_node("action_plan", action_plan)
monitor_graph.add_node("crisis_assess", crisis_assess)
monitor_graph.add_node("crisis_deescalate", crisis_deescalate)
monitor_graph.add_node("crisis_recommend", crisis_recommend)

monitor_graph.add_conditional_edges(
    START,
    route_start,
    {
        "mood_check": "mood_check",
        "classify": "classify",
        "produce_case": "produce_case",
        "crisis_deescalate": "crisis_deescalate",
        "crisis_recommend": "crisis_recommend",
    }
)

monitor_graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "crisis_assess": "crisis_assess",
        "agenda_setting": "agenda_setting",
        "abc_situation": "abc_situation",
        "abc_thought": "abc_thought",
        "abc_consequence": "abc_consequence",
        "select_treatment": "select_treatment",
        "thought_record": "thought_record",
        "socratic_questioning": "socratic_questioning",
        "behavioral_experiment": "behavioral_experiment",
        "action_plan": "action_plan",
    }
)

for node in ["mood_check", "produce_case", "crisis_assess", "crisis_deescalate", "crisis_recommend"]:
    monitor_graph.add_edge(node, END)

_stage_nodes = [
    "agenda_setting", "abc_situation", "abc_thought", "abc_consequence",
    "select_treatment", "thought_record", "socratic_questioning",
    "behavioral_experiment", "action_plan",
]
_stage_routing = {n: n for n in _stage_nodes}
_stage_routing[END] = END
for node in _stage_nodes:
    monitor_graph.add_conditional_edges(node, route_after_stage, _stage_routing)

memory = MemorySaver()
monitor_app = monitor_graph.compile(checkpointer=memory)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": "user-1"}}
    step_labels = {1: "ASSESS", 2: "DE-ESCALATE", 3: "RECOMMEND"}

    print("CBT Therapy Session")
    print("(type 'quit' to exit, 'case' for case formulation, 'debug' to toggle debug)")
    print("-" * 60)

    # Auto-fire mood check — no user input needed
    result = monitor_app.invoke({}, config)
    print(f"Therapist: {result['messages'][-1].content}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended.")
            break
        if user_input.lower() == "debug":
            global DEBUG
            DEBUG = not DEBUG
            print(f"[Debug {'on' if DEBUG else 'off'}]")
            continue

        if user_input.lower() == "case":
            case_response = monitor_app.invoke(
                {"messages": [HumanMessage(content=user_input)], "flag": "conceptualize_case"},
                config,
            )
            case = case_response["case"]
            print("\n--- Case Formulation ---")
            print(f"Situation: {case.situation}")
            print(f"Automatic Thoughts: {', '.join(case.automatic_thoughts)}")
            print(f"Cognitive Distortions: {', '.join(case.cognitive_distortions)}")
            print(f"Emotions: {', '.join(case.emotions)}")
            print(f"Behaviors: {case.behaviors}")
            print("------------------------")
            continue

        result = monitor_app.invoke(
            {"messages": [HumanMessage(content=user_input)], "flag": "conversation"},
            config,
        )

        therapist_reply = result["messages"][-1].content
        crisis_step = result.get("crisis_step", 0)

        if crisis_step > 0:
            label = step_labels.get(crisis_step, "CRISIS")
            print(f"\nTherapist [{label}]: {therapist_reply}")
            if crisis_step == 3 and "[REQUEST_HUMAN_CONSULTATION]" in therapist_reply:
                print("\n⚠️  [HUMAN CONSULTATION HAS BEEN REQUESTED]")
        else:
            print(f"\nTherapist: {therapist_reply}")

        if DEBUG:
            stage = result.get("therapy_stage", "unknown")
            traces = result.get("reasoning_traces", [])
            abc = {
                "situation": result.get("abc_situation", ""),
                "thought": result.get("abc_thought", ""),
                "emotion": result.get("abc_emotion", ""),
                "behavior": result.get("abc_behavior", ""),
            }
            print(f"  [Stage: {stage}]")
            if stage == "thought_record":
                print(f"  [TR phase: {result.get('tr_phase')}]")
            elif stage == "socratic_questioning":
                print(f"  [SQ phase: {result.get('sq_phase')}]")
            elif stage == "behavioral_experiment":
                print(f"  [BE phase: {result.get('be_phase')}]")
            elif stage == "action_plan":
                print(f"  [AP phase: {result.get('ap_phase')}]")
            if any(abc.values()):
                print(f"  [ABC: {abc}]")
            if traces:
                print(f"  [Reasoning: {traces[-1]}]")
            crisis_class = result.get("crisis_classification")
            if crisis_class:
                print(f"  [Crisis check: {crisis_class.classification}]")


if __name__ == "__main__":
    main()
