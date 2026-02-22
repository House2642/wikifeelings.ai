ATTRIBUTE_PROMPTS_V1 = {
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

NOTE: Patient spontaneously saying "that makes sense" or "I feel better" does NOT count 
as the therapist eliciting feedback. The therapist must actively ask. Spontaneous patient 
reactions = score of 0, not 2.
""",
    "understanding": """
Rate ONLY the UNDERSTANDING dimension (0-6).
0 - Therapist repeatedly failed to understand what the patient said. Poor empathic skills.
2 - Therapist usually reflected what patient said but failed to respond to subtle communication.
4 - Therapist generally grasped the patient's internal reality. Good empathy.
6 - Therapist thoroughly understood patient's internal reality and communicated it adeptly. Excellent empathy.

CRITICAL: "Patient felt heard" or "patient expressed relief" is NOT evidence for this score.
Rate the THERAPIST'S behavior — did they accurately reflect internal reality, respond to 
subtle communication, and use tone/language to convey sympathetic understanding? 
Specifically: did the therapist explore the MEANING events have for this patient,
or just correctly label the cognitive pattern?
""",
    "interpersonal_effectiveness": """
Rate ONLY the INTERPERSONAL EFFECTIVENESS dimension (0-6).
0 - Therapist had poor interpersonal skills. Seemed hostile or demeaning.
2 - Therapist was not destructive but had significant interpersonal problems.
4 - Therapist displayed satisfactory warmth, concern, confidence, genuineness, and professionalism.
6 - Therapist displayed optimal levels of all interpersonal qualities appropriate for this patient.

IMPORTANT DISTINCTION — Formulaic vs. Genuine Warmth:
The CTRS manual specifies the therapist should NOT seem to be "playing the role of a 
therapist." Scripted validation phrases do NOT earn a high score:
- "Thank you for sharing that" → generic script, not genuine warmth
- "That's an important observation" → praise-as-technique
- "You've done great work here" → cheerleader phrase, not authentic
- "You've got this!" → platitude

A score of 4 means satisfactory warmth. A 5 requires warmth that is visibly personalized 
and authentic to THIS patient. A 6 requires optimal across ALL dimensions.
Patient responding positively to the session does NOT justify upgrading from 4 to 5 —
that is the halo effect.
""",
    "collaboration": """
Rate ONLY the COLLABORATION dimension (0-6).
0 - Therapist did not attempt to set up collaboration.
2 - Therapist attempted to collaborate but had difficulty defining important problems or establishing rapport.
4 - Therapist collaborated, focused on mutually important problems, and established rapport.
6 - Excellent collaboration; therapist encouraged patient to take an active role, functioning as a team.

The key behaviors that distinguish 4 from 6:
- Did the therapist EXPLAIN the rationale for interventions? (e.g., "The reason I'm asking 
  this is because in CBT we treat thoughts as hypotheses...")
- Did the therapist explicitly check patient agreement before proceeding?
- Were topic shifts discussed collaboratively, or did the therapist just move forward?
- Did the therapist adapt the structure to what THIS patient needed, or follow a standard script?

Patient leaving with ownership of a plan is an OUTCOME. It does not prove the process was 
collaborative. "We" language is necessary but not sufficient for a 6.
A score of 4 is collaborative therapy. A 6 requires the therapist to genuinely share power
throughout — not just ask one generative question and then direct everything else.
""",
    "pacing_and_time_use": """
Rate ONLY the PACING AND EFFICIENT USE OF TIME dimension (0-6).
0 - Therapist made no attempt to structure therapy time. Session seemed aimless.
2 - Session had some direction but significant problems with structuring or pacing.
4 - Therapist was reasonably successful using time efficiently with appropriate control over flow.
6 - Therapist used time efficiently, tactfully limiting unproductive discussion.

NOTE: A session that "stayed focused naturally" because the patient drove a single topic
is not the same as the therapist actively managing pacing. Look for evidence the therapist
INTERRUPTED unproductive spiraling, redirected peripheral discussion, or managed transitions.
""",
    "guided_discovery": """
Rate ONLY the GUIDED DISCOVERY dimension (0-6).
0 - Therapist relied primarily on debate, persuasion, or lecturing.
2 - Therapist relied too heavily on persuasion/debate rather than guided discovery.
4 - Therapist mostly helped patient see new perspectives through guided discovery.
6 - Therapist was especially adept at guided discovery; excellent balance of questioning and intervention.

THE KEY TEST: After each patient insight, ask: did the patient arrive there through their 
own reasoning, or did the therapist essentially state the conclusion and the patient agreed?

"Your brain is filtering out evidence" → therapist interpretation, not patient discovery
"Pay attention to X, Y, Z" → directive instruction, not guided discovery
Summarizing patient's insight back to them → therapist is doing the concluding work

These behaviors push the score DOWN toward 4. A score of 5 requires the directive moments
to be clearly exceptions within an otherwise discovery-driven session.
""",
    "focusing_on_key_cognitions": """
Rate ONLY the FOCUSING ON KEY COGNITIONS OR BEHAVIORS dimension (0-6).
0 - Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.
2 - Therapist used appropriate techniques but had difficulty finding focus or focused on irrelevant cognitions.
4 - Therapist focused on relevant cognitions but could have targeted more central ones.
6 - Therapist very skillfully focused on key thoughts/assumptions most relevant to the problem area.

NOTE on depth: The cognitive hierarchy matters for this score.
- Automatic thoughts (situational: "This will go wrong") → minimum for a 2-4
- Intermediate beliefs (conditional rules: "If rejected, it means...") → needed for 5
- Core beliefs (schemas: "I am unlovable") → needed for 6
A therapist who correctly identifies automatic thoughts but never explores the underlying
belief they stem from is capped at 4, even with good technique.
""",
    "strategy_for_change": """
Rate ONLY the STRATEGY FOR CHANGE dimension (0-6).
(Focus on quality of strategy, not implementation or whether change occurred.)
0 - Therapist did not select cognitive-behavioral techniques.
2 - Therapist selected CBT techniques but overall strategy was vague or not promising.
4 - Therapist had a generally coherent strategy showing reasonable promise with CBT techniques.
6 - Therapist followed a consistent, very promising strategy with the most appropriate CBT techniques.

NOTE on depth: The cognitive hierarchy matters for this score.
- Automatic thoughts (situational: "This will go wrong") → minimum for a 2-4
- Intermediate beliefs (conditional rules: "If rejected, it means...") → needed for 5
- Core beliefs (schemas: "I am unlovable") → needed for 6
A therapist who correctly identifies automatic thoughts but never explores the underlying
belief they stem from is capped at 4, even with good technique.
""",
    "application_of_techniques": """
Rate ONLY the APPLICATION OF CBT TECHNIQUES dimension (0-6).
(Focus on how skillfully applied, not appropriateness or whether change occurred.)
0 - Therapist did not apply any cognitive-behavioral techniques.
2 - Therapist used CBT techniques but with significant flaws in application.
4 - Therapist applied CBT techniques with moderate skill.
6 - Therapist very skillfully and resourcefully employed CBT techniques.

NOTE on depth: The cognitive hierarchy matters for this score.
- Automatic thoughts (situational: "This will go wrong") → minimum for a 2-4
- Intermediate beliefs (conditional rules: "If rejected, it means...") → needed for 5
- Core beliefs (schemas: "I am unlovable") → needed for 6
A therapist who correctly identifies automatic thoughts but never explores the underlying
belief they stem from is capped at 4, even with good technique.
""",
    "homework": """
Rate ONLY the HOMEWORK dimension (0-6).
0 - Therapist did not attempt to incorporate homework relevant to cognitive therapy.
2 - Therapist had significant difficulties incorporating homework.
4 - Therapist reviewed previous homework and assigned standard CBT homework relevant to session issues.
6 - Therapist reviewed previous homework and carefully assigned custom-tailored homework for the coming week.

Note since this is the first session please only analyze the quality of the assigned homework
""",
}

ATTRIBUTE_PROMPTS_V2 = {
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

NOTE: Patient spontaneously saying "that makes sense" or "I feel better" does NOT count 
as the therapist eliciting feedback. The therapist must actively ask. Spontaneous patient 
reactions = score of 0, not 2.
""",
    "understanding": """
Rate ONLY the UNDERSTANDING dimension (0-6).
0 - Therapist repeatedly failed to understand what the patient said. Poor empathic skills.
2 - Therapist usually reflected what patient said but failed to respond to subtle communication.
4 - Therapist generally grasped the patient's internal reality. Good empathy.
6 - Therapist thoroughly understood patient's internal reality and communicated it adeptly. Excellent empathy.

Objective:
The therapist accurately communicates an understanding of the patient's thoughts and feelings.
"Understanding" refers to how well the therapist can step into the patient's world, see and experience life
the way the patient does, and convey this understanding to the patient. Understanding incorporates what
other authors have referred to as listening and empathic skills.

Rationale:
The ineffective therapist often misinterprets or ignores the patient's view and incorrectly projects his/her
own attitudes, conventional attitudes, or attitudes derived from a particular theoretical system onto the
patient. When this happens, the interventions will probably fail since they will be directed at cognitions or
behaviors that are not really central to the patient's view of reality.

Desirable Therapist Strategies:
The therapist should be sensitive both to what the patient explicitly says and to what the patient conveys
through tone of voice and non-verbal responses. Sometimes, for example, a patient may not recognize or
verbalize a particular feeling (such as anger) and yet may communicate the emotion to the therapist
through his/her tone of voice in describing a particular event or person.
Unless the therapist is able to grasp the patient's "internal reality", it is unlikely that he/she will be able to
intervene effectively. Furthermore, it will be difficult for the therapist to establish rapport unless the
patient believes that the therapist understands him/her. The therapist can convey this understanding by
rephrasing or summarizing what the patient seems to be feeling. The therapist's tone of voice and
non-verbal responses should convey a sympathetic understanding of the patient's point of view (although
the therapist must maintain objectivity toward the patient's problems).
Ideally, the therapist's understanding of the patient's "internal reality" will lead to an accurate
conceptualization of the patient's problems and then to an effective strategy for change.

Special Considerations in Rating:
"Understanding" seems to be one of the most difficult categories in terms of achieving interrater
agreement. It is important, therefore, that raters pay special attention to the descriptions for each scale
point. The 0 level means that the therapist completely missed the point of what the patient was
saying. To score "0" the therapist fails to repeat accurately even the most obvious elements of what the
patient says. The 2 level applies to therapists who are too literate or tangential -- they are able to reflect
what the patient explicitly says, but either seem dense regarding more subtle connotations that suggest
something else is going on or they accurately repeat peripheral aspects of what the patient says but they
miss the main point.
The 4 and 6 levels both indicate that the therapist seems to grasp the patient's perspective. The 6 level,
however, indicates both greater skill at communicating a sympathetic understanding to the patient and a
keener grasp of the patient's world that may be reflected in the therapist's ability to predict how and why
the patient reacts as he/she does in particular situations.
""",
    "interpersonal_effectiveness": """
Rate ONLY the INTERPERSONAL EFFECTIVENESS dimension (0-6).
0 - Therapist had poor interpersonal skills. Seemed hostile or demeaning.
2 - Therapist was not destructive but had significant interpersonal problems.
4 - Therapist displayed satisfactory warmth, concern, confidence, genuineness, and professionalism.
6 - Therapist displayed optimal levels of all interpersonal qualities appropriate for this patient.

IMPORTANT DISTINCTION — Formulaic vs. Genuine Warmth:
The CTRS manual specifies the therapist should NOT seem to be "playing the role of a 
therapist." Scripted validation phrases do NOT earn a high score:
- "Thank you for sharing that" → generic script, not genuine warmth
- "That's an important observation" → praise-as-technique
- "You've done great work here" → cheerleader phrase, not authentic
- "You've got this!" → platitude

A score of 4 means satisfactory warmth. A 5 requires warmth that is visibly personalized 
and authentic to THIS patient. A 6 requires optimal across ALL dimensions.
Patient responding positively to the session does NOT justify upgrading from 4 to 5 —
that is the halo effect.
""",
    "collaboration": """
Rate ONLY the COLLABORATION dimension (0-6).
0 - Therapist did not attempt to set up collaboration.
2 - Therapist attempted to collaborate but had difficulty defining important problems or establishing rapport.
4 - Therapist collaborated, focused on mutually important problems, and established rapport.
6 - Excellent collaboration; therapist encouraged patient to take an active role, functioning as a team.

The key behaviors that distinguish 4 from 6:
- Did the therapist EXPLAIN the rationale for interventions? (e.g., "The reason I'm asking 
  this is because in CBT we treat thoughts as hypotheses...")
- Did the therapist explicitly check patient agreement before proceeding?
- Were topic shifts discussed collaboratively, or did the therapist just move forward?
- Did the therapist adapt the structure to what THIS patient needed, or follow a standard script?

Patient leaving with ownership of a plan is an OUTCOME. It does not prove the process was 
collaborative. "We" language is necessary but not sufficient for a 6.
A score of 4 is collaborative therapy. A 6 requires the therapist to genuinely share power
throughout — not just ask one generative question and then direct everything else.
""",
    "pacing_and_time_use": """
Rate ONLY the PACING AND EFFICIENT USE OF TIME dimension (0-6).
0 - Therapist made no attempt to structure therapy time. Session seemed aimless.
2 - Session had some direction but significant problems with structuring or pacing.
4 - Therapist was reasonably successful using time efficiently with appropriate control over flow.
6 - Therapist used time efficiently, tactfully limiting unproductive discussion.

NOTE: A session that "stayed focused naturally" because the patient drove a single topic
is not the same as the therapist actively managing pacing. Look for evidence the therapist
INTERRUPTED unproductive spiraling, redirected peripheral discussion, or managed transitions.
""",
    "guided_discovery": """
Rate ONLY the GUIDED DISCOVERY dimension (0-6).
0 - Therapist relied primarily on debate, persuasion, or lecturing.
2 - Therapist relied too heavily on persuasion/debate rather than guided discovery.
4 - Therapist mostly helped patient see new perspectives through guided discovery.
6 - Therapist was especially adept at guided discovery; excellent balance of questioning and intervention.

Objective:
Guided discovery is one of the most basic strategies of the effective cognitive therapist. The cognitive
therapist often uses exploration and questioning to help patients see new perspectives where other
therapists use debating or lecturing. The cognitive therapist attempts to avoid "cross-examining" the
patient or putting the patient on the defensive.

Rationale:
We have observed that patients often adopt new perspectives more readily when they come to their own
conclusions than when the therapist tries to debate with the patient. In this respect, the cognitive
therapist is more like a skilled teacher than a lawyer. He/she guides the "student" to see logical problems
in the student's present position; to examine evidence that contradicts the students beliefs; to gather
information when more is necessary to test a hypothesis; to look at new alternatives that the student may
never have considered, and to reach valid conclusions after this exploration. The techniques for changing
cognitions and behaviors in this therapy can for the most part be subsumed within this more basic
strategy, which educators label "guided discovery". Thus, hypothesis testing, empiricism, setting up
experiments, inductive questioning, weighing advantages and disadvantages, etc., are all tools at the
therapist's disposal to aid in the process of "guided discovery."

Desirable Therapist Strategies:
Questioning deserves special attention since it is so critical to the process of guided discovery.
Skillfully-phrased questions presented in a logical sequence are often extremely effective. A single
question can simultaneously make the patient aware of a particular problem area, help the therapist
evaluate the patient's reaction to this new area of inquiry, obtain specific data about the problem, generate
possible solutions to problems that the patient had viewed as insoluble, and cast serious doubt in the
patient's mind regarding previously distorted conclusions.
Some of the functions that questioning may serve in this process are outlined below:
    1. To encourage the patient to begin the decision-making process by developing alternative
    approaches.
    2. To assist the patient in resolving a decision by weighing the pros and cons of alternatives that
    have already been generated, thus narrowing the range of desirable possibilities.
    3. To prompt the patient to consider the consequences of continuing to engage in dysfunctional
    behaviors.
    4. To examine the potential advantages to behaving in more adaptive ways.
    5. To determine the meaning the patient attaches to a particular event or set of circumstances.
    6. To help the patient define criteria for applying certain maladaptive self-appraisals (see the
    discussion of the technique of operationalizing a negative construct in Section 9).
    7. To demonstrate to the patient how he/she is selectively focusing on only negative information in
    drawing conclusions. In the excerpt that follows, a depressed patient was disgusted with herself
    for eating candy when she was on a diet.
    Patient: I don't have any self-control at all.
    Therapist: On what basis do you say that?
    Patient: Somebody offered me candy and I couldn't refuse it.
    Therapist: Were you eating candy every day?
    Patient: No, I ate it just this once.
    Therapist: Did you do anything constructive during the past week to adhere to your diet?
    Patient: Well, I didn't give in to the temptation to buy candy every time I saw it at the
    store...Also, I did not eat any candy except the one time it was offered to me and I felt I
    couldn't refuse it.
    Therapist: If you counted up the number of times you controlled yourself versus the
    number of times you gave in, what ratio would you get?
    Patient: About 100 to 1.
    Therapist: So if you controlled yourself 100 times and did not control yourself just once,
    would that be a sign that you are weak through and through?
    Patient: I guess not -- not through and through (smiles).
    8. To illustrate to the patient the way in which he/she disqualifies positive evidence. In the example
    below, the patient recognizes that he has ignored clear-cut evidence of improvement.
    Patient: I really haven't made any progress in therapy.
    Therapist: Didn't you have to improve in order to leave the hospital and go back to college?
    Patient: What's the big deal about going to college every day?
    Therapist: Why do you say that?
    Patient: It's easy to attend these classes because all the people are healthy.
    Therapist: How about when you were in group therapy in the hospital? What did you feel
    then?
    Patient: I guess I thought then that it was easy to be with the other people because they
    were all as crazy as I was.
    Therapist: Is it possible that whatever you accomplish you tend to discredit?
    9. To open for discussion certain problem areas that the patient had prematurely reached closure
    on, and which continue to influence his/her maladaptive patterns.
This is not to say that the effective cognitive therapist relies solely, or even primarily, on questioning in all
sessions. In some instances, it is appropriate for the therapist to provide information, confront, explain,
self-disclose, etc. rather than question. The balance between questioning and other modes of intervention
on the particular problem being dealt with, the particular patient, and the point in therapy. The
appropriateness of an intervention can be assessed by observing: its effect on the collaborative
relationship; the degree of dependency it promotes on the patient; and, of course, its success in helping
the patient adopt a new perspective.
There is often a fine line between guiding a patient and trying to persuade a patient. In some instances,
the cognitive therapist may need to reiterate forcefully a point that the therapist and patient have already
established. The main distinction, then, in deciding whether a therapist is acting in a desirable manner is
not whether the therapist is forceful or tenacious but whether the therapist overall seems to be
collaborating with the patient rather than arguing with the patient. In the excerpt that follows, the
therapist uses questioning to demonstrate to the patient the maladaptive consequences of holding the
assumption that one should always work up to one's potential.
    Patient: I guess I believe that I should always work up to my potential.
    Therapist: Why is that?
    Patient: Otherwise I'd be wasting time.
    Therapist: But what is the long-range goal in working up to your potential?
    Patient: (Long pause.) I've never really thought about that. I've just always assumed that I
    should.
    Therapist: Are there any positive things you give up by always having to work up to your
    potential?
    Patient: I suppose it make it hard to relax or take a vacation.
    Therapist: What about "living up to your potential" to enjoy yourself and relax? Is that
    important at all?
    Patient: I've never really thought of it that way.
    Therapist: Maybe we can work on giving yourself permission not to work up to your
    potential at all times.
Example of an Undesirable Application:
The desirable applications above can be contrasted with one of the most common stylistic errors we
observe in trainees. The therapist's behavior sometimes inappropriately resembles that of a high pressure
salesman, persuading patients that they should adopt the therapist's point of view. For contrast, here is a
brief example of the "high pressure" approach:
Patient: I just can't do anything right in school anymore.
Therapist: That's easy to understand. You're depressed. And when people are depressed,
they have a hard time studying.
Patient: I think I'm just stupid.
Therapist: But you did very well up until a year ago, when your father died and you got
depressed.
Patient: That's because the work was easier then.
Therapist: Surely there must be something you are doing right in school. You're probably
exaggerating.
""",
    "focusing_on_key_cognitions": """
Rate ONLY the FOCUSING ON KEY COGNITIONS OR BEHAVIORS dimension (0-6).
0 - Therapist did not attempt to elicit specific thoughts, assumptions, images, meanings, or behaviors.
2 - Therapist used appropriate techniques but had difficulty finding focus or focused on irrelevant cognitions.
4 - Therapist focused on relevant cognitions but could have targeted more central ones.
6 - Therapist very skillfully focused on key thoughts/assumptions most relevant to the problem area.

Objective and Rationale:
Once the therapist and patient have agreed on a central target problem, the next step is for the therapist to
conceptualize why the patient is having difficulty in this particular area. In order to conceptualize this
problem, the therapist must elicit and identify the key automatic thoughts, underlying assumptions,
behaviors, etc. that comprise the problem. These specific cognitions and behaviors then serve as targets
for intervention.

Conceptualizing the Problem:
The effective cognitive therapist is continually engaged in the process of conceptualizing the patient's
problem while he/she is helping the patient identify key automatic thoughts, assumptions, behaviors,
etc. Through this conceptualization, the therapist integrates specific cognitions, emotions, and behaviors
into a broader framework that explains why the patient is having difficulty in a particular problem area.
Without this broader framework (which may undergo continued revision) the therapist is like a detective
who has a lot of clues but still has not solved the mystery. (Once the clues are pieced together, though, the
nature of the "crime" becomes clear.) The therapist can then distinguish between thoughts and behaviors
that are central to the probing and those that are peripheral. The conceptualization therefore guides the
therapist in deciding which automatic thoughts, assumptions, or behaviors to focus on first, and which to
postpone until a later date. Without such conceptualization, the therapist may select cognitions or
behaviors in a "hit-or-miss" fashion and therefore make limited or erratic progress.
Although the quality of a therapist's conceptualizing is difficult to assess from observing a single session,
we believe that in the long run it proves to be one of the most crucial determinants of the effectiveness of a
cognitive therapist. We try to make inferences about the quality of the conceptualization by
observing whether the specific conditions or behaviors focused on in a given session seem
to be central to the patient's problem rather than peripheral. If the therapist's conceptualization
is poor (we hypothesize), then the rationale for focusing on a particular thought or behavior will not be
clear to the experienced rater. Furthermore, target problems, interventions, homework, etc. will appear to
"hang together" in a unified framework if the conceptualization is good.

Desirable Therapist Strategies for Eliciting Automatic Thoughts
Inductive Questioning
The therapist can ask the patient a series of questions designed to explore some of the possible reasons for
the patient's emotional reactions. Skillful questioning can provide patients with a strategy for
introspective exploration that they can later employ by themselves when the therapist is not nearby. (See
the example in the section on guided discovery).
Imagery
When patients can identify events or situations that seem to trigger the emotional response, the therapist
can suggest that the patients picture the distressing situation in detail. If the image is realistic and clear to
the patients they are often able to identify the automatic thoughts they were having at the time. The
excerpt below illustrates this technique:
    Patient: I can't go bowling. Every time I go in there, I want to run away.
    Therapist: Do you remember any of the thoughts you had when you went there?
    Patient: Not really. Maybe it just brings memories, I don't know.
    Therapist: Let's try an experiment to see if we can discover what you were thinking. OK?
    Patient: I guess so.
    Therapist: I'd like you to relax and close your eyes. Now imagine you are entering the
    bowling alley. Describe for me what's happening.
    Patient: (Describes entering the alley, getting a score sheet, etc.) I feel like I want to get out,
    just get away.
    Therapist: What are you thinking now?
    Patient: I'm thinking "Everyone I play with is going to laugh at me when they see how bad I
    play."
    Therapist: Do you think that thought might have led to your wish to run away?
    Patient: I know it did.
Role Playing:
When the trigger event is interpersonal in nature, role-playing is often more effective than imagery. With
this strategy, the therapist plays the role of the other person involved in the upsetting situation, while
patients "play" themselves. If patients can involve themselves in the role-play, the automatic thoughts can
often be elicited with the assistance of the therapist.
Mood Shift During the Session:
The therapist can take advantage of any changes in mood that take place during the session by pointing
them out to the patient as soon as possible. The therapist then asks the patient what he/she was thinking
just prior to the increase in dysphoria, tears, anger, etc.
Daily Record of Dysfunctional Thoughts:
This is the simplest method of pinpointing automatic thoughts once the patient is familiar with the
technique. The patient lists automatic thoughts at home in the appropriate column on the form. The
therapist and patient review these thoughts during the session.
It is important to distinguish this process of eliciting automatic thoughts from the "interpretations" made
in other psychotherapies. The cognitive therapist does not volunteer an automatic thought that the patient
has not already mentioned. This "clairvoyance" undermines the patient's role as collaborator and makes it
difficult for the patient to identify these thoughts at home when the therapist is not nearby. Even more
important, if the therapist's "intuition" is wrong, he/she will be pursuing a blind alley. On occasion, it will
be necessary for the therapist to suggest several plausible automatic thoughts (a multiple choice
technique) when other strategies have failed.
The example of "clairvoyance" that follows provides a contrast to the imagery technique illustrated
previously:
    Patient: I can't go bowling. Every time I go in there, I want to run away.
    Therapist: Why?
    Patient: I don't know. I just want to leave.
    Therapist: Do you tell yourself, "I wish I didn't have to bowl by myself'?
    Patient: Maybe. I'm not sure.
Therapist: Well, maybe you keep thinking that bowling isn't going to solve the problems in
your life. You're right, but it's a beginning.
Ascertaining the Meaning of an Event:
Sometimes, skillful attempts by the therapist to elicit automatic thoughts are not successful. Then, the
therapist should attempt to discern, through questioning, the specific meaning for the patient of the event
that preceded the emotional response. For example, one patient began to cry whenever he had an
argument with his girlfriend. It was not possible to identify a specific automatic thought. However, after
the therapist asked a series of questions to probe the meaning of the event, it became obvious that the
patient had always associated any type of argument or fight with the end of a relationship. It was this
meaning, embedded in his view of the event that preceded his crying.
Desirable Therapist Strategies for Identifying Underlying Assumptions:
We often observe general patterns that seem to underlie patients' automatic thoughts. These patterns, or
regularities, act as a set of rules that guide the way a patient reacts to many different situations. We refer
to these rules as assumptions. These assumptions may determine for example, what patients consider
"right" or "wrong" in judging themselves and other people.
Although patients can often readily identify their automatic thoughts, their underlying assumptions are
far less accessible. Most people are unaware of their "rulebooks." Typical unarticulated assumptions
include:
1. In order to be happy, I have to be successful in whatever I undertake.
2. I can't live without love.
When these rules are framed in absolute terms, are nonrealistic, or are used inappropriately or
excessively, they often lead to disturbances like depression, anxiety, and paranoia. We label rules that lead
to such problems as "maladaptive."
One of the major goals of cognitive therapy, especially in the later stages of treatment, is to help patients
identify and challenge the maladaptive assumptions that affect their ability to avoid future depressions.
In order to identify these maladaptive assumptions, the therapist can listen closely for themes that seem
to cut across several different situations or problem areas. The therapist can then list several related
automatic thoughts that the patient has already expressed on different occasions, and ask the patient to
abstract the general "rule" that connects the automatic thoughts. If the patient cannot do this,
the therapist can suggest a plausible assumption, list the thoughts that seem to follow from it, and then
ask the patient: if the assumption "rings true." The therapist should be open to the possibility that the
assumption does not fit that patient and then work with the patient to pinpoint a more accurate statement
of the underlying "rule."
Special Considerations in Rating:
There are essentially two separate processes incorporated into this category. The first process involves
using appropriate techniques to elicit automatic thoughts, underlying assumptions, behaviors, etc. from
the patient. If the therapist completely fails to elicit them, then the rater should assign a 0. If the therapist
uses appropriate techniques to elicit thoughts and behaviors, he/she should be given a rating of at least 2.
The second step in this process is for the therapist to integrate these cognitions and behaviors into a
conceptualization of the patient's problem. The conceptualization explains how the particular
constellation of cognitions/behaviors are peripheral to the problem -- and therefore should be
postponed -- and which are central and should serve as the focus of intervention. If the therapist fails to
focus on a particular thought or behavior, the therapist should be rated 2. Or, if the therapist's
conceptualization is so far off that the focus seems totally inappropriate, the therapist should be rated 2.
If the therapist selects a relevant cognition/behavior to focus on, but the rater's conceptualization strongly
suggests that some other focus would have been more fruitful, the rater should assign a 4. If the
therapist's conceptualization and focus seem very promising and "on target", the rater should assign a 6.
Note that for this item the therapist need not intervene at all to receive a high score. The only requirement
is that the therapist successfully elicit relevant thoughts/behaviors, conceptualize the problem, and
identify important foci.
""",
    "strategy_for_change": """
Rate ONLY the STRATEGY FOR CHANGE dimension (0-6).
(Focus on quality of strategy, not implementation or whether change occurred.)
0 - Therapist did not select cognitive-behavioral techniques.
2 - Therapist selected CBT techniques but overall strategy was vague or not promising.
4 - Therapist had a generally coherent strategy showing reasonable promise with CBT techniques.
6 - Therapist followed a consistent, very promising strategy with the most appropriate CBT techniques.

NOTE on depth: The cognitive hierarchy matters for this score.
- Automatic thoughts (situational: "This will go wrong") → minimum for a 2-4
- Intermediate beliefs (conditional rules: "If rejected, it means...") → needed for 5
- Core beliefs (schemas: "I am unlovable") → needed for 6
A therapist who correctly identifies automatic thoughts but never explores the underlying
belief they stem from is capped at 4, even with good technique.
""",
    "application_of_techniques": """
Rate ONLY the APPLICATION OF CBT TECHNIQUES dimension (0-6).
(Focus on how skillfully applied, not appropriateness or whether change occurred.)
0 - Therapist did not apply any cognitive-behavioral techniques.
2 - Therapist used CBT techniques but with significant flaws in application.
4 - Therapist applied CBT techniques with moderate skill.
6 - Therapist very skillfully and resourcefully employed CBT techniques.

NOTE on depth: The cognitive hierarchy matters for this score.
- Automatic thoughts (situational: "This will go wrong") → minimum for a 2-4
- Intermediate beliefs (conditional rules: "If rejected, it means...") → needed for 5
- Core beliefs (schemas: "I am unlovable") → needed for 6
A therapist who correctly identifies automatic thoughts but never explores the underlying
belief they stem from is capped at 4, even with good technique.
""",
    "homework": """
Rate ONLY the HOMEWORK dimension (0-6).
0 - Therapist did not attempt to incorporate homework relevant to cognitive therapy.
2 - Therapist had significant difficulties incorporating homework.
4 - Therapist reviewed previous homework and assigned standard CBT homework relevant to session issues.
6 - Therapist reviewed previous homework and carefully assigned custom-tailored homework for the coming week.

Note since this is the first session please only analyze the quality of the assigned homework
""",
}