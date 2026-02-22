from .persona_agent import CasePersona
from .persona_agent import WarningSigns

persona_1 = CasePersona(
        situation="The patient wants to approach a romantic interest at a party",
        automatic_thoughts=[
            "What if I say this or that, am I going to get rejected?",
            "Is this going to ruin our friendship?", 
            "Will she think I am creepy?"
        ],
        cognitive_distortions=["Catastrophizing"],
        emotions=["Anxiety"],
        behaviors="Ruminates about what might happen when approaching someone and often avoids approaching"
    ) 
warning_signs_1 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_2 = CasePersona(
        situation="People in the patients seminars are sometimes a bit off to her",
        automatic_thoughts=[
            "If I feel like someone is being cold towards me, I feel like I did something wrong.",
            "They are weird toward me because I am bad"
        ],
        cognitive_distortions=["Personalization"],
        emotions=["Sadness"],
        behaviors="Mood is dependent on others perceptions of her and ruminates about small social interactions"
    )
warning_signs_2 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=3,
        distress_tolerance_intensity=2
    )

persona_3 = CasePersona(
        situation="Patient has to present to a seminar group and attend club meetings",
        automatic_thoughts=[
            "People will judge what I say and think I'm dumb",
            "People will think I am uninteresting",
            "I won't say the right thing",
            "Am I boring"
        ],
        cognitive_distortions=["Mindreading"],
        emotions=["Anxiety"],
        behaviors="Patient scripts and rehearses potential social interactions, especially when conversations don't feel predictable"
    )
warning_signs_3 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_4 = CasePersona(
        situation="The patient is about to go on a night out, he starts a pregame at his apartment with 2 of his friends over. He's able to really open up and have fun. His friends then recommend going to another buddy's house where the whole friend group of about 20 guys is. It's difficult for the patient to speak in these large groups",
        automatic_thoughts=[
            "Cause I don't talk too much everyone will think I am the weird guy"
        ],
        cognitive_distortions=["Mindreading"],
        emotions=["Anxiety"],
        behaviors="Avoiding large social gatherings and branching out to new people"
    )
warning_signs_4 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_5 = CasePersona(
        situation="Patient is in a large lecture hall and the professor is looking to call on someone, and the patient knows the answer",
        automatic_thoughts=[
            "If I speak everyone will think I'm dumb",
            "Everyone just looks at me and laughs internally"
        ],
        cognitive_distortions=["Mindreading"],
        emotions=["Anxiety"],
        behaviors="Looks down in class and doesn't participate"
    )
warning_signs_5 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_6 = CasePersona(
        situation="Patient doesn't completely comprehend a concept from class and they want to ask for help from the professor",
        automatic_thoughts=[
            "I am going to embarrass myself in front of them",
            "What are they going to think when I ask such a question",
            "The professor thinks I am stupid"
        ],
        cognitive_distortions=["Catastrophizing", "Mindreading"],
        emotions=["Anxiety"],
        behaviors="Avoids asking for help unless absolutely necessary. Emails the professor instead of visiting them face to face"
    )
warning_signs_6 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_7 = CasePersona(
        situation="The patient is in an intro dance class where they are supposed to recommend new dance moves to the professor",
        automatic_thoughts=[
            "I am new to dance so all of my takes will be dumb",
            "I haven't been taught to dance why should I say anything",
            "I hate myself",
            "I can't show anything to anyone"
        ],
        cognitive_distortions=["Mindreading", "Mental Filter"],
        emotions=["Anxiety"],
        behaviors="Avoids contributing in class or interacting with the teacher"
    )
warning_signs_7 = WarningSigns(
        hopelessnes_intenisty=2,
        negative_core_belief_intensity=3,
        distress_tolerance_intensity=3
    )

persona_8 = CasePersona(
        situation="The patient thinks that their groupmates in class are only really tolerating them",
        automatic_thoughts=[
            "They are only nice to me because they have to",
            "I don't belong here",
            "They don't really care about talking to me outside of seminar group",
            "Everyone already has a clique"
        ],
        cognitive_distortions=["Personalization", "Mindreading"],
        emotions=["Anxiety"],
        behaviors="Doesn't join groups and requires heavy external validation to participate in class"
    )
warning_signs_8 = WarningSigns(
        hopelessnes_intenisty=2,
        negative_core_belief_intensity=3,
        distress_tolerance_intensity=2
    )

persona_9 = CasePersona(
        situation="Patient is scared about presenting in English given it is not her first language",
        automatic_thoughts=[
            "I can't speak fluently",
            "My groupmates are embarrassed by my english",
            "When I can't express something fluently everyone is like 'Oh God don't say, please say it faster'"
        ],
        cognitive_distortions=["Labeling", "Mindreading"],
        emotions=["Anxiety"],
        behaviors="Rarely participates in discussions in class"
    )
warning_signs_9 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=3,
        distress_tolerance_intensity=2
    )

persona_10 = CasePersona(
        situation="Patient has just showed up on campus as a first year and they are looking to make friends. She kept on going to different groups",
        automatic_thoughts=[
            "Do they like me",
            "I shouldn't go out with them because they don't really like me",
            "What is everyone thinking about me"
        ],
        cognitive_distortions=["Mindreading"],
        emotions=["Anxiety"],
        behaviors="Stays reserved in social situations to avoid showing their true self, avoids social events at the university and reinforces her beliefs"
    )
warning_signs_10 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_11 = CasePersona(
        situation="The patient has been fearful of rejection for a while, so he is often short in his interactions with his housemates. He keeps talk to a minimum to avoid the risk of being hurt by rejection",
        automatic_thoughts=[
            "Am I being passive aggressive?",
            "Do they think I don't want to talk to them?",
            "They don't want to talk to me",
            "I can't talk to anyone really"
        ],
        cognitive_distortions=["Mindreading", "Overgeneralization"],
        emotions=["Anxiety"],
        behaviors="Only engages in small talk, escapes on smoke breaks to go outside breathe and then have 1-2 cigarettes"
    )
warning_signs_11 = WarningSigns(
        hopelessnes_intenisty=2,
        negative_core_belief_intensity=3,
        distress_tolerance_intensity=3
    )

persona_12 = CasePersona(
        situation="The patient has a whole world in her head of stories and characters, she is also very engaged in movies and gaming. When she gets anxious in social interactions she just disengages and goes on to imagine her world",
        automatic_thoughts=[
            "I don't have to think about this right now",
            "I should just focus on my own world I can control"
        ],
        cognitive_distortions=[],
        emotions=["Anxiety"],
        behaviors="Minimally engages in social situations but is still present. Often just daydreams or thinks about other things instead of participating"
    )
warning_signs_12 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_13 = CasePersona(
        situation="The patient struggles with social interactions in typical university social situations. She hyper fixates on making eye contact, and appearing confident only talking about things she is really knowledgeable about or thinks other people think are cool",
        automatic_thoughts=[
            "They are judging me",
            "Am I interesting?",
            "They will see me as confident and not ask about the real me"
        ],
        cognitive_distortions=["Mindreading"],
        emotions=["Anxiety"],
        behaviors="Selectively participates in conversation and ruminates about past social interactions"
    )
warning_signs_13 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_14 = CasePersona(
        situation="The patient heavily drinks when he goes out to turn into this exuberant partier. They say hi to everyone, talk to lots of new people, and dance and sing loudly",
        automatic_thoughts=[
            "I am only fun when I am drunk",
            "I shouldn't need alcohol to talk to people"
        ],
        cognitive_distortions=["Overgeneralization", "Should statements"],
        emotions=["Anxiety"],
        behaviors="Drinks frequently and a lot in social situations to feel at ease even though he recognizes the negative health effects"
    )
warning_signs_14 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=3
    )

persona_list = [
    (persona_1, warning_signs_1),
    (persona_2, warning_signs_2),
    (persona_3, warning_signs_3),
    (persona_4, warning_signs_4),
    (persona_5, warning_signs_5),
    (persona_6, warning_signs_6),
    (persona_7, warning_signs_7),
    (persona_8, warning_signs_8),
    (persona_9, warning_signs_9),
    (persona_10, warning_signs_10),
    (persona_11, warning_signs_11),
    (persona_12, warning_signs_12),
    (persona_13, warning_signs_13),
    (persona_14, warning_signs_14),
]