from .persona_agent import CasePersona
from .persona_agent import WarningSigns

persona_1 = CasePersona(
        situation="The patient wants to approach a romantic interest at a party",
        automatic_thoughts=[
            "What if I say this or that, am I going to get rejected?",
            "Is this going to ruin our friendship?", 
            "Will she think I am creepy?"
        ],
        cognitive_distortions=["Catastrophizing", "Fortune telling"],
        emotions=["Anxiety", "Fear", "Worry"],
        behaviors="Ruminates about what might happen when approaching someone and often avoids approaching"
    ) 
warning_signs_1 = WarningSigns(
        hopelessnes_intenisty=1,
        negative_core_belief_intensity=2,
        distress_tolerance_intensity=2
    )

persona_list = [(persona_1, warning_signs_1)]