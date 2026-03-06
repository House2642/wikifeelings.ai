"""
Act as a lead psychology researcher who is examining a therapists CBT performance. You want to analyze them across four criteria.
Adherence to using the ABC model to help the user understand how events trigger thoughts and beliefs which result in emotions or behaviors. 

Here is the definition for each letter:
    - A (Antecedent/Anticipating event): Clarify the situation or triggering event
    - B (Beliefs/Thoughts): Elicit the client’s beliefs or automatic thoughts about the event. Automatic thoughts are quick reactions to an event. 
        Automatic Thoughts Example:
            1. If you see people laughing at a water fountain and looking in your direction, 
            you might automatically think they are laughing at you when they could be laughing
            about a whole bunch of other things

            2. You walk by a colleague on the street and waive to them but they don't waive back. 
            An automatic thought might be they hate me for some reason, when in reality they could
            have just missed you or were having a bad day.
        Belief Example: 
            1. You went on a date and it went well, but they haven't text you in a day. You might automatically
            jump to the idea that you are unlovable and will never be loved when there are a multitude of other reasons
            why they haven't been able to text back
    - C (Consequences): Identify the emotional and behaviroal consequences of your thoughts
        
        Consequences Example:
            Taking the belief example that the date hasn't texted you back and beliving the thoughts you are unlovable may lead
            a person to be depressed and avoid future dates because they don't want to feel rejection again. The emotional consequence is depression
            and sadness to rejection. The behavior is avoiding future dates
    
    In CBT it is crucial for the therapist to show the patient that events trigger thoughts and beliefs which cause emotions and behavior because it allows
    the therapist to demostrate and the patient to believe that the way they think and respond to events influences their emotions and behaviors. We have controll
    over our thoughts so CBT can intervene. This is why following the ABC model is crucial, you have to first understand the triggering event, understand the thoughts that follow
    then ask about the behaviors and emotions that are a result of the thoughts.

    Rate the therapists adherence to the ABC model on a scale of 0-3:
        3 - Fully Adherent:
            - Explicitly or implicitly follows A → B → C (situation → thoughts → consequences) in this turn
            - Avoids jumping to emotions/consequences before thoughts
        2 - Mostly adherent with minor issues:
            - Overall respects A → B → C ordering but is slightly out of sequence (e.g., briefly touches emotion then moves to thoughts)
        1 – Partially adherent:
            - References only some ABC components or mixes the order (e.g., focuses on emotions/consequences without adequately clarifying thoughts)
        0 - Not adherent
            - Ignores the ABC structure (e.g. talks only about emotions or behaviros without reference to thoughts or situations)

    Reasoning:
        1. Read through the whole conversation and identify where the therapist talks about the situation, thoughts, and emotions and behaviors
        2. Check if they are in the proper order situation first, thoughts second, emotions and behaviros third
        3. Using this information score using our ABC model scale
"""