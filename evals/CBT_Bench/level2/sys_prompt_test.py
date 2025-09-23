from openai import OpenAI

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

instructions = """
# System Prompt: Cognitive Distortion Labeler

You are an expert in Cognitive Behavioral Therapy (CBT).
Your task is to analyze the following data and classify the thoughts the user give you into one or more **cognitive distortions**.

Here is some background knowledge on the particular clients situation:
- **Origin story (ori_text)**: I’m 15, and for almost a year now I’ve had this issue with girls around the age of 6 to 8. I find them extremely sexually arousing and I can’t help myself but thinking about them. I’ve had plenty of girlfriends my age, but this just seems different. I’m worried about how later in life, if this will go away or not. Even my little neighbor, who cant be even 10 yet, I have fantasies about taking advantage of her innocence into having sex. I know this is horrible, and I even feel guilty constantly, but I can’t seem to help it. Also, when I see an older woman, about 25, who is very good looking, I don’t find them sexually attracting at all, unless they have a very childish look and feel to them. Even my close friend, who has a sister in 8th grade, I cannot get over how attractive she is, and how much I wish I could have her. I hope that this site could give some useful advise, as I love what you do to help so many people. Thank you ahead of time, hope you get to read this and help me out.,
- **Situation**:"I am finding younger girls sexually arousing.",

Base your output strictly on the definitions provided below.

------------------------------------------------------------
Available Cognitive Distortions
------------------------------------------------------------

1. All-or-nothing thinking (65)
   - Also called black-and-white, polarized, or dichotomous thinking.
   - You view a situation in only two categories instead of on a continuum.

2. Mind reading (47)
   - You believe you know what others are thinking, failing to consider other, more likely possibilities.

3. Fortune-telling (44)
   - You predict the future negatively without considering other, more likely outcomes.

4. Personalization (42)
   - You believe others are behaving negatively because of you, without considering more plausible explanations.

5. Emotional reasoning (36)
   - You think something must be true because you “feel” it so strongly, ignoring or discounting evidence to the contrary.

6. Overgeneralization (32)
   - You make a sweeping negative conclusion that goes far beyond the current situation.

7. Labeling (29)
   - You put a fixed, global label on yourself or others without considering that the evidence might more reasonably lead to a less extreme conclusion.

8. Should statements 
   - Also called imperatives. You have a precise, fixed idea of how you or others should behave, and you overestimate how bad it is that these expectations are not met.

9. Magnification
   - You unreasonably magnify the negative and/or minimize the positive when evaluating yourself, another person, or a situation.

10. Mental filter
    - Also called selective abstraction. You pay undue attention to one negative detail instead of seeing the whole picture.

------------------------------------------------------------
Instructions
------------------------------------------------------------

- Output should be a JSON array of distortion labels.
- Each array element must be one of the distortions listed above (verbatim).
- Multiple distortions may apply to a single thought.
- If no distortions apply, return an empty array [].
- Do not invent new categories. Stick to the list provided.
- Be conservative but precise — assign distortions only if they clearly fit the definition.

------------------------------------------------------------
Few-Shot Examples
------------------------------------------------------------

Example 1:
Input:
{
  "ori_text": "First off I would like to thank you for taking the time out to help me. But the problem is I’m depressed but nobody knows it. Half the reason I am is because I have no really close friends to hang out with...",
  "situation": "I’m depressed but nobody knows it. I do not have any friends. This started at age 11.",
  "thoughts": "I cannot make friends and have no one to hang out with. Therefore, I am always going to be alone and depressed."
}

Output:
[
  "mental filter",
  "personalization"
]

---

Example 2:
Input:
{
  "ori_text": "My girlfriend and I have been dating for 5 years. With being sexually active with each other for 3 years prior. Before we started dating and for the first year of our relationship her sex drive was amazing. Over time it slowly decreased...",
  "situation": "I have been with my girlfriend for 5 years and we have a history of a strong sexual intimacy connection. This has been getting worse over time.",
  "thoughts": "It is my girlfriend's sex drive that is the problem."
}

Output:
[
  "overgeneralization",
  "labeling"
]

---

Example 3:
Input:
{
  "ori_text": "I started going to therapy in December, after 3 other failed attempts. I’ve been consistent in going, and have really developed a rapport with my therapist. In my past, I have not been forthcoming with my feelings. I tend to hide, makeup stories of trauma to help cope with things that have happened...",
  "situation": "In my past, I have not been forthcoming with my feelings. I tend to hide, makeup stories of trauma to help cope with things that have happened. I was violently raped in college, and never told anyone about it.",
  "thoughts": "I was raped, so I don’t know if I can trust anyone again. If I do and something horrible happens, I don’t know if I could go on. I feel ugly, like everything is wrong with me. I’ve been hiding who I am because that trauma is the ugliest part of me. If I tell anyone what happened, they will judge me and reject me – and then I’ll be alone. So, I kept this ugly secret about myself and pretended everything was okay."
}

Output:
[
  "labeling",
  "magnification",
  "fortune-telling"
]
"""

ticket = "I am worried about whether or not this will go away later in life, this is horrible, and I even feel guilty constantly, I cannot help it."

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": ticket},
    ],
)
print(response.output_text)