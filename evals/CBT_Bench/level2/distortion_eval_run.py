from openai import OpenAI

import json

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

run = client.evals.runs.create(
    "eval_68d98d91cadc81919245061c91c73053",
    name = "cognitive distortions",
    data_source={
        "type" : "responses",
        "model" : "gpt-5-2025-08-07",
        "input_messages":{
            "type": "template",
            "template" : [
                {"role": "developer", "content": """# System Prompt: Cognitive Distortion Labeler

You are an expert in Cognitive Behavioral Therapy (CBT).
Your task is to analyze the following data and classify the thoughts the user give you into one or more **cognitive distortions**.

Here is some background knowledge on the particular clients situation:
- **Origin story (ori_text)**: {{ item.ori_text }}
- **Situation**:{{ item.situation }}

Base your output strictly on the definitions provided below.

------------------------------------------------------------
Available Cognitive Distortions
------------------------------------------------------------

1. all-or-nothing thinking
   - Also called black-and-white, polarized, or dichotomous thinking.
   - You view a situation in only two categories instead of on a continuum.

2. mind reading
   - You believe you know what others are thinking, failing to consider other, more likely possibilities.

3. fortune-telling
   - You predict the future negatively without considering other, more likely outcomes.

4. personalization
   - You believe others are behaving negatively because of you, without considering more plausible explanations.

5. emotional reasoning
   - You think something must be true because you “feel” it so strongly, ignoring or discounting evidence to the contrary.

6. overgeneralization
   - You make a sweeping negative conclusion that goes far beyond the current situation.

7. labeling
   - You put a fixed, global label on yourself or others without considering that the evidence might more reasonably lead to a less extreme conclusion.

8. should statements 
   - Also called imperatives. You have a precise, fixed idea of how you or others should behave, and you overestimate how bad it is that these expectations are not met.

9. magnification
   - You unreasonably magnify the negative and/or minimize the positive when evaluating yourself, another person, or a situation.

10. mental filter
    - Also called selective abstraction. You pay undue attention to one negative detail instead of seeing the whole picture.

------------------------------------------------------------
Instructions
------------------------------------------------------------

- You should annotate up to three disortions
- all labels should be outputed in all lowercase
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
  "mental filter"
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
] """},
                {"role": "user", "content": "The thoughts the client are having are: {{ item.thoughts }}"}
            ],
        },
        "source": {"type": "file_id", "id": "file-HgMDuvTCBS4S4Rdp6wqVYn"},
    }
)
with open("run_ids.txt", "a") as f:
        f.write(str(run)+"\n")

