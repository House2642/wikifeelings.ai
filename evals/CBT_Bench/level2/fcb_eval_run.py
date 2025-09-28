from openai import OpenAI

import json

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

run = client.evals.runs.create(
    "eval_68d99dbe17e48191ac51902a80e028a7",
    name = "cognitive distortions",
    data_source={
        "type" : "responses",
        "model" : "gpt-5-2025-08-07",
        "input_messages":{
            "type": "template",
            "template" : [
                {"role": "developer", "content": """
                You are an expert in Cognitive Behavioral Therapy (CBT).  
Your task is to analyze the following data and classify the thoughts the user gives you into one or more **Primary Core Beliefs** and then identify up to **three fine-grained beliefs for each detected primary core belief**.

Here is some background knowledge on the particular client's situation:
- **Origin story (ori_text)**: {{ item.ori_text }}
- **Situation**: {{ item.situation }}

Base your output strictly on the definitions provided below.

------------------------------------------------------------
Available Primary Core Beliefs and Fine-Grained Beliefs
------------------------------------------------------------

1. helpless  
   being ineffective — in getting things done, self-protection, and/or measuring up to others  
   Fine-grained beliefs:  
   - I am incompetent  
   - I am helpless  
   - I am powerless, weak, vulnerable  
   - I am a victim  
   - I am needy  
   - I am trapped  
   - I am out of control  
   - I am a failure, loser  
   - I am defective  

2. unlovable  
   having personal qualities resulting in an inability to get or maintain love and intimacy from others  
   Fine-grained beliefs:  
   - I am unlovable  
   - I am unattractive  
   - I am undesirable, unwanted  
   - I am bound to be rejected  
   - I am bound to be abandoned  
   - I am bound to be alone  

3. worthless  
   being an immoral sinner or dangerous to others  
   Fine-grained beliefs:  
   - I am worthless, waste  
   - I am immoral  
   - I am bad – dangerous, toxic, evil  
   - I don’t deserve to live  

------------------------------------------------------------
Instructions
------------------------------------------------------------

- Detect up to three **primary core beliefs**.  
- For each detected primary core belief, select up to **three of its fine-grained beliefs** that clearly fit the thought content.  
- **Only output the fine-grained beliefs** (do not output the primary labels).  
- Output should be a **single JSON array** containing all selected fine-grained beliefs (strings).  
- All labels should be output in **lowercase**. Except the letter i should be expressed as uppercase. I not i 
- If no beliefs apply, return an empty array [].  
- Do not invent new categories or fine-grained beliefs. Stick to the list provided.  
- Be conservative but precise — assign beliefs only if they clearly fit the definition.

------------------------------------------------------------
Few-Shot Examples
------------------------------------------------------------

Example 1:  
Input:
{
  "ori_text": "I have a great boyfriend of 2 years yet I fear something is wrong with me…I developed a crush on someone at work and think about this person a lot...",
  "situation": "I can’t figure out why I can’t just be satisfied like a normal person.",
  "thoughts": "- Something is wrong with me. - I will never be satisfied. - I have issues. - I should be able to settle down. - I am a horrible person. - If I lose what I have with my boyfriend, then I will end up alone. - I will never be happy."
}

Output:
[
  "i am defective",
  "i am bound to be alone",
  "i am bad – dangerous, toxic, evil"
]

---

Example 2:  
Input:
{
  "ori_text": "Everything just seems to be slipping out of my grasp lately. I love my girlfriend with all of my heart and our relationship’s on the rocks...",
  "situation": "It’s all my fault most likely all i do is constantly worry about everything.",
  "thoughts": "I should be able to control others around, and when I cant, I cope with it by worrying. There is something wrong with me that I can't control others. I am doing everything wrong."
}

Output:
[
  "i am out of control",
  "i am defective"
]

---

Example 3:  
Input:
{
  "ori_text": "A little history lesson… I was 14 years old when I moved out of my mothers house and was told not to come back...",
  "situation": "My father was shot in the head, my little brother died recently, my mother and I have never had a relationship.",
  "thoughts": "- Nobody really loves me. - I am all alone in this world. - I am a good person, so why do terrible things happen to me? - Only bad things are bound to happen to me. - Everyone I love dies."
}

Output:
[
  "i am bound to be alone",
  "i am bound to be rejected",
  "i am a victim"
]
                  """},
                {"role": "user", "content": "The thoughts the client are having are: {{ item.thoughts }}"}
            ],
        },
        "source": {"type": "file_id", "id": "file-U7e7Ju6kBoUaWrvVCfpnRs"},
    }
)
with open("run_ids.txt", "a") as f:
        f.write(str(run)+"\n")