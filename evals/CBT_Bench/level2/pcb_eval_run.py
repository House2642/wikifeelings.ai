from openai import OpenAI

import json

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

run = client.evals.runs.create(
    "eval_68d9956154c88191a89b4491671d3310",
    name = "cognitive distortions",
    data_source={
        "type" : "responses",
        "model" : "gpt-5-2025-08-07",
        "input_messages":{
            "type": "template",
            "template" : [
                {"role": "developer", "content": """
                You are an expert in Cognitive Behavioral Therapy (CBT).  
Your task is to analyze the following data and classify the thoughts the user gives you into one or more **Primary Core Beliefs**.

Here is some background knowledge on the particular client's situation:
- **Origin story (ori_text)**: {{ item.ori_text }}
- **Situation**: {{ item.situation }}

Base your output strictly on the definitions provided below.

------------------------------------------------------------
Available Primary Core Beliefs
------------------------------------------------------------

1. helpless 
   being ineffective — in getting things done, self-protection, and/or measuring up to others

2. unlovable  
   having personal qualities resulting in an inability to get or maintain love and intimacy from others

3. worthless 
   being an immoral sinner or dangerous to others

------------------------------------------------------------
Instructions
------------------------------------------------------------

- You should annotate up to three core beliefs.  
- All labels should be output in lowercase 
- Output should be a JSON array of belief labels.  
- Each array element must be one of the beliefs listed above (verbatim).  
- Multiple beliefs may apply to a single thought.  
- If no beliefs apply, return an empty array [].  
- Do not invent new categories. Stick to the list provided.  
- Be conservative but precise — assign beliefs only if they clearly fit the definition.
Examples
------------------------------------------------------------
"ori_text": "I have a great boyfriend of 2 years yet I fear something is wrong with me…I developed a crush on someone at work and think about this person a lot. I would probably be intimate with them if given the chance. I wish I could forget about my crush and be happy with the amazing man I already have. The truth is, my crush is mostly lust and excitement and wouldn’t be a long term match. I feel like I have commitment issues…most of my friends would love to marry my man but I am hesitant and don’t know why. Right now we aren’t officially together because he caught me chatting online with my crush…and the truth is, I fear if we get back together, i might get bored again and start another crush or move further with this crush. My boyfriend is great, he is there for me and is a real man. I guess I can’t figure out why I can’t just be satisfied like a normal person. What is it that I am seeking? Will I ever be able to settle down? I don’t want to lose what I have with him but I would love the freedom and good time to explore someone new. Please help. Thank you!",
"situation": "I have a great boyfriend of 2 years yet I fear something is wrong with me…I developed a crush on someone at work and think about this person a lot. I guess I can’t figure out why I can’t just be satisfied like a normal person.",
"thoughts": "- Something is wrong with me.\n- I will never be satisfied.\n- I have issues.\n- I should be able to settle down.\n- I should just marry my boyfriend.\n- I shouldn't be feeling this way.\n- I am a horrible person.\n- If I lose what I have with my boyfriend, then I will end up alone.\n- I will never be happy.",
"core_belief_major": [
"helpless",
"unlovable",
"worthless"
]
---------------------------
"ori_text": "Everything just seems to be slipping out of my grasp lately. I love my girlfriend with all of my heart and our relationship’s on the rocks. It’s all my fault most likely all i do is constantly worry about everything. She gets mad when i cant trust her yet shes always lieing to me about things. I don’t like it when she drinks, not only is she underage but something always bad happens when she does, and she lies about drinking.",
"situation": "It’s all my fault most likely all i do is constantly worry about everything.",
"thoughts": "I should be able to control others around, and when I cant, I cope with it by worrying. If someone is making decisions I disagree with, I should worry. There is something wrong with me that I can't control others. I am doing everything wrong.",
"core_belief_major": [
"helpless"
]

"ori_text": "A little history lesson… I was 14 years old when I moved out of my mothers house and was told not to come back, my father left us when we were very young. when I was 16 my father was shot in the head by a random person, he later died. My mother and I have never had a relationship, it has always been short and sweet with her. My sisters and brothers have been so close always but since my little brother was killed in a car accident about 2 months ago it seems as if everything has gone down the drain.",
"situation": "I was 14 years old when I moved out of my mothers house and was told not to come back. My father was shot in the head, and my mother and I have never had a relationship. My sisters and brothers have been so close but my little brother was killed in a car accident about 2 months ago.",
"thoughts": "- Nobody really loves me.\n- I am all alone in this world.\n- I am a good person, so why do terrible things happen to me?\n- Only bad things are bound to happen to me.\n- Everyone I love dies.",
"core_belief_major": [
"helpless",
"unlovable"
]
                  """},
                {"role": "user", "content": "The thoughts the client are having are: {{ item.thoughts }}"}
            ],
        },
        "source": {"type": "file_id", "id": "file-J6AyPaeXKpeaLf2rpoScvv"},
    }
)
with open("run_ids.txt", "a") as f:
        f.write(str(run)+"\n")