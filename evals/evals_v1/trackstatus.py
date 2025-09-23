from openai import OpenAI

with open("../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

run = client.evals.runs.retrieve(eval_id = 'eval_68d2a15dc1b4819192cdcfae66945898', run_id = "evalrun_68d2bef85cb081919401a7b64fdef40b")
print(run)