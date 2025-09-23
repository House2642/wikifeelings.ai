from openai import OpenAI

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

run = client.evals.runs.retrieve(eval_id="eval_68d2fa38ed6c8191abadb896f9d861bf", run_id="evalrun_68d303ec37f08191afce38b2064f672c")
print(run)