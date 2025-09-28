from openai import OpenAI

import json

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

records = []
with open("data/qa_test.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)   # record is a dict
        records.append(record)