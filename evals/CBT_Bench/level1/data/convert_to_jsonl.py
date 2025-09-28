import json

# Load JSON
with open("qa_test.json", "r") as f:
    data = json.load(f)

# Save as JSONL
with open("qa_test.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")