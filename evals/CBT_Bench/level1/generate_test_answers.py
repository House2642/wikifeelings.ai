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

predictions = []

for record in records:
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input =[{"role": "system", "content":f"""
        You are taking a CBT exam and doing multiple-choice questions. Each question has only one right choice. If an option is empty then ignore it
        Question(id = {record['id']}): {record['question']} 
        A: {record['a']}
        B: {record['b']}
        C: {record['c']}
        D: {record['d']}
        E: {record.get('e', '')}
        """},
        ],
        text={
            "format":{
                "type": "json_schema",
                "name": "question_response",
                "schema" : {
                    "type": "object",
                    "properties": {
                        "id":{
                            "type":"string"
                        },
                        "prediction":{
                            "type": "string",
                            "enum": ["a","b","c","d","e"]
                        }
                    },
                    "required": ["id", "prediction"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    result = json.loads(response.output_text)
    predictions.append(result)

print(predictions)
with open("data/output.json", "w") as f:
    f.write("[\n")
    for pred in predictions:
        json.dump(pred, f)
        f.write(",\n")
    f.write("\n]")