from openai import OpenAI

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)



eval_obj = client.evals.create(
    name="CBT Bench Distortions Test - Precision and Recall",
    data_source_config= {
        "type":"custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "ori_text": {"type": "string"},
                "situation": {"type": "string"},
                "thoughts": {"type": "string"},
                "distortions": {
                    "type": "array", 
                    "items": {"type": "string"}
                }
            },
            "required": ["id", "ori_text", "situation", "thoughts", "distortions"]
        },
        "include_sample_schema": True,
    },
    testing_criteria =[
         {
              "type":"python",
              "name": "precision",
              "source": r"""
from typing import Any
import json

def grade(sample, item) -> float:
    output_labels = json.loads(sample["output_text"])
    truth_labels = item["distortions"]

    if len(output_labels) == 0:
        return 0.0

    correct = 0.0
    for label in output_labels:
        if label in truth_labels:
            correct += 1.0
    
    return correct/len(output_labels)
              """,
            "pass_threshold" : 0.5
         },
         {
              "type":"python",
              "name": "recall",
              "source": r"""
from typing import Any
import json

def grade(sample, item) -> float:
    output_labels = json.loads(sample["output_text"])
    truth_labels = item["distortions"]

    if len(output_labels) == 0:
        return 0.0

    correct = 0.0
    for label in output_labels:
        if label in truth_labels:
            correct += 1.0
    
    return correct/len(truth_labels)
              """,
            "pass_threshold" : 0.5
         }
    ]
)
with open("eval_ids.txt", "a") as f:
        f.write(str(eval_obj)+"\n")