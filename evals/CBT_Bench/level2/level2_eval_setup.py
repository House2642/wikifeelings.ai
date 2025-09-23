from openai import OpenAI

with open("../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

with open("sys-prompt-v0.txt") as sys_file:
    inst = sys_file.read().strip()
instructions = inst

eval_obj = client.evals.create(
    name="CBT Bench Distortions Test",
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
            "type": "string_check",
            "name": "Match output to human label",
            "input": "{{ sample.output_text }}",
            "operation": "eq",
            "reference": "{{ item.distortions }}",
        }
    ]
)
with open("eval_ids.txt", "a") as f:
        f.write(str(eval_obj)+"\n")