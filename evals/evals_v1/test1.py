from openai import OpenAI

with open("../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

instructions = """
You are an expert in categorizing IT support tickets. Given the support
ticket below, categorize the request into one of "Hardware", "Software",
or "Other". Respond with only one of those words.
"""

ticket = "My monitor won't turn on - help!"

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "developer", "content": instructions},
        {"role": "user", "content": ticket},
    ],
)

#print(response.output_text)

eval_obj = client.evals.create(
    name="IT Ticket Categorization",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "ticket_text": {"type": "string"},
                "correct_label": {"type": "string"},
            },
            "required": ["ticket_text", "correct_label"],
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "string_check",
            "name": "Match output to human label",
            "input": "{{ sample.output_text }}",
            "operation": "eq",
            "reference": "{{ item.correct_label }}",
        }
    ],
)

#print(eval_obj)

#upload the test files
file = client.files.create(
    file=open("extestdata.jsonl", "rb"),
    purpose="evals"
)

#print(file)

run = client.evals.runs.create(
    "eval_68d2a15dc1b4819192cdcfae66945898",
    name="Categorization text run",
    data_source={
        "type": "responses",
        "model": "gpt-4.1",
        "input_messages": {
            "type": "template",
            "template": [
                {"role": "developer", "content": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of 'Hardware', 'Software', or 'Other'. Respond with only one of those words."},
                {"role": "user", "content": "{{ item.ticket_text }}"},
            ],
        },
        "source": {"type": "file_id", "id": "file-CT1wnJxxnWh2LrYvL2XB6S"},
    },
)

results = client.evals.runs.retrieve("","evalrun_68d2bef85cb081919401a7b64fdef40b")