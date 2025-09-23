from openai import OpenAI

import json

with open("../../../../api_keys/api.txt") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

filesnames = ["core_fine_test.json","core_major_test.json","distortions_test.json"]

for name in filesnames:
    with open(name, "r") as f:
        data = json.load(f)  # load full JSON array

    with open(name +"l", "w") as f:
        for obj in data:
            f.write("{\"item\": " +json.dumps(obj) + "}\n")

    file = client.files.create(
       file=open(name +"l", "rb"),
       purpose="evals"
    )
    with open("file_record.txt", "a") as f:
        f.write(str(file)+"\n")