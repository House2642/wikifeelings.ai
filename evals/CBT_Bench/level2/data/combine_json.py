import json

# Load all three JSON files
with open('core_fine_test.json', 'r') as f:
    core_fine_data = json.load(f)

with open('core_major_test.json', 'r') as f:
    core_major_data = json.load(f)

with open('distortions_test.json', 'r') as f:
    distortions_data = json.load(f)

# Create dictionaries with id as key for easy lookup
core_fine_dict = {item['id']: item for item in core_fine_data}
core_major_dict = {item['id']: item for item in core_major_data}
distortions_dict = {item['id']: item for item in distortions_data}

# Get all unique IDs from all three files
all_ids = set(core_fine_dict.keys()) | set(core_major_dict.keys()) | set(distortions_dict.keys())

# Combine data for each ID
combined_data = []
for id_val in sorted(all_ids, key=lambda x: int(x)):
    combined_item = {'id': id_val}

    # Add data from core_fine if exists
    if id_val in core_fine_dict:
        for key, value in core_fine_dict[id_val].items():
            if key != 'id':
                combined_item[key] = value

    # Add data from core_major if exists
    if id_val in core_major_dict:
        for key, value in core_major_dict[id_val].items():
            if key != 'id':
                combined_item[key] = value

    # Add data from distortions if exists
    if id_val in distortions_dict:
        for key, value in distortions_dict[id_val].items():
            if key != 'id':
                combined_item[key] = value

    combined_data.append(combined_item)

# Save combined data
with open('combined_test.json', 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"Combined {len(combined_data)} cases from all three files")
print(f"Core fine cases: {len(core_fine_data)}")
print(f"Core major cases: {len(core_major_data)}")
print(f"Distortions cases: {len(distortions_data)}")
print(f"Unique IDs: {len(all_ids)}")
print(f"\nCombined data saved to: combined_test.json")
