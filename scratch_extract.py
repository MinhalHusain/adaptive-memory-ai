import json
with open('data/evaluation/dataset.json', 'r') as f:
    data = json.load(f)
for case in data:
    for msg in case['conversation']:
        if msg['role'] == 'user':
            print(f"{msg['memory_source_id']} | {msg['content']}")
