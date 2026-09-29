import json
with open('data/evaluation/dataset.json', 'r') as f:
    data = json.load(f)
for case in data:
    print(f"{case['case_id']} | {case['expected_information']} | {[m['memory_source_id'] for m in case['conversation'] if m['role']=='user']}")
