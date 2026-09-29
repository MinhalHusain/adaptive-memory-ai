import json
from pathlib import Path

def annotate_dataset():
    path = Path("D:/Projects/Capstone/adaptive-memory-ai/data/evaluation/dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for case in dataset:
        expected_infos = case.get("expected_information", [])
        
        # Add memory_source_id to all user messages
        user_msg_idx = 0
        for msg in case["conversation"]:
            if msg["role"] == "user":
                msg["memory_source_id"] = f"{case['case_id']}_msg_{user_msg_idx}"
                user_msg_idx += 1

        # Determine relevant memory IDs based on expected information
        relevant_ids = set()
        for msg in case["conversation"]:
            if msg["role"] == "user":
                content_lower = msg["content"].lower()
                for exp in expected_infos:
                    # If expected info is found in this message, mark it relevant
                    # We check by token/exact match as a strong heuristic for the dataset
                    if exp.lower() in content_lower:
                        relevant_ids.add(msg["memory_source_id"])
                        
        case["relevant_memory_ids"] = list(relevant_ids)

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Annotated {len(dataset)} cases.")

if __name__ == "__main__":
    annotate_dataset()
