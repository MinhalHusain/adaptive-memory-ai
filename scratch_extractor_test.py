from src.memory.graph_memory import GraphMemory
mem = GraphMemory()

sentences = [
    "Hi there! My name is Alex.", # FACT_RETENTION
    "I have a doctor's appointment on November 3rd with Dr. Patel at the City Medical Center.", # LRR
    "I just adopted a golden retriever puppy named Biscuit.", # INFO_UPDATE / FACT_RETENTION
    "My best friend's name is Marcus and he lives in Berlin.", # MULTI_HOP
    "I'm allergic to shellfish and penicillin.", # PERSONALIZATION
    "I like jazz and I like Radiohead.", # REDUNDANT
    "I live in Seattle.", # CONFLICTING (before)
    "I actually live in Portland now." # CONFLICTING (after)
]

for s in sentences:
    print(f"[{s}]")
    for t in mem._extract_triples(s):
        print(f"  -> {t}")
    print()
