import re

def extract_triples(text):
    text = text.replace('!', '.').replace('?', '.')
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    triples = []
    
    # Common verbs/prepositions to split on
    # Ordered by length to match longest first
    splitters = [
        " am allergic to ", " is allergic to ", " have a ", " has a ", 
        " work as a ", " working as a ", " work at ", " working at ",
        " study at ", " studying at ", " live in ", " lives in ",
        " is named ", " named ", " is a ", " are a ",
        " am ", " is ", " are ", " was ", " were ",
        " have ", " has ", " had ",
        " like ", " likes ", " love ", " loves ",
        " work ", " works ", " study ", " studies ",
        " to ", " in ", " at ", " with ", " on ", " for "
    ]
    
    for sentence in sentences:
        sentence = sentence.lower()
        # Very naive: find the first splitter that exists in the sentence
        for splitter in splitters:
            if splitter in sentence:
                parts = sentence.split(splitter, 1)
                sub = parts[0].strip()
                obj = parts[1].strip()
                
                # Cleanup subjects
                if sub in ["i", "i'm", "i am", "my", "mine", "we", "we're"]:
                    sub = "user"
                elif sub.startswith("my "):
                    sub = "user's " + sub[3:]
                    
                # Secondary split on object (e.g. "software engineer at Stripe")
                for sub_split in [" at ", " in ", " with ", " on ", " for "]:
                    if sub_split in obj:
                        obj_parts = obj.split(sub_split, 1)
                        triples.append((sub, splitter.strip(), obj_parts[0].strip()))
                        triples.append((obj_parts[0].strip(), sub_split.strip(), obj_parts[1].strip()))
                        break
                else:
                    triples.append((sub, splitter.strip(), obj))
                break
                
    return triples

# Test it
texts = [
    "Hi there! My name is Alex.",
    "I'm studying computer science at MIT.",
    "I just adopted a golden retriever puppy named Biscuit.",
    "I have a doctor's appointment on November 3rd with Dr. Patel at the City Medical Center.",
    "I work as a senior software engineer at Stripe."
]
for t in texts:
    print(f"TEXT: {t}")
    print(f"TRIPLES: {extract_triples(t)}\n")
