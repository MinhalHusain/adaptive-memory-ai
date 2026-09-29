import re

def extract_triples(text):
    text = text.replace('!', '.').replace('?', '.').replace(',', '.')
    sentences = [s.strip().lower() for s in text.split('.') if s.strip()]
    triples = []
    
    for sentence in sentences:
        # Simplify common phrases
        sentence = re.sub(r"i\'m|i am", "user is", sentence)
        sentence = re.sub(r"my name is", "user name is", sentence)
        sentence = re.sub(r"my", "user", sentence)
        sentence = re.sub(r"i have", "user has", sentence)
        sentence = re.sub(r"i live in", "user lives in", sentence)
        sentence = re.sub(r"i study", "user studies", sentence)
        sentence = re.sub(r"i work as a", "user profession is", sentence)
        sentence = re.sub(r"i work at", "user company is", sentence)
        sentence = re.sub(r"i work in", "user department is", sentence)
        sentence = re.sub(r"i", "user", sentence)
        
        # Split by verbs/prepositions
        splitters = [" is ", " has ", " lives in ", " studies ", " works at ", " profession is ", " company is ", " named ", " at ", " in ", " with ", " on ", " about ", " for ", " to "]
        
        for splitter in splitters:
            if splitter in sentence:
                parts = sentence.split(splitter, 1)
                sub = parts[0].strip()
                obj = parts[1].strip()
                
                # Further split the object if it contains another relation
                for sub_split in [" at ", " in ", " with ", " on ", " about ", " for ", " to "]:
                    if sub_split in obj:
                        obj_parts = obj.split(sub_split, 1)
                        triples.append((sub, splitter.strip(), obj_parts[0].strip()))
                        triples.append((obj_parts[0].strip(), sub_split.strip(), obj_parts[1].strip()))
                        break
                else:
                    triples.append((sub, splitter.strip(), obj))
                break
                
    # Fallback if no triples
    if not triples:
        triples.append(("user", "stated", text.strip()))
        
    return triples

texts = [
    "Hi there! My name is Alex.",
    "I'm studying computer science at MIT.",
    "I just adopted a golden retriever puppy named Biscuit.",
    "I have a doctor's appointment on November 3rd with Dr. Patel at the City Medical Center."
]
for t in texts:
    print(f"TEXT: {t}")
    print(f"TRIPLES: {extract_triples(t)}\n")
