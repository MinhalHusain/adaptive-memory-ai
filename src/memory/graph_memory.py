import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple
import datetime
import uuid
import re
from src.memory.base import BaseMemory
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphMemory(BaseMemory):
    def __init__(self, top_k: int = 5):
        self.graph = nx.MultiDiGraph()
        self.top_k = top_k
        self.predicates = [
            " is studying ", " am studying ", " studies ", " study ",
            " lives in ", " live in ", " is located in ", " located in ",
            " works at ", " work at ", " works as ", " work as ",
            " is named ", " named ", " name is ",
            " is allergic to ", " am allergic to ", " allergic to ",
            " has a ", " have a ", " has ", " have ",
            " is ", " are ", " was ", " were ",
            " like ", " likes ", " love ", " loves ",
            " to ", " in ", " at ", " with ", " on ", " for ", " about "
        ]
        
        # Define which predicates should overwrite previous facts (single-valued)
        self.single_valued_predicates = {
            " lives in ", " live in ", " is located in ", " located in ",
            " works at ", " work at ", " works as ", " work as ",
            " name is "
        }
        
    def _extract_triples(self, text: str) -> List[tuple]:
        # Handle Mr. Dr. etc so they don't split
        text = text.replace('Dr.', 'Dr').replace('Mr.', 'Mr').replace('Ms.', 'Ms')
        text = text.replace('!', '.').replace('?', '.').replace(',', '.')
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        triples = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Normalize pronouns
            if sentence_lower.startswith("i'm "):
                sentence_lower = "user is " + sentence_lower[4:]
                sentence = "User is " + sentence[4:]
            elif sentence_lower.startswith("i am "):
                sentence_lower = "user is " + sentence_lower[5:]
                sentence = "User is " + sentence[5:]
            elif sentence_lower.startswith("my "):
                sentence_lower = "user's " + sentence_lower[3:]
                sentence = "User's " + sentence[3:]
            elif sentence_lower.startswith("i "):
                sentence_lower = "user " + sentence_lower[2:]
                sentence = "User " + sentence[2:]
                
            # Find earliest predicate
            earliest_idx = -1
            best_pred = None
            for pred in self.predicates:
                idx = sentence_lower.find(pred)
                if idx != -1:
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
                        best_pred = pred
                        
            if best_pred:
                sub = sentence[:earliest_idx].strip()
                obj = sentence[earliest_idx + len(best_pred):].strip()
                if sub and obj:
                    if sub.lower() in ["i", "we"]: sub = "User"
                    if sub.lower() in ["my", "mine", "our"]: sub = "User's"
                    triples.append((sub, best_pred.strip(), obj))
                else:
                    triples.append(("User", "stated", sentence))
            else:
                triples.append(("User", "stated", sentence))
                
        return triples

    def add(self, session_id: str, data: Dict[str, Any]) -> None:
        content = data.get("user_input", "")
        if not content:
            return
            
        metadata = data.get("metadata", {})
        source_id = metadata.get("memory_source_id")
        
        triples = self._extract_triples(content)
        
        for sub, pred, obj in triples:
            if not self.graph.has_node(sub):
                self.graph.add_node(sub, type="Entity")
            if not self.graph.has_node(obj):
                self.graph.add_node(obj, type="Entity")
                
            edges_to_remove = []
            # Only enforce uniqueness (conflict overwrite) for explicitly single-valued predicates
            is_single_valued = any(pred == p.strip() for p in self.single_valued_predicates)
            
            if is_single_valued:
                for _, _, key, edge_data in self.graph.out_edges(sub, keys=True, data=True):
                    if edge_data.get("predicate") == pred:
                        edges_to_remove.append((sub, _, key))
            
            for edge in edges_to_remove:
                self.graph.remove_edge(*edge)
                
            edge_id = str(uuid.uuid4())
            self.graph.add_edge(
                sub, obj, 
                key=edge_id,
                predicate=pred, 
                memory_source_id=source_id,
                content=content,
                timestamp=datetime.datetime.now().isoformat(),
                metadata=metadata
            )
            logger.debug(f"GraphMemory added edge: ({sub}) -[{pred}]-> ({obj})")

    def _tokenize(self, text: str) -> Set[str]:
        words = set(re.findall(r'\b\w+\b', text.lower()))
        stop_words = {"what", "is", "my", "where", "am", "i", "can", "you", "tell", "me", "about", "who", "when", "how", "do", "does", "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "up", "about", "into", "over", "after", "and", "or", "but", "user", "s"}
        return words - stop_words

    def retrieve(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_keywords = self._tokenize(query)
        
        if not query_keywords:
            return []
            
        entry_nodes = set()
        
        # 1. Identify entry nodes from query overlap
        for node in self.graph.nodes():
            node_keywords = self._tokenize(node)
            if query_keywords.intersection(node_keywords):
                entry_nodes.add(node)
                
        # Also check edge contents for entry points to catch predicates/full text
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            content_keywords = self._tokenize(data.get("content", ""))
            if query_keywords.intersection(content_keywords):
                entry_nodes.add(u)
                entry_nodes.add(v)

        # 2. Bounded Traversal (Max 2 Hops)
        traversed_edges = {} # edge_key -> (score, edge_tuple)
        
        # Helper to score an edge
        def score_edge(u, v, data, hop_penalty=1.0):
            score = 0
            u_lower = u.lower()
            v_lower = v.lower()
            p_lower = data.get("predicate", "").lower()
            c_lower = data.get("content", "").lower()
            
            for kw in query_keywords:
                if kw in u_lower: score += 1.0
                if kw in v_lower: score += 1.0
                if kw in p_lower: score += 0.5
                if kw in c_lower and kw not in u_lower and kw not in v_lower and kw not in p_lower:
                    score += 0.2
            return score * hop_penalty
            
        # Hop 1
        hop1_nodes = set()
        for node in entry_nodes:
            # Outgoing
            for u, v, key, data in self.graph.out_edges(node, keys=True, data=True):
                score = score_edge(u, v, data, hop_penalty=1.0)
                if score > 0:
                    traversed_edges[key] = (score, (u, v, key, data))
                    hop1_nodes.add(v)
            # Incoming
            for u, v, key, data in self.graph.in_edges(node, keys=True, data=True):
                score = score_edge(u, v, data, hop_penalty=1.0)
                if score > 0:
                    traversed_edges[key] = (score, (u, v, key, data))
                    hop1_nodes.add(u)
                    
        # Hop 2
        for node in hop1_nodes:
            # Outgoing
            for u, v, key, data in self.graph.out_edges(node, keys=True, data=True):
                if key not in traversed_edges:
                    score = score_edge(u, v, data, hop_penalty=0.5) # Penalty for distance
                    if score > 0:
                        traversed_edges[key] = (score, (u, v, key, data))
            # Incoming
            for u, v, key, data in self.graph.in_edges(node, keys=True, data=True):
                if key not in traversed_edges:
                    score = score_edge(u, v, data, hop_penalty=0.5)
                    if score > 0:
                        traversed_edges[key] = (score, (u, v, key, data))

        # 3. Sort and Format
        scored_edges = sorted(list(traversed_edges.values()), key=lambda x: x[0], reverse=True)
        
        results = []
        seen_contents = set()
        for score, (u, v, key, data) in scored_edges:
            content = data.get("content", f"{u} {data.get('predicate')} {v}")
            if content in seen_contents:
                continue
            seen_contents.add(content)
            
            results.append({
                "memory_id": key,
                "memory_source_id": data.get("memory_source_id"),
                "content": content,
                "similarity": score,
                "metadata": data.get("metadata", {})
            })
            if len(results) >= self.top_k:
                break
                
        return results

    def update(self, memory_id: str, data: Dict[str, Any]) -> None:
        new_content = data.get("content")
        if not new_content:
            return
            
        target_edge = None
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            if key == memory_id:
                target_edge = (u, v, key, edge_data)
                break
                
        if target_edge:
            u, v, key, edge_data = target_edge
            self.graph.remove_edge(u, v, key=key)
            self.add("dummy_session", {"user_input": new_content, "metadata": edge_data.get("metadata", {})})

    def forget(self, memory_id: str) -> None:
        target_edge = None
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            if key == memory_id:
                target_edge = (u, v, key)
                break
        if target_edge:
            self.graph.remove_edge(*target_edge)

    def reset(self) -> None:
        self.graph.clear()
        logger.info("GraphMemory reset: all memories cleared")
        
    def count(self) -> int:
        return self.graph.number_of_edges()
