from src.memory.hybrid_memory import HybridMemory
from src.memory.vector_memory import VectorMemory
from src.memory.graph_memory import GraphMemory
from src.memory.embeddings import FakeEmbedding
import tempfile

tmpdir = tempfile.mkdtemp()
vm = VectorMemory(embedding=FakeEmbedding(64), persist_dir=tmpdir, collection_name='test', top_k=5)
gm = GraphMemory(top_k=5)
hm = HybridMemory(vm, gm, alpha=0.5, top_k=5)

hm.add('s1', {'user_input': 'My name is Alice.', 'metadata': {'memory_source_id': 'msg_1'}})
hm.add('s1', {'user_input': 'I live in Berlin.', 'metadata': {'memory_source_id': 'msg_2'}})
hm.add('s1', {'user_input': 'I work at Google.', 'metadata': {'memory_source_id': 'msg_3'}})

results = hm.retrieve('Where do I live?')
print('Results:', len(results))
for r in results:
    src = r.get('retrieval_source', '?')
    sid = r.get('memory_source_id', '?')
    hyb = r.get('hybrid_score', 0)
    content = r.get('content', '')[:60]
    nv = r.get('norm_vector_score', 0)
    ng = r.get('norm_graph_score', 0)
    print(f"  [{src}] src={sid} hyb={hyb:.3f} nv={nv:.3f} ng={ng:.3f} content={content}")
print('Vector count:', hm.count()['vector'])
print('Graph count:', hm.count()['graph'])

# Test reset
hm.reset()
print('After reset - Vector:', hm.count()['vector'], 'Graph:', hm.count()['graph'])
