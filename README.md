# Adaptive Memory Management for Long-Term Conversational AI

## Project Purpose
This research project experimentally compares different long-term conversational memory approaches for Large Language Models (LLMs). The project evaluates how memory architectures affect recall, precision, context limits, and cost across extended multi-session interactions.

**Currently implemented:**
1. **No-Memory (Baseline):** A pipeline that operates strictly on the current session's history and persists nothing.
2. **Vector Memory (Baseline):** A persistent baseline storing full user messages embedded via Sentence Transformers into ChromaDB.
3. **Evaluation Framework:** A dataset of 50 curated test scenarios with ground-truth memory annotations and a pluggable evaluation runner.

**Future planned memory approaches (not yet implemented):**
- Knowledge graph memory
- Hybrid vector + knowledge graph memory
- Adaptive memory management

## Project Architecture

```
adaptive-memory-ai/
├── src/
│   ├── llm/
│   │   └── client.py              # LLM abstraction (OpenAI, Mock)
│   ├── conversation/
│   │   ├── session.py             # Short-term session history
│   │   └── pipeline.py            # Coordinates LLM + memory + session
│   ├── memory/
│   │   ├── base.py                # Abstract BaseMemory interface
│   │   ├── no_memory.py           # No-Memory baseline
│   │   ├── vector_memory.py       # ChromaDB-based vector memory
│   │   └── embeddings.py          # Configurable embedding providers
│   ├── evaluation/
│   │   ├── schema.py              # EvaluationRecord, TestCase, constants
│   │   ├── evaluators.py          # Evaluators including Retrieval metrics
│   │   └── runner.py              # Dataset loader + EvaluationRunner
│   └── utils/
│       └── logger.py              # Centralized logging
├── data/
│   └── evaluation/
│       └── dataset.json           # 50 scenarios with memory_source_ids
├── experiments/
│   ├── run_baseline.py            # Basic No-Memory session demo
│   ├── run_evaluation.py          # No-Memory dataset evaluation run
│   └── run_vector_memory.py       # Vector Memory dataset evaluation run
├── results/                       # Exported evaluation outputs
├── tests/
│   └── ...                        # Comprehensive pytest suite
├── requirements.txt
└── .env.example
```

## Evaluation Dataset & Metrics

### Ground-Truth Annotation
Each user message in the dataset scenarios has a stable `memory_source_id`. The dataset scenarios define `relevant_memory_ids` marking the exact messages required to answer the query. This decoupled ground-truth annotation ensures that retrieval metrics are calculated objectively regardless of whether the system uses Vector Memory or Knowledge Graphs.

### Retrieval Metrics vs. Answer Metrics
We strictly separate:
- **Retrieval Metrics:** Did the memory system fetch the right information? (Measured via `Recall@K` and `Precision@K`).
- **Answer Metrics:** Did the LLM use that information to generate a correct response? (Measured via Exact Match and Semantic Evaluation).

### Diagnostic vs. Research Metrics
- **Research Metrics:** (e.g., ID-based Recall@K). These evaluate exact matching against ground-truth source IDs.
- **Diagnostic Metrics:** (e.g., Legacy Keyword Proxy Recall). These heuristically check if the retrieved text contains expected keywords, useful for debugging when ground-truth mappings fail or for systems that heavily mutate the original text.

## Vector Memory Baseline

The Vector Memory baseline uses ChromaDB for persistent storage.

**Why Sentence Transformers?**
For real research experiments, we use the `all-MiniLM-L6-v2` model via `SentenceTransformerEmbedding` because it provides true semantic capabilities (e.g., synonym matching, semantic proximity) completely offline and free, avoiding API costs during high-volume experiments.

**Why FakeEmbedding?**
We include a deterministic `FakeEmbedding` class (using trigram hashing) purely for offline unit tests to run instantly without downloading gigabytes of neural network weights or requiring GPU availability.

## Setup Instructions

1. Ensure Python 3.9+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install sentence-transformers  # Required for real Vector Memory experiments
   ```
3. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```
   If you want answer-level evaluation, you must set `OPENAI_API_KEY`. If not set, the framework uses a Mock LLM that returns placeholder text (Answer Correctness will be 0.0, but Retrieval Metrics will still be calculated accurately).

## Running the Real Vector Memory Experiment
```bash
# Uses sentence-transformers and ground-truth ID metrics
python experiments/run_vector_memory.py
```
The results will be exported to `results/vector_memory_real_embedding.json`.
