# 🤖 QwenRAG

QwenRAG is a retrieval-augmented generation project built around LangGraph and Ollama. It supports multiple retrieval strategies (static, dense, sparse BM25, and Qwen embeddings), provides a Streamlit chatbot UI, and includes batch generation plus retrieval and QA evaluation scripts.

## 📁 What is in this repo

```text
qwenRAG/
├── configs/
│   ├── agent.yaml
│   └── eval.yaml
├── data/
├── results/
└── src/
    ├── rag_agent/
  │   ├── rag.py
  │   ├── retriever.py
  │   ├── dataloader.py
  │   ├── chatbot.py
  │   └── batch_generate.py
    └── eval/
    ├── eval_retrieval.py
    └── eval_hotpotqa.py
```

Key paths:

- `configs/agent.yaml`: central runtime configuration (model, retriever, paths, defaults)
- `configs/eval.yaml`: evaluation defaults (gold/pred paths, retrieval cutoffs, top-k)
- `data/`: collection and QA datasets
- `results/`: batch outputs and score files
- `src/rag_agent/rag.py`: LangGraph workflow for rewrite, retrieval, and answer generation
- `src/rag_agent/retriever.py`: Chroma (static/dense/qwen) and BM25 sparse retriever loader
- `src/rag_agent/dataloader.py`: builds Chroma collections from `data/collection.jsonl`
- `src/rag_agent/chatbot.py`: Streamlit chatbot frontend
- `src/rag_agent/batch_generate.py`: batch inference over JSONL question files
- `src/eval/eval_retrieval.py`: retrieval metrics via pytrec_eval
- `src/eval/eval_hotpotqa.py`: Hotpot-style answer/supporting-fact evaluation

## ✅ Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency sync
- [Ollama](https://ollama.com/) with local model pulled

## ⚙️ Setup

Pull the default model used by `configs/agent.yaml`:

```bash
ollama pull qwen2.5:7b-instruct
```

Install project dependencies:

```bash
uv sync
```

All commands below can be run with `uv run ...` so you do not need to manually activate a virtual environment.

## 🚀 Quickstart

After completing Setup, run these commands from the repository root:

```bash
# 1) 🗂️ Build/refresh Chroma collections (static, dense, qwen)
uv run python src/rag_agent/dataloader.py

# 2) 🧠 Generate validation predictions with dense retrieval
uv run python src/rag_agent/batch_generate.py -f data/validation.jsonl -e dense

# 3) 📊 Evaluate retrieval quality (uses defaults from configs/eval.yaml)
uv run python src/eval/eval_retrieval.py

# 4) 🧪 Evaluate QA quality and supporting docs (uses defaults from configs/eval.yaml)
uv run python src/eval/eval_hotpotqa.py
```

Optional: launch the chatbot UI after indexing:

```bash
uv run streamlit run src/rag_agent/chatbot.py
```

## 🧩 Configuration

Main runtime config is in `configs/agent.yaml`:

- `llm`: provider/model/temperature/history settings
- `data.collection_path`: input corpus JSONL
- `rag.vector_store.persistence_dir`: Chroma persistence directory
- `rag.retriever.embedding`: model names for static/dense/qwen
- `batch`: default dataset, output dir, and embedding type

Evaluation defaults are in `configs/eval.yaml`:

- `defaults.gold` / `defaults.pred`: default input/output files used by eval scripts
- `retrieval.k_values`: cutoffs for MAP/NDCG/Recall/Precision in retrieval eval
- `hotpotqa.topk`: number of retrieved docs considered in Hotpot-style eval

Default retrieval strategies available at runtime:

- `static`
- `dense`
- `sparse`
- `qwen`

## 🗂️ Build or refresh vector stores

Generate Chroma collections for static, dense, and qwen embeddings:

```bash
uv run python src/rag_agent/dataloader.py
```

This reads `data/collection.jsonl` and writes to `data/chromadb/`.

## 💬 Run chatbot UI

```bash
uv run streamlit run src/rag_agent/chatbot.py
```

In the UI, select retrieval strategy (`static`, `sparse`, `dense`, `qwen`) from the dropdown.

## 🏭 Run batch generation

Generate answers for a dataset:

```bash
uv run python src/rag_agent/batch_generate.py -f data/test.jsonl -e dense
```

Useful options:

- `-f, --file`: input JSONL file with at least `id` and `text`
- `-e, --embed`: retrieval mode (`static`, `dense`, `sparse`, `qwen`)
- `-s, --skip_chain`: disable chain-of-thought node and answer directly from retrieved context

Output is written to `results/<input_stem>_<embed>.jsonl`.

## 📊 Evaluate retrieval

Default command (uses `configs/eval.yaml`):

```bash
uv run python src/eval/eval_retrieval.py
```

Optional overrides:

```bash
uv run python src/eval/eval_retrieval.py \
  --gold data/validation.jsonl \
  --pred results/validation_dense.jsonl \
  --k_values 2 5 10
```

The script reports MAP, NDCG, Recall, and Precision at cutoffs 2/5/10.

## 🧪 Evaluate QA + supporting docs

Default command (uses `configs/eval.yaml`):

```bash
uv run python src/eval/eval_hotpotqa.py
```

Optional overrides:

```bash
uv run python src/eval/eval_hotpotqa.py \
  --gold data/validation.jsonl \
  --pred results/validation_dense.jsonl \
  --topk 10
```

This reports EM/F1 for answers, supporting-doc metrics, and joint metrics.

## 🧾 Input and output JSONL format

Typical input row (`data/validation.jsonl`):

```json
{
  "id": "...",
  "text": "question",
  "answer": "gold answer",
  "supporting_ids": ["doc-1", "doc-2"]
}
```

Typical batch output row:

```json
{
  "id": "...",
  "text": "question",
  "answer": "predicted answer",
  "retrieved_docs": [
    ["doc-1", 0.67],
    ["doc-2", 0.59]
  ]
}
```

## 📝 Notes

- Sparse retrieval requires `rank-bm25` (already included in project dependencies).
- If Chroma collections are missing, run the dataloader first.
- `src/api/main.py` is currently empty and not part of the runtime workflow.
