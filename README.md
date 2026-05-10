<div align="center">

<br>

<h1>🧠 Hybrid-Search RAG Engine</h1>

**Production-grade Retrieval-Augmented Generation API for long-document QA.**

<br>

Combines **FAISS dense search** and **BM25 keyword search** fused with **Reciprocal Rank Fusion**, powered by GPT-4o.

<br>

<a href="https://github.com/im-anishraj/Hybrid-Search-RAG-Engine"><img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python&logoColor=white&labelColor=0d1117" alt="Python"></a>&nbsp;
<a href="https://github.com/im-anishraj/Hybrid-Search-RAG-Engine"><img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white&labelColor=0d1117" alt="FastAPI"></a>&nbsp;
<a href="https://github.com/im-anishraj/Hybrid-Search-RAG-Engine"><img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white&labelColor=0d1117" alt="Docker"></a>&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&labelColor=0d1117" alt="MIT"></a>&nbsp;
<a href="https://gssoc.girlscript.tech/"><img src="https://img.shields.io/badge/GSSoC-2026-ff6b35?style=flat-square&labelColor=0d1117" alt="GSSoC 2026"></a>

<br><br>

<a href="#%EF%B8%8F-architecture">Architecture</a>&ensp;·&ensp;<a href="#-tech-stack">Tech Stack</a>&ensp;·&ensp;<a href="#-quickstart">Quickstart</a>&ensp;·&ensp;<a href="#-api-endpoints">API</a>&ensp;·&ensp;<a href="#-benchmarks">Benchmarks</a>

</div>

<br>

---

<br>

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                               │
│                                                                         │
│  PDF/DOCX/TXT ──► loader.py ──► chunker.py ──► embedder.py             │
│                   (PageRecord)  (SemanticChunker  (OpenAI               │
│                                  95th-pct split)   text-emb-3-small)    │
│                                       │                                 │
│                          ┌────────────┴────────────┐                   │
│                          ▼                         ▼                   │
│                   vector_store.py           bm25_store.py              │
│                   (FAISS IndexFlatIP)        (BM25Okapi)               │
│                   data/vector_store/         data/bm25_store/          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                                  │
│                                                                         │
│  Question ──► get_query_embedding()                                     │
│                    │ (one API call)                                      │
│         ┌──────────┴──────────┐                                         │
│         ▼                     ▼                                         │
│   VectorStore             BM25Store                                     │
│   .search(k=10)           .search(k=10)                                 │
│   cosine ANN              token TF-IDF                                  │
│         └──────────┬──────────┘                                         │
│                    ▼                                                     │
│          reciprocal_rank_fusion()                                        │
│          score(d) = Σ 1/(rank_i(d) + 60)                               │
│                    │                                                     │
│                    ▼                                                     │
│            top-5 fused chunks                                            │
│                    │                                                     │
│                    ▼                                                     │
│           RAGGenerator.generate()                                        │
│           GPT-4o · temp=0 · strict citation prompt                     │
│                    │                                                     │
│                    ▼                                                     │
│   {"answer": "...", "sources": [...], "confidence_score": 0.94}        │
└─────────────────────────────────────────────────────────────────────────┘
```

<br>

---

<br>

## 🛠 Tech Stack

<table>
  <tr>
    <th>Layer</th>
    <th>Library</th>
    <th>Version</th>
  </tr>
  <tr><td><b>Embedding</b></td><td>OpenAI text-embedding-3-small</td><td><code>openai 1.75.0</code></td></tr>
  <tr><td><b>Generation</b></td><td>GPT-4o (temp=0)</td><td><code>langchain-openai 1.1.11</code></td></tr>
  <tr><td><b>Dense index</b></td><td>FAISS IndexFlatIP / IVFFlat</td><td><code>faiss-cpu 1.9.0</code></td></tr>
  <tr><td><b>Sparse index</b></td><td>BM25Okapi (k1=1.5, b=0.75)</td><td><code>rank-bm25 0.2.2</code></td></tr>
  <tr><td><b>Chunking</b></td><td>SemanticChunker (95th-pct)</td><td><code>langchain-experimental 0.3.4</code></td></tr>
  <tr><td><b>Chain</b></td><td>LCEL RunnableSequence</td><td><code>langchain-core 1.2.18</code></td></tr>
  <tr><td><b>API</b></td><td>FastAPI + uvicorn</td><td><code>0.115.14</code> / <code>0.34.3</code></td></tr>
  <tr><td><b>Validation</b></td><td>Pydantic v2</td><td><code>2.11.4</code></td></tr>
</table>

<br>

---

<br>

## 🚀 Quickstart

### 1. Clone and install
```bash
git clone https://github.com/im-anishraj/Hybrid-Search-RAG-Engine.git
cd Hybrid-Search-RAG-Engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run
```bash
uvicorn app.main:app --reload --port 8000
```

> OpenAPI docs are available at `http://localhost:8000/docs`.

### 4. Docker (Recommended for production)
```bash
cp .env.example .env        # fill in OPENAI_API_KEY
docker compose up --build -d
docker compose logs -f
```

<br>

---

<br>

## 🌐 API Endpoints

### `POST /ingest`
Upload a PDF, DOCX, or TXT file. Additive — each call accumulates into the same corpus without replacing prior documents.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@annual_report.pdf"
```

<details>
<summary><b>View Response</b></summary>

```json
{
  "doc_id": "annual_report.pdf",
  "filename": "annual_report.pdf",
  "pages_loaded": 47,
  "chunks_indexed": 112,
  "message": "'annual_report.pdf' ingested successfully. 47 pages → 112 chunks indexed."
}
```
</details>

### `POST /query`
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the EBITDA margin in Q3 2023?"}'
```

<details>
<summary><b>View Response</b></summary>

```json
{
  "question": "What was the EBITDA margin in Q3 2023?",
  "answer": "The EBITDA margin in Q3 2023 was 18.5%, driven by operational efficiency improvements [Source: annual_report.pdf, page 15].",
  "sources": [
    {"filename": "annual_report.pdf", "page_num": 15, "chunk_id": "annual_report.pdf::p15::c2"}
  ],
  "confidence_score": 0.94,
  "can_answer": true,
  "model": "gpt-4o",
  "retrieved_chunks": []
}
```
</details>

<br>

---

<br>

## 🏆 Benchmarks

**Hit Rate@5 on a 20-query synthetic corpus (30 chunks, 5 topics)**

| Engine | Hit Rate @ 5 | Accuracy |
|:---|:---:|:---:|
| **Hybrid (RRF)** | **19 / 20** | **95%** |
| BM25 (Sparse only) | 18 / 20 | 90% |
| FAISS (Dense only) | 17 / 20 | 85% |

> The hybrid-exclusive hit demonstrates the RRF value: FAISS finds a chunk via semantic similarity while BM25 misses it due to zero lexical overlap. RRF promotes chunks both retrievers agree on.

Run the benchmark locally:
```bash
pytest tests/test_retrieval.py -v -s
```

<br>

---

<br>

## 🧠 Design Notes

- **Why `--workers 1`?** FAISS's C++ index is not fork-safe. Scale horizontally with multiple containers behind a load balancer instead of multiple workers per container.
- **Why semantic chunking?** Fixed 512-token windows slice sentences mid-thought. `SemanticChunker` detects topic-shift boundaries via cosine distance spikes, producing one-complete-idea chunks. The LLM receives coherent passages, not sentence fragments.
- **Why BM25 alongside FAISS?** Dense embeddings smear rare tokens. "EBITDA" and "CRISPR-Cas9" map to broad semantic regions shared by adjacent-but-wrong terms. BM25's exact token matching catches them precisely. 

<br>

---

<br>

<div align="center">

<br>

**Built with precision. Open for contributions.**

<br>

<a href="https://github.com/im-anishraj/Hybrid-Search-RAG-Engine/stargazers"><img src="https://img.shields.io/github/stars/im-anishraj/Hybrid-Search-RAG-Engine?style=flat-square&logo=github&labelColor=0d1117&color=e3b341&label=stars" alt="Stars"></a>&ensp;
<a href="https://github.com/im-anishraj/Hybrid-Search-RAG-Engine/network/members"><img src="https://img.shields.io/github/forks/im-anishraj/Hybrid-Search-RAG-Engine?style=flat-square&logo=github&labelColor=0d1117&color=8b949e&label=forks" alt="Forks"></a>

<br>

<sub>Licensed under MIT · Maintained by <a href="https://github.com/im-anishraj">@im-anishraj</a></sub>

</div>