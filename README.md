# Financial NLP Pipeline

An end-to-end NLP system for analyzing SEC 10-K annual filings across major public companies. Extracts named entities, classifies their context using zero-shot NLP, builds a knowledge graph, and exposes everything via a semantic search API.

## What It Does

- Ingests 10-K filings from SEC EDGAR for 10 major companies covering 2021–2026
- Extracts named entities (companies, people, locations, products, financial figures) using spaCy
- Classifies each entity mention by context using zero-shot NLP with BART — distinguishing China as a supplier vs market vs regulatory risk, executives vs key-person risks, competitors vs partners
- Builds a Neo4j knowledge graph connecting companies, executives, locations, and filings with typed relationships
- Indexes all filing text with sentence transformers for semantic search via FAISS
- Discovers latent themes across filings using BERTopic topic modeling
- Exposes everything via a FastAPI REST API

## Architecture

```
SEC EDGAR API
      ↓
Ingestion (requests, BeautifulSoup)
      ↓
Text Cleaning + Chunking
      ↓
NER (spaCy en_core_web_lg)
      ↓
Zero-shot Classification (BART facebook/bart-large-mnli)
      ↓
┌─────────────────────┬──────────────────────┐
│  Neo4j Graph        │  FAISS Vector Index  │
│  (entities +        │  (semantic search)   │
│   relationships)    │                      │
└─────────────────────┴──────────────────────┘
                  ↓
            FastAPI REST API
```

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion | SEC EDGAR API, requests, BeautifulSoup |
| NLP | spaCy, HuggingFace Transformers (BART zero-shot) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2), FAISS |
| Topic modeling | BERTopic |
| Graph database | Neo4j |
| API | FastAPI, uvicorn |
| Infrastructure | Docker, GitHub Actions CI/CD |

## Setup

```bash
# 1. clone and install
git clone https://github.com/ayushhhj/financial-nlp-pipeline
cd financial-nlp-pipeline
make install

# 2. start Neo4j
make up

# 3. run full pipeline (ingestion → NER → embeddings → topics → graph)
make pipeline

# 4. start API
uvicorn src.api.main:app --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/search` | POST | Semantic search over all filing text |
| `/entities/{ticker}` | GET | Named entities from a company's filings with roles and categories |
| `/analysis/location-purpose` | GET | Why companies mention specific locations |
| `/graph/{ticker}/related` | GET | Related companies via graph traversal |
| `/companies` | GET | List all companies in the system |

### Example Queries

```bash
# semantic search across all filings
curl -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "supply chain risk China manufacturing", "top_k": 5}'

# is China a market or supplier for Apple?
curl "localhost:8000/analysis/location-purpose?location=China&category=supplier"

# Apple executive team with roles
curl localhost:8000/entities/AAPL

# companies related to Tesla
curl localhost:8000/graph/TSLA/related
```

## Key Findings

- **Apple** mentions China primarily as a market (29 mentions) vs supplier (13) — counterintuitive given manufacturing narrative, but reflects the relative weight of revenue risk vs supply chain risk in their 10-K language
- **NVIDIA** uniquely mentions Israel as a supplier location due to the Mellanox acquisition
- **JPMorgan and Goldman Sachs** have the richest climate risk disclosure, reflecting financial regulatory requirements around physical and transition risk
- **Apple** dominates single-source supplier concentration risk language across all years

## Limitations

- Entity classification uses zero-shot inference — strong on clear cases, weaker on ambiguous financial language
- Location deduplication handles common variants but misses unusual abbreviations
- Dataset covers 10 companies; scaling to S&P 500 would require distributed processing and GPU-accelerated classification
- Co-mention relationships between companies are sparse since 10-Ks rarely name competitors directly

## Project Structure

```
financial-nlp-pipeline/
├── src/
│   ├── ingestion/      # SEC EDGAR fetching and cleaning
│   ├── nlp/            # NER, classification, embeddings, topics
│   ├── graph/          # Neo4j loading
│   └── api/            # FastAPI endpoints
├── tests/              # pytest test suite
├── Makefile            # setup, pipeline, test commands
├── docker-compose.yml  # Neo4j container
└── requirements.txt
```