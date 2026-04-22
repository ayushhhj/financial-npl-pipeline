import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

RAW_DIR = Path("data/raw")
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_chunks() -> tuple[list[str], list[dict]]:
    chunks = []
    metadata = []

    for company_dir in sorted(RAW_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        for filing_path in sorted(company_dir.glob("*.json")):
            with open(filing_path) as f:
                data = json.load(f)
            for i, chunk in enumerate(data["chunks"]):
                if len(chunk.split()) < 30:
                    continue
                chunks.append(chunk)
                metadata.append({
                    "ticker": data["ticker"],
                    "accession": data["accession"],
                    "date": data["date"],
                    "chunk_index": i,
                })

    return chunks, metadata


def build_faiss_index(chunks: list[str], metadata: list[dict]) -> None:
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(EMBEDDINGS_DIR / "index.faiss"))
    with open(EMBEDDINGS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(EMBEDDINGS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Index built with {index.ntotal} vectors")


def search(query: str, top_k: int = 5) -> list[dict]:
    index = faiss.read_index(str(EMBEDDINGS_DIR / "index.faiss"))
    with open(EMBEDDINGS_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(EMBEDDINGS_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": chunks[idx][:300],
            "metadata": metadata[idx],
        })

    return results


if __name__ == "__main__":
    chunks, metadata = load_all_chunks()
    build_faiss_index(chunks, metadata)

    print("\n--- Test searches ---")
    for query in [
        "supply chain risk and manufacturing disruption",
        "artificial intelligence machine learning investment",
        "interest rate exposure credit risk",
        "regulatory antitrust government investigation",
    ]:
        print(f"\nQuery: '{query}'")
        results = search(query, top_k=3)
        for r in results:
            print(f"  [{r['metadata']['ticker']} {r['metadata']['date']}] score={r['score']:.3f}")
            print(f"  {r['text'][:150]}...")