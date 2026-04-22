import json
import pickle
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
TOPICS_DIR = Path("data/topics")
TOPICS_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks_with_metadata() -> tuple[list[str], list[dict]]:
    chunks, metadata = [], []
    for company_dir in sorted(RAW_DIR.iterdir()):
        if not company_dir.is_dir():
            continue
        for filing_path in sorted(company_dir.glob("*.json")):
            with open(filing_path) as f:
                data = json.load(f)
            for i, chunk in enumerate(data["chunks"]):
                if len(chunk.split()) < 50:
                    continue
                chunks.append(chunk)
                metadata.append({
                    "ticker": data["ticker"],
                    "date": data["date"],
                    "chunk_index": i,
                })
    return chunks, metadata


def run_topic_modeling() -> None:
    chunks, metadata = load_chunks_with_metadata()
    print(f"Running BERTopic on {len(chunks)} chunks...")

    topic_model = BERTopic(
        embedding_model=model,
        min_topic_size=5,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(chunks)

    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics")
    print(topic_info.head(15).to_string())

    topic_model.save(str(TOPICS_DIR / "bertopic_model"))
    with open(TOPICS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(TOPICS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(TOPICS_DIR / "topics.pkl", "wb") as f:
        pickle.dump(topics, f)

    results = []
    for i, (topic, chunk) in enumerate(zip(topics, chunks)):
        results.append({
            "topic": int(topic),
            "ticker": metadata[i]["ticker"],
            "date": metadata[i]["date"],
            "text_preview": chunk[:200],
        })

    with open(TOPICS_DIR / "assignments.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved topic assignments to {TOPICS_DIR / 'assignments.json'}")


if __name__ == "__main__":
    run_topic_modeling()