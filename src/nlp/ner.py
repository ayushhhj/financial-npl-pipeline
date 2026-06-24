import json
import spacy
from pathlib import Path
from collections import defaultdict
from src.nlp.classify import classify_entities_batch, LABEL_CONFIGS

nlp = spacy.load("en_core_web_trf")

FINANCIAL_LABELS = {"ORG", "PERSON", "GPE", "MONEY", "DATE", "LAW", "PRODUCT"}

def extract_entities(text: str) -> list[dict]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in FINANCIAL_LABELS:
            context_start = max(0, ent.start_char - 150)
            context_end = min(len(text), ent.end_char + 150)
            context = text[context_start:context_end]
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "context": context,
            })
    return entities

def process_filing(filepath: Path) -> dict:
    with open(filepath) as f:
        data = json.load(f)

    all_entities = []
    for chunk in data["chunks"]:
        entities = extract_entities(chunk)
        all_entities.extend(entities)

    # classify all entity types that have context
    classifiable_types = list(LABEL_CONFIGS.keys())
    for label_type in classifiable_types:
        typed_entities = [
            ent for ent in all_entities
            if ent["label"] == label_type and ent.get("context")
        ]
        if typed_entities:
            print(f"  Classifying {len(typed_entities)} {label_type} entities...")
            categories = classify_entities_batch(typed_entities, label_type)
            for ent, category in zip(typed_entities, categories):
                ent["category"] = category

    # aggregate counts
    entity_counts = defaultdict(lambda: defaultdict(int))
    for ent in all_entities:
        entity_counts[ent["label"]][ent["text"]] += 1

    # aggregate categories per entity type and text
    entity_categories = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for ent in all_entities:
        if ent.get("category"):
            entity_categories[ent["label"]][ent["text"]][ent["category"]] += 1

    return {
        "ticker": data["ticker"],
        "accession": data["accession"],
        "date": data["date"],
        "entity_counts": {
            label: dict(sorted(texts.items(), key=lambda x: x[1], reverse=True)[:20])
            for label, texts in entity_counts.items()
        },
        "entity_categories": {
            label: {
                text: dict(cats)
                for text, cats in sorted(
                    texts.items(),
                    key=lambda x: sum(x[1].values()),
                    reverse=True,
                )[:20]
            }
            for label, texts in entity_categories.items()
        },
        "total_entities": len(all_entities),
    }

def run_ner(raw_dir: Path = Path("data/raw"), output_dir: Path = Path("data/ner")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for company_dir in sorted(raw_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        ticker = company_dir.name
        print(f"\nProcessing {ticker}...")

        for filing_path in sorted(company_dir.glob("*.json")):
            print(f"  {filing_path.name}...", end=" ")
            try:
                result = process_filing(filing_path)
                out_path = output_dir / ticker / filing_path.name
                out_path.parent.mkdir(exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"found {result['total_entities']} entities")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    run_ner()