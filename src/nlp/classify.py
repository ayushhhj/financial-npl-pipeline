from transformers import pipeline
from pathlib import Path
import json

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

LOCATION_LABELS = [
    "supplier or manufacturer location",
    "sales market or customer base",
    "regulatory or legal environment",
    "geopolitical or security risk",
    "office facility or headquarters",
]

PERSON_LABELS = [
    "executive or management role",
    "key person risk or dependency",
    "legal or regulatory party",
    "board member or director",
]

ORG_LABELS = [
    "competitor or rival company",
    "business partner or supplier",
    "regulatory body or government agency",
    "acquisition target or acquired company",
    "customer or distribution partner",
]

MONEY_LABELS = [
    "revenue or sales figure",
    "capital expenditure or investment",
    "legal settlement or fine",
    "debt or liability",
    "cash or liquidity position",
]

PRODUCT_LABELS = [
    "product launch or new release",
    "revenue growth or market success",
    "product risk or recall or decline",
    "competitive threat or substitute",
]

LABEL_CONFIGS = {
    "GPE": LOCATION_LABELS,
    "PERSON": PERSON_LABELS,
    "ORG": ORG_LABELS,
    "MONEY": MONEY_LABELS,
    "PRODUCT": PRODUCT_LABELS,
}

LABEL_SHORTNAMES = {
    "supplier or manufacturer location": "supplier",
    "sales market or customer base": "market",
    "regulatory or legal environment": "regulatory",
    "geopolitical or security risk": "geopolitical_risk",
    "office facility or headquarters": "facility",
    "executive or management role": "executive",
    "key person risk or dependency": "key_person_risk",
    "legal or regulatory party": "legal_party",
    "board member or director": "board_member",
    "competitor or rival company": "competitor",
    "business partner or supplier": "partner",
    "regulatory body or government agency": "regulator",
    "acquisition target or acquired company": "acquisition",
    "customer or distribution partner": "customer",
    "revenue or sales figure": "revenue",
    "capital expenditure or investment": "capex",
    "legal settlement or fine": "fine",
    "debt or liability": "debt",
    "cash or liquidity position": "cash",
    "product launch or new release": "launch",
    "revenue growth or market success": "growth",
    "product risk or recall or decline": "risk",
    "competitive threat or substitute": "competitive_threat",
}


def classify_entities_batch(
    entities: list[dict],
    label_type: str,
    batch_size: int = 8,
) -> list[str]:
    candidate_labels = LABEL_CONFIGS.get(label_type)
    if not candidate_labels:
        return ["general"] * len(entities)

    contexts = [ent.get("context", "") for ent in entities]
    categories = []

    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        valid_batch = [c if c and len(c.strip()) > 10 else "no context available" for c in batch]
        results = classifier(valid_batch, candidate_labels, multi_label=False)
        for result in results:
            top_label = result["labels"][0]
            categories.append(LABEL_SHORTNAMES.get(top_label, "general"))
        print(f"    [{label_type}] {min(i + batch_size, len(contexts))}/{len(contexts)}")

    return categories