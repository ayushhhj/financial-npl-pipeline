import json
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

NER_DIR = Path("data/ner")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password"),
    ),
)

LOCATION_NORMALIZATIONS = {
    "us": "United States",
    "u.s.": "United States",
    "the united states": "United States",
    "united states": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "the u.s.": "United States",
    "uk": "United Kingdom",
    "the u.k.": "United Kingdom",
    "prc": "China",
    "the people's republic of china": "China",
}

def normalize_location(name: str) -> str:
    name_clean = name.strip().lower()
    if name_clean in LOCATION_NORMALIZATIONS:
        return LOCATION_NORMALIZATIONS[name_clean]
    return name.strip()

def normalize_person(name: str) -> str:
    return " ".join(word.capitalize() for word in name.strip().split())

def clear_graph(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared")


def create_constraints(session):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    ]
    for constraint in constraints:
        session.run(constraint)
    print("Constraints created")


TICKER_ALIASES = {
    "AAPL": ["Apple", "Apple Inc.", "Apple Inc", "APPLE INC", "APPLE COMPUTER INC"],
    "MSFT": ["Microsoft", "Microsoft Corporation", "Microsoft Corp"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc.", "Google LLC"],
    "AMZN": ["Amazon", "Amazon.com", "Amazon.com Inc.", "AMAZON COM INC"],
    "TSLA": ["Tesla", "Tesla Inc.", "Tesla Motors"],
    "JPM": ["JPMorgan", "JPMorgan Chase", "JP Morgan", "JPMorgan Chase & Co"],
    "BAC": ["Bank of America", "BofA", "Bank of America Corporation"],
    "GS": ["Goldman Sachs", "Goldman", "Goldman Sachs Group"],
    "NVDA": ["NVIDIA", "Nvidia", "NVIDIA Corporation"],
    "META": ["Meta", "Facebook", "Meta Platforms", "Meta Platforms Inc."],
}

def resolve_company(text: str) -> str | None:
    text_clean = text.strip()
    for ticker, aliases in TICKER_ALIASES.items():
        if text_clean == ticker:
            return ticker
        for alias in aliases:
            if fuzz.ratio(text_clean.lower(), alias.lower()) > 88:
                return ticker
    return None


def create_company_nodes(session):
    company_info = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corporation",
        "GS": "The Goldman Sachs Group Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
    }
    for ticker, name in company_info.items():
        session.run(
            """
            MERGE (c:Company {id: $ticker})
            SET c.name = $name, c.ticker = $ticker
            """,
            ticker=ticker,
            name=name,
        )
    print(f"Created {len(company_info)} company nodes")


def load_filing(session, ner_path: Path):
    with open(ner_path) as f:
        data = json.load(f)

    ticker = data["ticker"]
    accession = data["accession"]
    date = data["date"]
    entity_counts = data.get("entity_counts", {})
    entity_categories = data.get("entity_categories", {})

    # create filing node and link to company
    session.run(
        """
        MERGE (f:Filing {accession: $accession})
        SET f.date = $date, f.ticker = $ticker
        WITH f
        MATCH (c:Company {id: $ticker})
        MERGE (c)-[:FILED]->(f)
        """,
        accession=accession,
        date=date,
        ticker=ticker,
    )

    # load persons with count and dominant role
    for person, count in entity_counts.get("PERSON", {}).items():
        person = normalize_person(person)
        if len(person) < 4 or len(person.split()) < 2:
            continue
        cats = entity_categories.get("PERSON", {}).get(person, {})
        dominant_role = max(cats, key=cats.get) if cats else "unknown"
        session.run(
            """
            MERGE (p:Person {name: $name})
            WITH p
            MATCH (f:Filing {accession: $accession})
            MERGE (p)-[r:MENTIONED_IN]->(f)
            SET r.count = $count
            WITH p
            MATCH (c:Company {id: $ticker})
            MERGE (p)-[r2:ASSOCIATED_WITH]->(c)
            ON CREATE SET r2.count = $count, r2.dominant_role = $role
            ON MATCH SET r2.count = r2.count + $count, r2.dominant_role = $role
            """,
            name=person,
            accession=accession,
            ticker=ticker,
            count=count,
            role=dominant_role,
        )

    # load org co-mentions with relationship type
    for org, count in entity_counts.get("ORG", {}).items():
        resolved = resolve_company(org)
        if resolved and resolved != ticker:
            cats = entity_categories.get("ORG", {}).get(org, {})
            dominant_rel = max(cats, key=cats.get) if cats else "unknown"
            session.run(
                """
                MATCH (c1:Company {id: $ticker})
                MATCH (c2:Company {id: $resolved})
                MERGE (c1)-[r:CO_MENTIONED_WITH]->(c2)
                ON CREATE SET r.count = $count, r.relationship_type = $rel_type
                ON MATCH SET r.count = r.count + $count, r.relationship_type = $rel_type
                """,
                ticker=ticker,
                resolved=resolved,
                count=count,
                rel_type=dominant_rel,
            )

    # load locations with category
    for location, count in entity_counts.get("GPE", {}).items():
        raw_location = location.strip()
        cats = entity_categories.get("GPE", {}).get(raw_location, {})
        location = normalize_location(raw_location)
        if len(location) < 2:
            continue
    dominant_category = max(cats, key=cats.get) if cats else "general"
    session.run(
        """
        MERGE (l:Location {name: $name})
        WITH l
        MATCH (f:Filing {accession: $accession})
        MERGE (f)-[r:MENTIONS_LOCATION]->(l)
        SET r.count = $count,
            r.dominant_category = $category,
            r.category_counts = $cats
        """,
        name=location,
        accession=accession,
        count=count,
        category=dominant_category,
        cats=json.dumps(cats),
    )


def run_graph_loading():
    with driver.session() as session:
        clear_graph(session)
        create_constraints(session)
        create_company_nodes(session)

        total = 0
        for company_dir in sorted(NER_DIR.iterdir()):
            if not company_dir.is_dir():
                continue
            ticker = company_dir.name
            print(f"\nLoading {ticker}...")
            for ner_path in sorted(company_dir.glob("*.json")):
                load_filing(session, ner_path)
                total += 1
                print(f"  loaded {ner_path.name}")

        print(f"\nDone — loaded {total} filings into graph")

        print("\n--- Graph summary ---")
        for label in ["Company", "Person", "Location", "Filing"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            print(f"  {label}: {result.single()['count']}")

        print("\n--- Sample queries ---")

        print("\nCompanies with most mentioned persons:")
        result = session.run("""
            MATCH (p:Person)-[:ASSOCIATED_WITH]->(c:Company)
            RETURN c.ticker, count(p) as person_count
            ORDER BY person_count DESC LIMIT 5
        """)
        for r in result:
            print(f"  {r['c.ticker']}: {r['person_count']} persons")

        print("\nMost mentioned locations across all filings:")
        result = session.run("""
            MATCH (f:Filing)-[r:MENTIONS_LOCATION]->(l:Location)
            RETURN l.name, sum(r.count) as total
            ORDER BY total DESC LIMIT 10
        """)
        for r in result:
            print(f"  {r['l.name']}: {r['total']}")

        print("\nCompany co-mentions:")
        result = session.run("""
            MATCH (c1:Company)-[r:CO_MENTIONED_WITH]->(c2:Company)
            RETURN c1.ticker, c2.ticker, r.count
            ORDER BY r.count DESC LIMIT 10
        """)
        for r in result:
            print(f"  {r['c1.ticker']} -> {r['c2.ticker']}: {r['r.count']}")


if __name__ == "__main__":
    run_graph_loading()