import os
import json
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "JPM": "0000019617",
    "BAC": "0000070858",
    "GS": "0000886982",
    "NVDA": "0001045810",
    "META": "0001326801",
}


def get_filings(cik: str, form_type: str = "8-K", limit: int = 5) -> list[dict]:
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == form_type and len(results) < limit:
            results.append({
                "accession": filings["accessionNumber"][i],
                "date": filings["filingDate"][i],
                "form": form,
                "cik": cik,
            })

    return results


def fetch_filing_text(cik: str, accession: str) -> str | None:
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    accession_clean = accession.replace("-", "")
    cik_stripped = str(int(cik))
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession_clean}/{accession}-index.htm"

    time.sleep(0.15)
    response = requests.get(index_url, headers=HEADERS)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")
    txt_url = None

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 3:
            filename = cells[2].text.strip()
            if filename.endswith(".txt") and accession.replace("-", "") in filename.replace("-", ""):
                link = cells[2].find("a")
                if link:
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://www.sec.gov" + href
                    txt_url = href
                    break

    if not txt_url:
        txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession_clean}/{accession}.txt"

    time.sleep(0.15)
    doc_response = requests.get(txt_url, headers=HEADERS)
    if doc_response.status_code != 200:
        return None

    soup = BeautifulSoup(doc_response.content, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    return text if text.strip() else None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


def validate_filing(text: str, min_length: int = 100) -> tuple[bool, str]:
    if not text:
        return False, "empty text"
    if len(text.split()) < min_length:
        return False, f"text too short ({len(text.split())} words)"
    if not any(c.isalpha() for c in text):
        return False, "no alphabetic characters"
    return True, "ok"


def save_filing(ticker: str, accession: str, date: str, chunks: list[str]) -> Path:
    company_dir = RAW_DATA_DIR / ticker
    company_dir.mkdir(exist_ok=True)

    filename = f"{date}_{accession.replace('-', '')}.json"
    filepath = company_dir / filename

    data = {
        "ticker": ticker,
        "accession": accession,
        "date": date,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def run_ingestion(form_type: str = "8-K", limit: int = 5) -> dict:
    results = {"success": [], "failed": [], "skipped": []}

    for ticker, cik in COMPANIES.items():
        print(f"\nFetching {ticker}...")

        try:
            filings = get_filings(cik, form_type, limit)
            print(f"  Found {len(filings)} {form_type} filings")

            for filing in filings:
                accession = filing["accession"]
                date = filing["date"]

                print(f"  Fetching {accession} ({date})...", end=" ")
                text = fetch_filing_text(cik, accession)

                valid, reason = validate_filing(text)
                if not valid:
                    print(f"skipped ({reason})")
                    results["skipped"].append({"ticker": ticker, "accession": accession, "reason": reason})
                    continue

                chunks = chunk_text(text)
                filepath = save_filing(ticker, accession, date, chunks)
                print(f"saved {len(chunks)} chunks -> {filepath.name}")
                results["success"].append({"ticker": ticker, "accession": accession, "chunks": len(chunks)})

        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append({"ticker": ticker, "error": str(e)})

    return results


if __name__ == "__main__":
    results = run_ingestion()
    print(f"\n--- Summary ---")
    print(f"Success: {len(results['success'])} filings")
    print(f"Skipped: {len(results['skipped'])} filings")
    print(f"Failed:  {len(results['failed'])} companies")