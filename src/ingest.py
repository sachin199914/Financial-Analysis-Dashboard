"""Download and catalog SEC filings from EDGAR.

The script intentionally uses only the Python standard library so Phase 2 can run
before project dependencies are installed.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


COMPANIES = {
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "INTC": "Intel",
}


TARGET_FILING_TYPE = "10-K"
TARGET_YEARS = [2022, 2023, 2024]
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}"
ROOT = Path(__file__).resolve().parents[1]
FILINGS_DIR = ROOT / "data" / "filings"
SOURCES_CSV = ROOT / "data" / "sources.csv"

# SEC asks automated clients to identify themselves. Replace this with your
# email before submitting if desired.
USER_AGENT = "financial-analysis-dashboard sachin@example.com"


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def ticker_to_cik() -> dict[str, str]:
    companies = fetch_json(SEC_TICKERS_URL)
    lookup = {}
    for item in companies.values():
        ticker = item["ticker"].upper()
        lookup[ticker] = f"{int(item['cik_str']):010d}"
    return lookup


def recent_filings(cik: str) -> list[dict[str, str]]:
    submissions = fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
    recent = submissions["filings"]["recent"]
    keys = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "primaryDocument",
    ]
    return [dict(zip(keys, values)) for values in zip(*(recent[key] for key in keys))]


def selected_10ks(filings: list[dict[str, str]]) -> list[dict[str, str]]:
    matches = [
        filing
        for filing in filings
        if filing["form"] == TARGET_FILING_TYPE
        and filing["reportDate"]
        and int(filing["reportDate"][:4]) in TARGET_YEARS
    ]
    matches.sort(key=lambda filing: filing["reportDate"])
    return matches


def filing_url(cik: str, filing: dict[str, str]) -> str:
    accession_no_dashes = filing["accessionNumber"].replace("-", "")
    return SEC_ARCHIVES_URL.format(
        cik_int=int(cik),
        accession_no_dashes=accession_no_dashes,
        primary_doc=filing["primaryDocument"],
    )


def write_sources(rows: list[dict[str, str]]) -> None:
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "company",
        "ticker",
        "filing_type",
        "fiscal_year",
        "filing_date",
        "accession_number",
        "sec_url",
        "local_path",
    ]
    with SOURCES_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    cik_lookup = ticker_to_cik()
    rows = []

    for ticker, company in COMPANIES.items():
        cik = cik_lookup[ticker]
        filings = selected_10ks(recent_filings(cik))
        if len(filings) != len(TARGET_YEARS):
            print(f"Warning: expected {len(TARGET_YEARS)} filings for {ticker}, found {len(filings)}")

        for filing in filings:
            fiscal_year = filing["reportDate"][:4]
            url = filing_url(cik, filing)
            suffix = Path(filing["primaryDocument"]).suffix or ".html"
            local_path = FILINGS_DIR / f"{ticker}_{fiscal_year}_{TARGET_FILING_TYPE.replace('-', '')}{suffix}"

            if not local_path.exists():
                print(f"Downloading {ticker} {fiscal_year} {TARGET_FILING_TYPE}")
                download_file(url, local_path)
                time.sleep(0.2)
            else:
                print(f"Already exists: {local_path.name}")

            rows.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "filing_type": TARGET_FILING_TYPE,
                    "fiscal_year": fiscal_year,
                    "filing_date": filing["filingDate"],
                    "accession_number": filing["accessionNumber"],
                    "sec_url": url,
                    "local_path": str(local_path.relative_to(ROOT)),
                }
            )

    write_sources(rows)
    print(f"Wrote {len(rows)} source rows to {SOURCES_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
