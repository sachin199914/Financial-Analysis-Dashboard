# Comparative Financial Analysis Dashboard

Take-home project for CustomerInsights.AI: a Streamlit dashboard that compares public US companies using SEC filings, cited extracted metrics, derived financial ratios, and RAG-based Q&A.

## Scope

Chosen sector: semiconductors.

Companies:

- NVIDIA (`NVDA`)
- AMD (`AMD`)
- Intel (`INTC`)

Initial filing target:

- 10-K filings across three fiscal years per company
- Expand to 10-Qs only after the core dashboard, citations, and evaluation are working

## Planned Features

- SEC filing ingestion from public EDGAR documents
- Extracted financial metrics with source traceability
- Derived metrics such as revenue growth, operating margin, net margin, and leverage
- Comparative charts across companies and years
- Natural-language Q&A with cited source snippets
- Refusal behavior for unanswerable or insufficiently sourced questions
- Evaluation set covering correctness, citation accuracy, and hallucination rate

## Project Structure

```text
.
├── app.py
├── data/
│   ├── extracted_metrics.csv
│   ├── filings/
│   │   └── .gitkeep
│   └── sources.csv
├── eval/
│   ├── questions.csv
│   └── results.csv
├── src/
│   ├── __init__.py
│   ├── extract_metrics.py
│   ├── ingest.py
│   ├── metrics.py
│   └── rag.py
├── README.md
├── WRITEUP.md
└── requirements.txt
```

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Document Sources

Source metadata is tracked in `data/sources.csv`.

Current filing set:

- NVIDIA 10-K filings for fiscal years 2022, 2023, and 2024
- AMD 10-K filings for fiscal years 2022, 2023, and 2024
- Intel 10-K filings for fiscal years 2022, 2023, and 2024

Each row records the company, ticker, filing type, fiscal year, filing date, accession number, SEC URL, and local path.

## Current Status

Phase 1 complete: scope and scaffold.

Phase 2 complete: SEC filings downloaded and cataloged.
