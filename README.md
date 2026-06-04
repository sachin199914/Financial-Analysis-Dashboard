# Comparative Financial Analysis Dashboard

Take-home project for CustomerInsights.AI: a Streamlit dashboard that compares public US semiconductor companies using SEC Form 10-K filings, cited extracted metrics, derived financial ratios, and RAG-based Q&A.

## Scope

Chosen sector: semiconductors.

Companies:
- NVIDIA (`NVDA`)
- AMD (`AMD`)
- Intel (`INTC`)

Filings:
- Form 10-K filings across three fiscal years per company (2022, 2023, and 2024)
- 9 total filings cataloged and processed.

---

## Project Structure

```text
.
├── app.py                     # Streamlit dashboard interface
├── data/
│   ├── derived_metrics.csv    # Derived financial ratios calculated in Phase 3
│   ├── extracted_metrics.csv  # Audited raw XBRL metrics extracted in Phase 3
│   ├── extraction_issues.csv # Ledger of missing/complex tags found in filings
│   ├── sources.csv            # Ledger matching accessions, dates, URLs, and local paths
│   ├── filings/               # Cloned raw SEC filing XHTML files
│   └── vectorstore/           # Chroma vector database folder (created in Phase 5)
├── eval/
│   ├── questions.csv          # Evaluation labeled test set
│   └── results.csv            # Q&A test result logs & correctness metrics
├── src/
│   ├── __init__.py
│   ├── ingest.py              # Download filings & catalog metadata from SEC EDGAR
│   ├── extract_metrics.py     # Parse facts & clean quotes from XHTML XBRL statements
│   ├── metrics.py             # Financial margin, FCF, and leverage math formulas
│   ├── rag.py                 # Retrieval and Gemini ChatPromptTemplate pipeline
│   ├── rag_ingest.py          # BeautifulSoup parser and local Chroma indexer
│   └── evaluate.py            # Batch query evaluator and grader
├── README.md                  # This file
├── WRITEUP.md                 # Design decisions and evaluation analysis writeup
└── requirements.txt           # Project library requirements
```

---

## Running Locally

### 1. Initialize Virtual Environment & Dependencies
Create a virtual environment and install the required libraries:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Get a Google Gemini API Key from **[Google AI Studio](https://aistudio.google.com/)**. Create a `.env` file in the project root and add your key:
```env
GEMINI_API_KEY=your_copied_api_key_here
```

### 3. Parse & Index Filings into the Vector Store
Ingest the 10-K XHTML filings into the local vector database using local sentence embeddings:
```bash
python3 src/rag_ingest.py
```
This builds the index under `data/vectorstore/`.

### 4. Start the Dashboard UI
Run the Streamlit web application:
```bash
streamlit run app.py
```
Access the dashboard at **[http://localhost:8501](http://localhost:8501)**.

### 5. Run Q&A Evaluations
To execute the automated evaluation test suite:
```bash
python3 src/evaluate.py
```
This updates `eval/questions.csv` and `eval/results.csv` and displays a statistics summary in the console.

---

## Dashboard Tab Breakdown
1. **Overview & Scope:** Visualizes high-level stats, source catalog metadata, clickable links to EDGAR URLs, and known extraction limitations/issues.
2. **Comparative Analysis:** Interactive Plotly charts for revenues, margin ratios (gross, operating, net), free cash flow volume, YoY growth rate, and leverage ratios.
3. **Metric Evidence & Traceability:** Select any company, year, and metric. The dashboard shows the value, XBRL schemas/fact IDs, and highlights the **exact text quote from the Form 10-K** where it was drawn. For calculated metrics, it recursively maps and displays the source quotes and values for **all underlying inputs**.
4. **Automated Insights:** Rule-based peer benchmarking that highlights top/bottom performers and yields a clean peer performance matrix.
5. **RAG Chat Assistant:** Auditable chat interface that uses metadata-prefiltering to search the vector database and generate answers with document citations and source quotes.

---

## Current Status

- **Phase 1 complete:** Scope defined and project scaffolding structured.
- **Phase 2 complete:** SEC Form 10-K files downloaded and cataloged in `data/sources.csv`.
- **Phase 3 complete:** Financial figures parsed from inline XBRL, and raw/derived metrics created.
- **Phase 4 complete:** Dashboard layout, Plotly comparative charts, auto-insights, and metric audit trails implemented.
- **Phase 5 complete:** BeautifulSoup HTML chunking, local Chroma DB index built, and Gemini Q&A chain integrated with strict refusal and citation logic.
- **Phase 6 complete:** 15-question evaluation set defined, results logged, and the [WRITEUP.md](file:///Users/sachin/Documents/Codex/financialProject/WRITEUP.md) finalized.
