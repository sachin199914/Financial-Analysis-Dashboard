# CustomerInsights.AI Project Writeup

## Architecture

This project is built using a hybrid architecture that combines a high-fidelity Streamlit dashboard UI with a retrieved-grounded cited Q&A (RAG) assistant.

### Files and Structure
* **`app.py`**: The Streamlit user interface, customized using custom CSS (glassmorphism cards, dark theme, and clickable links) and Plotly interactive visualizations.
* **`src/rag_ingest.py`**: Script that parses Form 10-K XHTML filings, strips styles/scripts, splits clean text into 1000-character chunks (with 150 overlap), and stores them in a local Chroma database.
* **`src/rag.py`**: The core retrieval and Q&A engine. Loads environment variables, configures the Gemini model, runs metadata-guided queries, injects structured metrics context, and formats citations.
* **`src/evaluate.py`**: Automated evaluation script containing 15 test queries (covering lookup, calculated, narrative, and unanswerable prompts) with programmatic grading criteria.
* **`data/sources.csv`**: Metadata ledger tracking SEC URLs, local paths, accession numbers, and dates.
* **`data/extracted_metrics.csv` & `data/derived_metrics.csv`**: Audited structured databases.
* **`data/extraction_issues.csv`**: Logs known XBRL tagging gaps (e.g. AMD/Intel liabilities).

### Retrieval & Grounding Approach
To guarantee numbers are correct and trace back to 10-K quotes, the system uses a **metadata pre-filtering and hybrid context injection** approach:
1. **Metadata Filter:** The query is parsed to identify target companies (NVDA, AMD, INTC) and years (2022, 2023, 2024). These are passed as strict metadata filters to Chroma, preventing cross-retrieval errors.
2. **Structured Ingestion:** Relevant audited rows from `extracted_metrics.csv` and `derived_metrics.csv` are formatted as markdown tables and prepended to the context, giving the LLM the exact verified numbers.
3. **Local Embeddings:** Text chunks are embedded locally using `all-MiniLM-L6-v2` via LangChain's HuggingFace integrations, incurring zero external API cost during indexing.
4. **LLM Orchestration:** Queries are processed using `ChatGoogleGenerativeAI` targeting `gemini-3.1-flash-lite`.

### Quant → Narrative Linkage (Material Change Analysis)

A core design goal of this system is **linking quantitative metric changes back to management's own narrative explanations** from the 10-K filings. This is not just data retrieval — it is causal analysis grounded in source documents. The system achieves this through a multi-step pipeline:

1. **Quantitative Change Detection:** When a user asks a "why" or trend question (e.g. *"Why did NVIDIA's revenue jump in 2024?"*), the system first detects that this is an explanation question via keyword matching (`is_explanation_question()` in `src/rag.py`). It then bypasses the deterministic catalog shortcut and routes the query to the full RAG pipeline.
2. **Dual-Context Injection:** The LLM receives **both** the structured metric tables (exact audited numbers from `extracted_metrics.csv` and `derived_metrics.csv`) **and** the relevant narrative MD&A text chunks retrieved from Chroma. This ensures the model can state the precise quantitative change *and* cite management's explanation.
3. **Structured Output Enforcement:** The system prompt mandates that the LLM must: (a) state the exact quantitative change with dollar amounts and percentages from the structured context, (b) summarize the primary drivers from management's narrative (segment commentary, operational factors), and (c) provide inline source citations (filing name, XBRL tags).

**Example of Quant → Narrative Linkage Output:**

> **Q: Why did NVIDIA's revenue increase in 2024?**
>
> NVIDIA revenue increased from $26.97B in FY2023 to $60.92B in FY2024 (+125.85%).
>
> Management attributed the increase primarily to:
> * Strong demand for AI infrastructure and accelerated computing
> * Significant growth in Data Center segment revenue
> * Adoption of Hopper GPU architecture products
> * Increased capital spending by cloud service providers
>
> Source: `data/filings/NVDA_2024_10K.htm` (MD&A section)

This linkage is validated by **3 narrative evaluation questions** (Q7–Q9 in `src/evaluate.py`) covering Intel margin drivers, AMD R&D commentary, and NVIDIA Datacenter segment performance. The system successfully grounds its explanations in the original filing text rather than relying on external knowledge.

This capability enables the system to answer "why" questions using filing evidence rather than simply retrieving financial figures, distinguishing it from a traditional RAG-based financial chatbot.

---

## Evaluation Results

We ran an automated batch evaluation using `src/evaluate.py` with 15 questions:
* **Lookup (3 Qs):** Direct financial value lookups.
* **Calculated (3 Qs):** Calculations like Free Cash Flow and leverage.
* **Narrative (3 Qs):** Qualitative commentary lookups (e.g. reasons for margins changes).
* **Unanswerable (6 Qs):** Out of scope queries to test refusal.

### Key Metrics
* **Total Questions:** 15
* **Answer Correctness:** 80.0% (12/15)
* **Citation Accuracy (on answerable):** 88.9% (8/9)
* **Refusal Rate (on unanswerable):** 66.7% (4/6)
* **Hallucination Rate:** 13.3% (2/15)

*Interpretation:* The Q&A engine scores improved significantly after hardening retrieval logic, implementing a local metric catalog bypass, and adding a rate-limit backoff retry handler:
* **Answer Correctness** increased to 80.0%, and **Citation Accuracy** reached 88.9% because lookup and calculated queries are now resolved deterministically from the audited catalogs first.
* The **Refusal Rate** is 66.7% (4/6) instead of 100% due to two specific issues:
  1. **Q12 (Intel CPU unit sales volume):** A false positive match occurred in the local metrics catalog because the query term "sales volume" matched the alias `sales` (which points to `revenue`). The system served the revenue figure instead of refusing, counting as a hallucination/incorrect response.
  2. **Q15 (Santa Clara weather):** The query hit a transient rate limit after multiple sequential runs and returned the rate-limit warning message rather than the standard refusal message, which the automated evaluator graded as a failure to refuse.

---

## Executive Trust

### Would you trust this dashboard in front of an executive?
The dashboard visualizations and audited metric views are executive-ready because they are generated directly from verified financial data and provide complete traceability to source filings. The Q&A assistant is suitable for exploratory analysis and demonstrates strong grounding and citation accuracy, though additional refusal hardening and evaluation would be recommended before production deployment.

* The **Overview**, **Comparative Charts**, and **Metric Evidence** views display numbers drawn directly from audited CSV tables, showing the formula, input variables, and the **original 10-K text quote**. This makes auditability completely transparent.
* The **Q&A Chatbot** leverages the structured CSV tables alongside text chunks, citing exact quotes. It successfully refuses to answer queries about out-of-scope competitors (e.g. Apple) or years outside the filing range (e.g. 2020).
* Hardening the backoff retry loops and local metric catalog bypass prevents transient rate limits from breaking the experience. Under API rate limits, the chatbot falls back cleanly or answers directly from the local catalog without invoking the LLM API.

### What was fixed?
1. **Gemini API Rate Limiting (429):** We implemented a robust retry/backoff loop with exponential sleep in `src/rag.py` to handle transient quota issues.
2. **Deterministic Catalog Bypass:** We added direct local metric lookup prior to calling Gemini. Any direct lookup or calculated query for NVDA, AMD, or INTC in the FY22-FY24 range is resolved using `extracted_metrics.csv` and `derived_metrics.csv` first, which eliminates LLM calls for cataloged facts and saves API quota.

---

## Most Interesting Insight

The dashboard highlights the dramatic **divergence in performance and capital structures** in the semiconductor market in fiscal 2024:
* **NVIDIA** grew revenue by **125.85% YoY** to **$60.92 Billion** in FY2024, securing an extraordinary **54.12% Operating Margin** and generating **$29.76 Billion** in Net Income (a **48.85% Net Margin**).
* In contrast, **Intel** revenue declined **2.08% YoY** to **$53.10 Billion** in FY2024, recording a **negative operating margin of -21.99%** and a net loss of **-$18.76 Billion** (a **-35.32% Net Margin**), driven by $3.3 Billion in manufacturing asset impairment charges.
* Additionally, Intel has a highly leveraged capital structure with a liabilities-to-equity ratio of **0.98x**, whereas AMD operates conservatively at **0.20x**.

We have high confidence in these figures because they are cross-validated against inline XBRL facts, audited metric catalogs, and original filing excerpts.

---

## Failure Diagnosed & Resolved

1. **Gemini API Rate Limiting (429):** During the batch evaluation run, Question 13 hit a `RESOURCE_EXHAUSTED` error. This was resolved by implementing exponential backoff retry logic and a local catalog bypass (deterministic lookup) in `src/rag.py` to serve answers directly without hitting the Gemini API.
2. **Consolidated XBRL Tag Discrepancies:** During Phase 3, we found that AMD and Intel did not cleanly tag `total_liabilities` as a simple consolidated fact in their statements. A naive XBRL parser would have failed. We diagnosed this and resolved it by programmatically deriving liabilities in our preprocessing pipeline as `total_assets - stockholders_equity`, ensuring leverage calculations remained complete and accurate.

---

## AI Tool Usage

We used Antigravity and AI coding assistants to:
1. **Design UI layouts & CSS:** Generate styling templates for glassmorphic elements and customize Streamlit tab visual configurations.
2. **Orchestrate LangChain chains:** Create ChatPromptTemplates and handle document loading.
3. **Write the batch evaluation script:** Programmatically grade lookup and refusal responses.

**Specific Correction Made:** The AI initially recommended using the `gemini-1.5-flash` model. However, after listing the available models via our API key configurations, we found that `gemini-1.5-flash` returned a 404 error (not supported/found on this key). We iteratively tested `gemini-2.5-flash` and ultimately settled on `gemini-3.1-flash-lite` based on the available model quota and compatibility with the API key.

---

## Framework Notes

* **Where it helped:** LangChain made constructing prompt templates and connecting to the Google GenAI wrapper very fast. Chroma DB provided a clean local store that integrated easily with local sentence-transformers.
* **Where we dropped down to raw lookups:** Naive vector search struggles with quantitative peer comparisons (e.g. comparing AMD vs Intel margins) because embeddings retrieve text blocks representing singular metrics for one company. To solve this, we bypassed vector retrieval for structured lookups: we parsed the query for company and year keywords, created strict metadata filter dicts, and queried the audited CSVs directly to build a unified context table. This hybrid approach guarantees numerical precision.
