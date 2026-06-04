"""Retrieval-Augmented Generation (RAG) Q&A system for SEC filings."""

import os
import pathlib
import re
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

VECTORSTORE_DIR = ROOT / "data" / "vectorstore"
EXTRACTED_CSV = ROOT / "data" / "extracted_metrics.csv"
DERIVED_CSV = ROOT / "data" / "derived_metrics.csv"
REFUSAL_MESSAGE = "I cannot answer this question because the provided filings and metric catalogs do not contain sufficient information."
TEMPORARY_UNAVAILABLE_MESSAGE = (
    "The filing Q&A service is temporarily unavailable because the LLM provider is rate-limited. "
    "Please retry in a minute, or ask a direct metric question that can be answered from the local metric catalog."
)
VECTORSTORE_NOT_READY_MESSAGE = (
    "The narrative filing index is not available in this environment, so I cannot retrieve management commentary. "
    "I can still answer from the verified metric catalog below."
)

METRIC_ALIASES = {
    "revenue": ["revenue", "sales"],
    "gross_profit": ["gross profit"],
    "operating_income": ["operating income", "income from operations"],
    "net_income": ["net income", "net loss"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "stockholders_equity": ["stockholders equity", "stockholders' equity", "shareholders equity", "shareholders' equity", "equity"],
    "operating_cash_flow": ["operating cash flow", "cash provided by operating activities"],
    "capital_expenditures": ["capital expenditures", "capex"],
    "gross_margin": ["gross margin"],
    "operating_margin": ["operating margin"],
    "net_margin": ["net margin"],
    "revenue_growth_yoy": ["revenue growth", "yoy revenue growth", "year-over-year revenue"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "liabilities_to_equity": ["liabilities-to-equity", "liabilities to equity", "leverage ratio"],
}

# Global indicators to check if database exists
embeddings = None
db = None
extracted_df = None
derived_df = None

def init_resources():
    global embeddings, db, extracted_df, derived_df
    if db is None:
        if VECTORSTORE_DIR.exists():
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=str(VECTORSTORE_DIR), embedding_function=embeddings)
        else:
            db = None
            
    if extracted_df is None and EXTRACTED_CSV.exists():
        extracted_df = pd.read_csv(EXTRACTED_CSV)
        
    if derived_df is None and DERIVED_CSV.exists():
        derived_df = pd.read_csv(DERIVED_CSV)

def detect_retrieval_filters(question: str) -> dict:
    """Parse the query to extract metadata filters (ticker, year) for Chroma."""
    filter_dict = {}
    query_lower = question.lower()
    
    # 1. Company matching
    if "nvidia" in query_lower or "nvda" in query_lower:
        filter_dict["ticker"] = "NVDA"
    elif "amd" in query_lower:
        filter_dict["ticker"] = "AMD"
    elif "intel" in query_lower or "intc" in query_lower:
        filter_dict["ticker"] = "INTC"
        
    # 2. Year matching
    if "2022" in query_lower:
        filter_dict["fiscal_year"] = 2022
    elif "2023" in query_lower:
        filter_dict["fiscal_year"] = 2023
    elif "2024" in query_lower:
        filter_dict["fiscal_year"] = 2024
        
    return filter_dict

def mentions_out_of_scope_company(question: str) -> bool:
    query_lower = question.lower()
    outside_company_terms = [
        "apple",
        "aapl",
        "google",
        "alphabet",
        "goog",
        "googl",
        "microsoft",
        "msft",
        "amazon",
        "amzn",
        "meta",
        "tesla",
        "tsla",
    ]
    if any(term in query_lower for term in outside_company_terms):
        return True
    return False

def mentions_out_of_scope_year(question: str) -> bool:
    years = [int(match) for match in re.findall(r"\b20\d{2}\b", question)]
    return any(year not in {2022, 2023, 2024} for year in years)

def detect_requested_metrics(question: str) -> list[str]:
    query_lower = question.lower().replace("/", " ")
    requested = []
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            requested.append(metric)
    return requested

def format_catalog_value(value: float, unit: str, metric: str) -> str:
    if unit == "ratio":
        return f"{value:.4f} ({value * 100:.2f}%)"
    if unit == "USD":
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1_000_000_000:
            return f"{sign}${abs_value / 1_000_000_000:.2f}B ({value / 1_000_000:.0f} million)"
        if abs_value >= 1_000_000:
            return f"{sign}${abs_value / 1_000_000:.2f}M"
    return f"{value:,.2f} {unit}"

def metric_input_rows(ticker: str, fiscal_year: int, formula: str) -> pd.DataFrame:
    if extracted_df is None:
        return pd.DataFrame()

    formula_lower = formula.lower()
    inputs = []
    for _, row in extracted_df[
        (extracted_df["ticker"] == ticker) & (extracted_df["fiscal_year"] == fiscal_year)
    ].iterrows():
        metric = row["metric"]
        if metric in formula_lower or metric.replace("_", "") in formula_lower.replace("_", ""):
            inputs.append(row)

    if "assets - equity" in formula_lower:
        for _, row in extracted_df[
            (extracted_df["ticker"] == ticker)
            & (extracted_df["fiscal_year"] == fiscal_year)
            & (extracted_df["metric"].isin(["total_assets", "stockholders_equity"]))
        ].iterrows():
            inputs.append(row)

    return pd.DataFrame(inputs).drop_duplicates() if inputs else pd.DataFrame()

def is_explanation_question(question: str) -> bool:
    query_lower = question.lower()
    explanation_keywords = [
        "why", "reason", "explain", "cause", "driver", "driven", "drive", 
        "attribute", "attribution", "commentary", "discuss", "discussion", 
        "narrative", "detail", "describe", "description"
    ]
    return any(keyword in query_lower for keyword in explanation_keywords)

def answer_from_metric_catalog(question: str, filters: dict) -> str | None:
    """Answer direct quantitative questions without calling the LLM."""
    init_resources()
    if extracted_df is None or derived_df is None:
        return None
    if "ticker" not in filters or "fiscal_year" not in filters:
        return None

    if is_explanation_question(question):
        return None

    requested_metrics = detect_requested_metrics(question)
    if not requested_metrics:
        return None

    ticker = filters["ticker"]
    fiscal_year = filters["fiscal_year"]
    answers = []

    for metric in requested_metrics:
        raw_matches = extracted_df[
            (extracted_df["ticker"] == ticker)
            & (extracted_df["fiscal_year"] == fiscal_year)
            & (extracted_df["metric"] == metric)
        ]
        derived_matches = derived_df[
            (derived_df["ticker"] == ticker)
            & (derived_df["fiscal_year"] == fiscal_year)
            & (derived_df["metric"] == metric)
        ]

        if not raw_matches.empty:
            row = raw_matches.iloc[0]
            value = format_catalog_value(float(row["value"]), row["unit"], metric)
            answers.append(
                f"- {row['company']} FY{fiscal_year} {metric.replace('_', ' ')}: {value}. "
                f"Source: {row['source_file']} ({row['source_section']}). Quote: \"{row['source_quote']}\""
            )
        elif not derived_matches.empty:
            row = derived_matches.iloc[0]
            value = format_catalog_value(float(row["value"]), row["unit"], metric)
            input_rows = metric_input_rows(ticker, fiscal_year, row["formula"])
            source_bits = []
            for _, input_row in input_rows.iterrows():
                source_bits.append(
                    f"{input_row['metric']} from {input_row['source_file']} "
                    f"({input_row['source_section']}): \"{input_row['source_quote']}\""
                )
            source_text = " Inputs: " + " | ".join(source_bits) if source_bits else ""
            answers.append(
                f"- {row['company']} FY{fiscal_year} {metric.replace('_', ' ')}: {value}. "
                f"Formula: {row['formula']}.{source_text}"
            )

    if not answers:
        return REFUSAL_MESSAGE

    return "Answered from the verified local metric catalog:\n" + "\n".join(answers)

def answer_explanation_from_metric_catalog(question: str, filters: dict) -> str | None:
    """Fallback for why/explain questions when narrative Chroma retrieval is unavailable."""
    init_resources()
    if extracted_df is None or derived_df is None:
        return None
    if "ticker" not in filters or "fiscal_year" not in filters:
        return None

    requested_metrics = detect_requested_metrics(question)
    if not requested_metrics:
        return None

    ticker = filters["ticker"]
    fiscal_year = filters["fiscal_year"]
    lines = [VECTORSTORE_NOT_READY_MESSAGE]

    for metric in requested_metrics:
        current = extracted_df[
            (extracted_df["ticker"] == ticker)
            & (extracted_df["fiscal_year"] == fiscal_year)
            & (extracted_df["metric"] == metric)
        ]
        prior = extracted_df[
            (extracted_df["ticker"] == ticker)
            & (extracted_df["fiscal_year"] == fiscal_year - 1)
            & (extracted_df["metric"] == metric)
        ]
        derived_current = derived_df[
            (derived_df["ticker"] == ticker)
            & (derived_df["fiscal_year"] == fiscal_year)
            & (derived_df["metric"] == metric)
        ]

        if not current.empty:
            row = current.iloc[0]
            value = format_catalog_value(float(row["value"]), row["unit"], metric)
            lines.append(
                f"- {row['company']} FY{fiscal_year} {metric.replace('_', ' ')} was {value}. "
                f"Source: {row['source_file']} ({row['source_section']}). Quote: \"{row['source_quote']}\""
            )
            if not prior.empty:
                prior_row = prior.iloc[0]
                current_value = float(row["value"])
                prior_value = float(prior_row["value"])
                if prior_value:
                    change = (current_value - prior_value) / prior_value
                    lines.append(
                        f"- The year-over-year change was {change:.4f} ({change * 100:.2f}%), "
                        f"from {format_catalog_value(prior_value, prior_row['unit'], metric)} in FY{fiscal_year - 1}."
                    )
        elif not derived_current.empty:
            row = derived_current.iloc[0]
            value = format_catalog_value(float(row["value"]), row["unit"], metric)
            lines.append(
                f"- {row['company']} FY{fiscal_year} {metric.replace('_', ' ')} was {value}. "
                f"Formula: {row['formula']}."
            )

    if len(lines) == 1:
        return None

    lines.append(
        "- To answer the qualitative 'why' with management commentary, include/build `data/vectorstore/` "
        "or run `python3 src/rag_ingest.py` before launching the app."
    )
    return "\n".join(lines)

def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ["429", "resource_exhausted", "rate limit", "quota"])

def get_structured_context(filter_dict: dict) -> str:
    """Retrieve corresponding structured metrics to ground the LLM's numbers."""
    init_resources()
    if extracted_df is None or derived_df is None:
        return ""
        
    # Build filter conditions
    ext_sub = extracted_df.copy()
    der_sub = derived_df.copy()
    
    if "ticker" in filter_dict:
        ext_sub = ext_sub[ext_sub["ticker"] == filter_dict["ticker"]]
        der_sub = der_sub[der_sub["ticker"] == filter_dict["ticker"]]
    if "fiscal_year" in filter_dict:
        ext_sub = ext_sub[ext_sub["fiscal_year"] == filter_dict["fiscal_year"]]
        der_sub = der_sub[der_sub["fiscal_year"] == filter_dict["fiscal_year"]]
        
    context_str = ""
    
    # Add structured extracted metrics
    if not ext_sub.empty:
        context_str += "### Structured Extracted Financial Facts:\n"
        for _, row in ext_sub.iterrows():
            context_str += f"- {row['company']} ({row['ticker']}) FY{row['fiscal_year']} {row['metric']}: {row['value']} {row['unit']} | Source Document: {row['source_file']} | Quote: \"{row['source_quote']}\"\n"
            
    # Add derived metrics
    if not der_sub.empty:
        context_str += "\n### Calculated Derived Metrics:\n"
        for _, row in der_sub.iterrows():
            context_str += f"- {row['company']} ({row['ticker']}) FY{row['fiscal_year']} {row['metric']}: {row['value']} {row['unit']} | Formula: {row['formula']}\n"
            
    return context_str

def answer_question(question: str) -> str:
    if mentions_out_of_scope_company(question) or mentions_out_of_scope_year(question):
        return REFUSAL_MESSAGE

    filters = detect_retrieval_filters(question)
    catalog_answer = answer_from_metric_catalog(question, filters)
    if catalog_answer is not None:
        return catalog_answer

    if is_explanation_question(question) and not VECTORSTORE_DIR.exists():
        fallback_answer = answer_explanation_from_metric_catalog(question, filters)
        if fallback_answer is not None:
            return fallback_answer

    # Check for API key (OpenAI or Gemini)
    api_key_openai = os.environ.get("OPENAI_API_KEY")
    api_key_gemini = os.environ.get("GEMINI_API_KEY")
    
    has_openai = api_key_openai and api_key_openai != "your_openai_api_key_here"
    has_gemini = api_key_gemini and api_key_gemini != "your_gemini_api_key_here"
    
    if not has_openai and not has_gemini:
        raise NotImplementedError(
            "API key is not configured. Please add either GEMINI_API_KEY or OPENAI_API_KEY to your .env file."
        )
        
    init_resources()
    if db is None:
        fallback_answer = answer_explanation_from_metric_catalog(question, filters)
        if fallback_answer is not None:
            return fallback_answer
        return (
            "The narrative filing index is not available in this environment. "
            "Please build it with `python3 src/rag_ingest.py`, or ask a direct metric question."
        )
        
    # Construct Chroma filter
    chroma_filter = None
    if len(filters) == 1:
        chroma_filter = filters
    elif len(filters) > 1:
        chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}
        
    # 2. Retrieve narrative text chunks from Chroma
    retrieved_docs = db.similarity_search(question, k=5, filter=chroma_filter)
    
    # Format chunks
    text_context = "### Narrative Chunks from SEC Filings:\n"
    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        text_context += f"--- Chunk {i+1} ---\n"
        text_context += f"Source: {meta['source_file']} | Company: {meta['company']} ({meta['ticker']}) | Year: {meta['fiscal_year']}\n"
        text_context += f"Text Content:\n{doc.page_content}\n\n"
        
    # 3. Retrieve structured metrics context
    structured_context = get_structured_context(filters)
    
    # Combine contexts
    full_context = f"{structured_context}\n\n{text_context}"
    
    # 4. Invoke LLM (Gemini)
    system_prompt = (
        "You are an executive-level financial analysis assistant. Your job is to answer questions about "
        "NVIDIA, AMD, and Intel Form 10-K filings based ONLY on the provided context (structured metrics tables "
        "and narrative text chunks).\n\n"
        "STRICT COMPLIANCE RULES:\n"
        "1. Ground every claim, percentage, and dollar amount directly in the provided context. If a number is mentioned in your response, "
        "it MUST be identical to the one in the context.\n"
        "2. Provide source file citations (e.g. data/filings/INTC_2024_10K.htm) and original quote references for all key claims.\n"
        "3. REFUSAL BEHAVIOR: If the question cannot be answered completely using ONLY the provided context, or if the question is "
        "unrelated/asks about metrics/years outside of the scope, you MUST respond exactly with: "
        "'I cannot answer this question because the provided filings and metric catalogs do not contain sufficient information.' "
        "Do not guess, speculate, or apply any external financial knowledge.\n"
        "4. Be concise, direct, and professional.\n"
        "5. For analytical, trend, or 'why' questions (e.g., explaining why a metric like revenue, margin, or capital structure changed), "
        "you must first state the quantitative change (citing the exact values and calculating/stating the percentage change if present in the context) "
        "and then summarize management's narrative explanation (primary drivers, segment commentary, and operational factors) "
        "in clean, structured bullet points with inline citations."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    try:
        # Load LLM dynamically
        if has_openai:
            from langchain_openai import ChatOpenAI
            model_name = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                api_key=api_key_openai,
                max_retries=0
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                temperature=0.0,
                google_api_key=api_key_gemini,
                max_retries=0
            )
        
        chain = prompt | llm
        last_error = None
        for attempt in range(3):
            try:
                response = chain.invoke({"context": full_context, "question": question})
                content = response.content
                # Handle structured content (list of parts) from newer Gemini models
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts) if text_parts else str(content)
                return content
            except Exception as exc:
                last_error = exc
                if not is_rate_limit_error(exc) or attempt == 2:
                    break
                time.sleep(2 ** attempt)

        if last_error and is_rate_limit_error(last_error):
            return TEMPORARY_UNAVAILABLE_MESSAGE
        return REFUSAL_MESSAGE
        
    except Exception:
        return REFUSAL_MESSAGE

if __name__ == "__main__":
    # Test script locally if run directly
    test_q = "What was NVIDIA's revenue and gross profit in fiscal 2024?"
    print(f"Testing local query: '{test_q}'")
    try:
        ans = answer_question(test_q)
        print("Answer:\n", ans)
    except Exception as e:
        print("Test failed:", str(e))
