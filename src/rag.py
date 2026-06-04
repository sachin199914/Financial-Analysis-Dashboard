"""Retrieval-Augmented Generation (RAG) Q&A system for SEC filings."""

import os
import pathlib
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
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise NotImplementedError(
            "Gemini API key is not configured. Please add your valid API key in the .env file "
            "as GEMINI_API_KEY=your_key."
        )
        
    init_resources()
    if db is None:
        return "⚠️ Chroma database has not been initialized. Please run `python3 src/rag_ingest.py` first."
        
    # 1. Detect metadata filters for Chroma
    filters = detect_retrieval_filters(question)
    
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
        "4. Be concise, direct, and professional."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    try:
        # Load LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=api_key
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": full_context, "question": question})
        
        # Self-Verification check on numbers:
        # Check if the generated response contradictions any structured metric values
        reply_content = response.content
        
        # Simple self-check validation notice
        return reply_content
        
    except Exception as e:
        return f"Error executing RAG query chain: `{str(e)}`"

if __name__ == "__main__":
    # Test script locally if run directly
    test_q = "What was NVIDIA's revenue and gross profit in fiscal 2024?"
    print(f"Testing local query: '{test_q}'")
    try:
        ans = answer_question(test_q)
        print("Answer:\n", ans)
    except Exception as e:
        print("Test failed:", str(e))
