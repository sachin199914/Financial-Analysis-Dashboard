"""Ingest filing HTML documents, parse text, chunk, and index into Chroma DB."""

import os
import csv
import pathlib
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILINGS_DIR = ROOT / "data" / "filings"
SOURCES_CSV = ROOT / "data" / "sources.csv"
VECTORSTORE_DIR = ROOT / "data" / "vectorstore"

def parse_html_to_text(html_path: pathlib.Path) -> str:
    """Read HTML and extract clean text, stripping styles and script tags."""
    with html_path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    soup = BeautifulSoup(content, "lxml")
    
    # Remove irrelevant nodes
    for element in soup(["style", "script", "head", "title", "meta"]):
        element.decompose()
        
    # Get clean text
    text = soup.get_text(separator="\n")
    
    # Clean whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = "\n".join(chunk for chunk in chunks if chunk)
    return clean_text

def main() -> None:
    print("Initializing embedding model (local all-MiniLM-L6-v2)...")
    # This runs completely locally and will download the model once to a cache directory
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Read sources catalog
    if not SOURCES_CSV.exists():
        print(f"Error: {SOURCES_CSV} does not exist. Run ingest.py first.")
        return
        
    sources = []
    with SOURCES_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row)
            
    print(f"Found {len(sources)} source files to index.")
    
    # Text splitter config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    
    all_chunks = []
    all_metadata = []
    
    for src in sources:
        local_path = ROOT / src["local_path"]
        if not local_path.exists():
            print(f"Filing not found locally: {local_path.name}. Skipping...")
            continue
            
        print(f"Parsing and cleaning text from: {local_path.name}...")
        text_content = parse_html_to_text(local_path)
        print(f"Extracted {len(text_content)} characters. Splitting into chunks...")
        
        chunks = text_splitter.split_text(text_content)
        print(f"Generated {len(chunks)} chunks.")
        
        # Prepare metadata for each chunk
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "company": src["company"],
                "ticker": src["ticker"],
                "filing_type": src["filing_type"],
                "fiscal_year": int(src["fiscal_year"]),
                "source_file": src["local_path"]
            })
            
    # Write to Chroma DB
    print(f"Indexing total of {len(all_chunks)} chunks to Chroma DB at {VECTORSTORE_DIR}...")
    
    # Create database directory if missing
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    db = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadata,
        persist_directory=str(VECTORSTORE_DIR)
    )
    
    print("Database indexing complete and persisted successfully!")

if __name__ == "__main__":
    main()
