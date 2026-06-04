"""Automated evaluation framework for the RAG Q&A engine."""

import os
import csv
import math
import pathlib
import re
import sys
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Ensure src is on path
sys.path.append(str(ROOT))
import src.rag as rag

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?\b")


def parse_number(value: str) -> float | None:
    cleaned = value.strip().replace("$", "").replace(",", "")
    is_percent = cleaned.endswith("%")
    cleaned = cleaned.rstrip("%")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number / 100 if is_percent else number


def numbers_from_text(text: str) -> list[float]:
    return [num for token in NUMBER_RE.findall(text) if (num := parse_number(token)) is not None]


def numeric_variants(number: float) -> list[float]:
    variants = {number}
    if abs(number) > 1000:
        variants.add(number / 1_000)
        variants.add(number / 1_000_000)
        variants.add(number * 1_000_000)
    if 0 < abs(number) <= 100:
        variants.add(number / 100)
        variants.add(number * 100)
    return list(variants)


def numbers_close(expected: float, actual: float) -> bool:
    tolerance = max(0.02, abs(expected) * 0.025)
    return math.isclose(expected, actual, rel_tol=0.025, abs_tol=tolerance)


def expected_token_matched(expected: str, answer: str, answer_numbers: list[float]) -> bool:
    expected_lower = expected.lower()
    answer_lower = answer.lower()
    expected_number = parse_number(expected)

    if expected_number is None:
        return expected_lower in answer_lower

    for variant in numeric_variants(expected_number):
        if any(numbers_close(variant, actual) for actual in answer_numbers):
            return True
    return expected_lower in answer_lower


def expected_content_matches(expected: list[str], answer: str, answer_type: str) -> bool:
    answer_lower = answer.lower()
    answer_numbers = numbers_from_text(answer)

    if answer_type == "narrative":
        return any(token.lower() in answer_lower for token in expected)

    return all(expected_token_matched(token, answer, answer_numbers) for token in expected)


def has_citation(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(token in answer_lower for token in ["10k", "10-k", ".htm", "filings/", "source:", "quote:"])

QUESTIONS = [
    # 1. Lookups
    {
        "id": 1,
        "question": "What was NVIDIA's revenue and gross profit in fiscal 2024?",
        "type": "lookup",
        "expected": ["60,922", "44,301", "NVDA_2024_10K.htm"],
        "notes": "Verify direct lookup and citations for NVDA 2024."
    },
    {
        "id": 2,
        "question": "What was AMD's gross profit and net income in 2023?",
        "type": "lookup",
        "expected": ["10,460", "854", "AMD_2023_10K.htm"],
        "notes": "Verify AMD 2023 gross profit and net income facts."
    },
    {
        "id": 3,
        "question": "What was Intel's operating income in 2022?",
        "type": "lookup",
        "expected": ["2,334", "INTC_2022_10K.htm"],
        "notes": "Verify Intel 2022 operating income fact."
    },
    # 2. Calculated metrics
    {
        "id": 4,
        "question": "What was AMD's free cash flow in 2024?",
        "type": "calculated",
        "expected": ["2,405", "3,041", "636"],
        "notes": "Verify derived FCF calculation (Operating Cash Flow - Capex)."
    },
    {
        "id": 5,
        "question": "What was Intel's liabilities-to-equity ratio in 2023?",
        "type": "calculated",
        "expected": ["0.81", "81.4", "105,590", "191,572"],
        "notes": "Verify dynamic liabilities-to-equity ratio computation."
    },
    {
        "id": 6,
        "question": "What was NVIDIA's operating margin in 2024?",
        "type": "calculated",
        "expected": ["54.12", "54.1"],
        "notes": "Verify derived operating margin (Operating Income / Revenue)."
    },
    # 3. Narrative MD&A
    {
        "id": 7,
        "question": "What reason did Intel give for its lower gross margins in 2024?",
        "type": "narrative",
        "expected": ["INTC_2024_10K.htm", "gross margin"],
        "notes": "Verify narrative commentary retrieval on INTC margin drop."
    },
    {
        "id": 8,
        "question": "What did AMD management say about their R&D investments in 2024?",
        "type": "narrative",
        "expected": ["AMD_2024_10K.htm", "R&D", "research and development"],
        "notes": "Verify AMD R&D narrative commentary."
    },
    {
        "id": 9,
        "question": "How did NVIDIA's Datacenter segment perform in 2024 according to management?",
        "type": "narrative",
        "expected": ["NVDA_2024_10K.htm", "datacenter", "data center"],
        "notes": "Verify NVIDIA Datacenter segment growth details."
    },
    # 4. Unanswerable (Refusal)
    {
        "id": 10,
        "question": "What was Apple's revenue in 2024?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Out of company scope: must refuse."
    },
    {
        "id": 11,
        "question": "What was NVIDIA's market capitalization in fiscal 2020?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Out of year scope: must refuse."
    },
    {
        "id": 12,
        "question": "What was Intel's exact CPU unit sales volume in 2024?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Fact not in scope or filings: must refuse."
    },
    {
        "id": 13,
        "question": "What was AMD's total employee count in 2020?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Out of scope: must refuse."
    },
    {
        "id": 14,
        "question": "What was Google's net income in 2023?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Out of scope: must refuse."
    },
    {
        "id": 15,
        "question": "What was the weather like in Santa Clara in 2024?",
        "type": "unanswerable",
        "expected": ["cannot answer", "sufficient information"],
        "notes": "Completely irrelevant question: must refuse."
    }
]

def main() -> None:
    api_key_openai = os.environ.get("OPENAI_API_KEY")
    api_key_gemini = os.environ.get("GEMINI_API_KEY")
    has_openai = api_key_openai and api_key_openai != "your_openai_api_key_here"
    has_gemini = api_key_gemini and api_key_gemini != "your_gemini_api_key_here"
    if not has_openai and not has_gemini:
        print("Error: GEMINI_API_KEY or OPENAI_API_KEY must be populated in the .env file to run evaluation.")
        return
        
    print("Starting automated evaluation framework...")
    
    questions_file = ROOT / "eval" / "questions.csv"
    results_file = ROOT / "eval" / "results.csv"
    
    # 1. Write questions.csv
    print(f"Writing questions to {questions_file}...")
    with questions_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "expected_answer_type", "expected_source", "notes"])
        for q in QUESTIONS:
            writer.writerow([q["id"], q["question"], q["type"], ", ".join(q["expected"]), q["notes"]])
            
    # 2. Run Q&A and evaluate
    results = []
    correct_count = 0
    refused_correctly = 0
    hallucination_count = 0
    citation_accurate_count = 0
    
    print("\nRunning RAG Q&A queries and grading outputs...")
    for q in QUESTIONS:
        print(f"\n--- Running Question {q['id']} ({q['type']}) ---")
        print(f"Q: {q['question']}")
        
        answer = rag.answer_question(q["question"])
        print(f"A: {answer}")
        
        # Grading logic
        answer_lower = answer.lower()
        
        # Check correctness & citations
        is_correct = False
        is_citation_accurate = False
        is_refused = "cannot answer" in answer_lower or "sufficient information" in answer_lower
        is_hallucination = False
        
        # A. Unanswerable grading
        if q["type"] == "unanswerable":
            if is_refused:
                is_correct = True
                is_citation_accurate = True  # Refusal doesn't need filing citations
                refused_correctly += 1
            else:
                # LLM didn't refuse -> hallucination/inaccuracy!
                is_correct = False
                is_hallucination = True
                hallucination_count += 1
                
        # B. Lookup / Calculated / Narrative grading
        else:
            matches_expected = expected_content_matches(q["expected"], answer, q["type"])

            if matches_expected:
                is_correct = True
                correct_count += 1
                
            # Check if source file is cited in the text
            if has_citation(answer):
                is_citation_accurate = True
                citation_accurate_count += 1
                
            # If incorrect and claimed some outside numbers, flag hallucination
            if not is_correct and not is_refused:
                is_hallucination = True
                hallucination_count += 1
                
        results.append({
            "id": q["id"],
            "question": q["question"],
            "answer_correct": "True" if is_correct else "False",
            "citation_accurate": "True" if is_citation_accurate else "False",
            "refused_when_unanswerable": "True" if is_refused else "False",
            "hallucination": "True" if is_hallucination else "False",
            "notes": f"Answer matched expected content: {is_correct} | Cited: {is_citation_accurate}"
        })
        
    # 3. Write results.csv
    print(f"\nWriting results to {results_file}...")
    with results_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer_correct", "citation_accurate", "refused_when_unanswerable", "hallucination", "notes"])
        for r in results:
            writer.writerow([r["id"], r["question"], r["answer_correct"], r["citation_accurate"], r["refused_when_unanswerable"], r["hallucination"], r["notes"]])
            
    # Calculate statistics
    total_q = len(QUESTIONS)
    unanswerable_q = sum(1 for q in QUESTIONS if q["type"] == "unanswerable")
    answerable_q = total_q - unanswerable_q
    
    correctness_pct = (correct_count + refused_correctly) / total_q * 100
    citation_pct = citation_accurate_count / answerable_q * 100
    refusal_pct = refused_correctly / unanswerable_q * 100
    hallucination_pct = hallucination_count / total_q * 100
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Questions: {total_q}")
    print(f"Answer Correctness: {correctness_pct:.1f}% ({correct_count + refused_correctly}/{total_q})")
    print(f"Citation Accuracy (on answerable): {citation_pct:.1f}% ({citation_accurate_count}/{answerable_q})")
    print(f"Refusal Rate (on unanswerable): {refusal_pct:.1f}% ({refused_correctly}/{unanswerable_q})")
    print(f"Hallucination Rate: {hallucination_pct:.1f}% ({hallucination_count}/{total_q})")
    print("====================================================")

if __name__ == "__main__":
    main()
