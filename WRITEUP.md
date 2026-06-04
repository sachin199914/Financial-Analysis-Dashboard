# WRITEUP

## Architecture

This project uses a small Streamlit app backed by local SEC filing files, a structured metrics table, and a retrieval layer for cited question answering. The intended flow is:

1. Download public SEC filings for NVIDIA, AMD, and Intel.
2. Extract a compact set of high-confidence financial figures.
3. Store extracted figures with document, year, unit, and source evidence.
4. Compute derived metrics from the extracted figures.
5. Use RAG over filing chunks for management commentary and narrative explanations.
6. Evaluate answer correctness, citation accuracy, and hallucination behavior.

## Evaluation

To be completed after the RAG and metrics flows are implemented.

## Executive Trust

To be completed after evaluating citation accuracy and numerical correctness.

## Most Interesting Insight

To be completed after extracted metrics and charts are available.

## Failure Diagnosed

To be completed. Candidate failure modes include unit mismatches, ambiguous table labels, missing values, or bad retrieval.

## AI Tool Usage

To be completed with specific examples from implementation.

## Framework Notes

To be completed after implementation.

