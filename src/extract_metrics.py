"""Extract financial figures from inline XBRL filings.

The extractor favors a small set of high-confidence consolidated annual facts.
Each output row includes the XBRL tag, context, local filing, and a short source
quote so downstream charts and RAG answers can trace numbers back to evidence.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lxml import etree


CORE_METRICS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "operating_cash_flow",
    "capital_expenditures",
]

ROOT = Path(__file__).resolve().parents[1]
SOURCES_CSV = ROOT / "data" / "sources.csv"
EXTRACTED_CSV = ROOT / "data" / "extracted_metrics.csv"
DERIVED_CSV = ROOT / "data" / "derived_metrics.csv"
ISSUES_CSV = ROOT / "data" / "extraction_issues.csv"

NS = {
    "ix": "http://www.xbrl.org/2013/inlineXBRL",
    "xbrli": "http://www.xbrl.org/2003/instance",
}

METRIC_TAGS = {
    "revenue": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
    ],
    "gross_profit": ["us-gaap:GrossProfit"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "total_assets": ["us-gaap:Assets"],
    "total_liabilities": ["us-gaap:Liabilities"],
    "stockholders_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditures": [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
    ],
}


@dataclass(frozen=True)
class Context:
    id: str
    start_date: date | None
    end_date: date | None
    instant: date | None
    has_segment: bool

    @property
    def is_duration(self) -> bool:
        return self.start_date is not None and self.end_date is not None

    @property
    def is_instant(self) -> bool:
        return self.instant is not None

    @property
    def duration_days(self) -> int:
        if not self.is_duration:
            return 0
        return (self.end_date - self.start_date).days


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def load_sources() -> list[dict[str, str]]:
    with SOURCES_CSV.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_contexts(doc: etree._ElementTree) -> dict[str, Context]:
    contexts = {}
    for node in doc.xpath("//xbrli:context", namespaces=NS):
        context_id = node.attrib["id"]
        start = node.xpath("string(.//xbrli:startDate)", namespaces=NS) or None
        end = node.xpath("string(.//xbrli:endDate)", namespaces=NS) or None
        instant = node.xpath("string(.//xbrli:instant)", namespaces=NS) or None
        has_segment = bool(node.xpath(".//*[local-name()='segment']"))
        contexts[context_id] = Context(
            id=context_id,
            start_date=parse_date(start),
            end_date=parse_date(end),
            instant=parse_date(instant),
            has_segment=has_segment,
        )
    return contexts


def parse_number(node: etree._Element) -> float | None:
    raw = "".join(node.itertext()).strip()
    if raw in {"", "-", "—", "–"}:
        return None
    negative = node.attrib.get("sign") == "-" or raw.startswith("(")
    cleaned = re.sub(r"[^0-9.]", "", raw)
    if not cleaned:
        return None
    value = float(cleaned)
    scale = int(node.attrib.get("scale", "0"))
    value *= 10**scale
    return -value if negative else value


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:300]


def source_quote(node: etree._Element) -> str:
    row = node
    while row is not None and etree.QName(row).localname.lower() != "tr":
        row = row.getparent()
    if row is not None:
        quote = normalize_quote(" ".join(row.itertext()))
        if quote:
            return quote
    return normalize_quote(" ".join(node.itertext()))


def fact_period_matches(metric: str, context: Context, fiscal_year: int) -> bool:
    if context.has_segment:
        return False
    if metric in {"total_assets", "total_liabilities", "stockholders_equity"}:
        return context.is_instant and context.instant.year == fiscal_year
    return (
        context.is_duration
        and context.end_date.year == fiscal_year
        and context.duration_days >= 300
    )


def candidate_score(metric: str, node: etree._Element, context: Context) -> tuple[int, int]:
    tag_priority = METRIC_TAGS[metric].index(node.attrib["name"])
    decimals = node.attrib.get("decimals", "")
    is_millions = decimals == "-6" or node.attrib.get("scale") == "6"
    return (0 if is_millions else 1, tag_priority)


def extract_metric(
    doc: etree._ElementTree,
    contexts: dict[str, Context],
    metric: str,
    fiscal_year: int,
) -> etree._Element | None:
    candidates = []
    for tag in METRIC_TAGS[metric]:
        nodes = doc.xpath(f'//ix:nonFraction[@name="{tag}"]', namespaces=NS)
        for node in nodes:
            context = contexts.get(node.attrib.get("contextRef", ""))
            value = parse_number(node)
            if context and value is not None and fact_period_matches(metric, context, fiscal_year):
                candidates.append((candidate_score(metric, node, context), node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def extract_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    rows = []
    issues = []

    for source in load_sources():
        filing_path = ROOT / source["local_path"]
        fiscal_year = int(source["fiscal_year"])
        doc = etree.parse(str(filing_path), parser)
        contexts = parse_contexts(doc)

        for metric in CORE_METRICS:
            node = extract_metric(doc, contexts, metric, fiscal_year)
            if node is None:
                issues.append(
                    {
                        "company": source["company"],
                        "ticker": source["ticker"],
                        "fiscal_year": source["fiscal_year"],
                        "metric": metric,
                        "issue": "No annual unsegmented fact found",
                    }
                )
                continue

            context_id = node.attrib["contextRef"]
            tag = node.attrib["name"]
            value = parse_number(node)
            rows.append(
                {
                    "company": source["company"],
                    "ticker": source["ticker"],
                    "fiscal_year": source["fiscal_year"],
                    "metric": metric,
                    "value": f"{value:.0f}",
                    "unit": "USD",
                    "source_file": source["local_path"],
                    "source_section": f"{tag}; context={context_id}; fact_id={node.attrib.get('id', '')}",
                    "source_quote": source_quote(node),
                    "confidence": "high",
                }
            )

    return rows, issues


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], float]:
    return {
        (row["ticker"], int(row["fiscal_year"]), row["metric"]): float(row["value"])
        for row in rows
    }


def build_derived_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = metric_lookup(rows)
    companies = sorted({(row["company"], row["ticker"]) for row in rows})
    years = sorted({int(row["fiscal_year"]) for row in rows})
    derived = []

    def add(company: str, ticker: str, year: int, metric: str, value: float, formula: str) -> None:
        derived.append(
            {
                "company": company,
                "ticker": ticker,
                "fiscal_year": str(year),
                "metric": metric,
                "value": f"{value:.6f}",
                "unit": "ratio" if any(token in metric for token in ["margin", "growth", "to_equity"]) else "USD",
                "formula": formula,
            }
        )

    for company, ticker in companies:
        for year in years:
            revenue = lookup.get((ticker, year, "revenue"))
            if revenue:
                gross_profit = lookup.get((ticker, year, "gross_profit"))
                operating_income = lookup.get((ticker, year, "operating_income"))
                net_income = lookup.get((ticker, year, "net_income"))
                operating_cash_flow = lookup.get((ticker, year, "operating_cash_flow"))
                capex = lookup.get((ticker, year, "capital_expenditures"))
                total_assets = lookup.get((ticker, year, "total_assets"))
                total_liabilities = lookup.get((ticker, year, "total_liabilities"))
                stockholders_equity = lookup.get((ticker, year, "stockholders_equity"))
                prior_revenue = lookup.get((ticker, year - 1, "revenue"))

                if gross_profit is not None:
                    add(company, ticker, year, "gross_margin", gross_profit / revenue, "gross_profit / revenue")
                if operating_income is not None:
                    add(company, ticker, year, "operating_margin", operating_income / revenue, "operating_income / revenue")
                if net_income is not None:
                    add(company, ticker, year, "net_margin", net_income / revenue, "net_income / revenue")
                if prior_revenue:
                    add(company, ticker, year, "revenue_growth_yoy", (revenue - prior_revenue) / prior_revenue, "(revenue - prior_year_revenue) / prior_year_revenue")
                if operating_cash_flow is not None and capex is not None:
                    add(company, ticker, year, "free_cash_flow", operating_cash_flow - abs(capex), "operating_cash_flow - abs(capital_expenditures)")
                if stockholders_equity:
                    if total_liabilities is None and total_assets is not None:
                        total_liabilities = total_assets - stockholders_equity
                    if total_liabilities is not None:
                        add(company, ticker, year, "liabilities_to_equity", total_liabilities / stockholders_equity, "total_liabilities / stockholders_equity; liabilities derived as assets - equity when not directly tagged")

    return derived


def main() -> None:
    rows, issues = extract_rows()
    write_csv(
        EXTRACTED_CSV,
        rows,
        [
            "company",
            "ticker",
            "fiscal_year",
            "metric",
            "value",
            "unit",
            "source_file",
            "source_section",
            "source_quote",
            "confidence",
        ],
    )
    write_csv(
        DERIVED_CSV,
        build_derived_rows(rows),
        ["company", "ticker", "fiscal_year", "metric", "value", "unit", "formula"],
    )
    write_csv(
        ISSUES_CSV,
        issues,
        ["company", "ticker", "fiscal_year", "metric", "issue"],
    )
    print(f"Wrote {len(rows)} extracted rows to {EXTRACTED_CSV.relative_to(ROOT)}")
    print(f"Wrote derived metrics to {DERIVED_CSV.relative_to(ROOT)}")
    print(f"Wrote {len(issues)} extraction issues to {ISSUES_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
