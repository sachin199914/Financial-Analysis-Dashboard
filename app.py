import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pathlib
import sys

# Set up page config
st.set_page_config(
    page_title="Comparative Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern premium styling (dark mode, clean typography, custom cards)
st.markdown(
    """
    <style>
    /* Main body background & font family */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Title styling */
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Header subtitle styling */
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #8A99AD;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphism Cards */
    .premium-card {
        background: rgba(33, 37, 43, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 1.2rem;
    }
    
    .card-title {
        font-size: 0.85rem;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    
    .card-accent {
        color: #00F2FE;
    }
    
    /* Alert styles */
    .issue-badge {
        background-color: rgba(239, 83, 80, 0.1);
        border: 1px solid rgba(239, 83, 80, 0.3);
        color: #EF5350;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Evidence Quote panel */
    .quote-box {
        background: rgba(0, 242, 254, 0.03);
        border-left: 4px solid #00F2FE;
        border-radius: 4px;
        padding: 1.2rem;
        margin-top: 1rem;
        font-style: italic;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    .quote-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #00F2FE;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-style: normal;
    }
    
    /* Navigation styling tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #8A99AD;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load CSV files helper
@st.cache_data
def load_financial_data():
    root = pathlib.Path(__file__).parent.resolve()
    
    # Read files, handling defaults/errors gracefully
    try:
        sources = pd.read_csv(root / "data" / "sources.csv")
    except Exception:
        sources = pd.DataFrame(columns=["company", "ticker", "filing_type", "fiscal_year", "filing_date", "accession_number", "sec_url", "local_path"])
        
    try:
        extracted = pd.read_csv(root / "data" / "extracted_metrics.csv")
    except Exception:
        extracted = pd.DataFrame(columns=["company", "ticker", "fiscal_year", "metric", "value", "unit", "source_file", "source_section", "source_quote", "confidence"])
        
    try:
        derived = pd.read_csv(root / "data" / "derived_metrics.csv")
    except Exception:
        derived = pd.DataFrame(columns=["company", "ticker", "fiscal_year", "metric", "value", "unit", "formula"])
        
    try:
        issues = pd.read_csv(root / "data" / "extraction_issues.csv")
    except Exception:
        issues = pd.DataFrame(columns=["company", "ticker", "fiscal_year", "metric", "issue"])
        
    return sources, extracted, derived, issues

# Load data
sources, extracted, derived, issues = load_financial_data()

# Helper formats
def format_currency(val):
    if pd.isna(val) or val == "" or val == "N/A":
        return "N/A"
    try:
        val = float(val)
        abs_val = abs(val)
        sign = "-" if val < 0 else ""
        if abs_val >= 1e9:
            return f"{sign}${abs_val / 1e9:.2f} B"
        elif abs_val >= 1e6:
            return f"{sign}${abs_val / 1e6:.2f} M"
        else:
            return f"{sign}${abs_val:,.2f}"
    except ValueError:
        return str(val)

def format_metric_value(val, unit, metric_name=None):
    if pd.isna(val) or val == "" or val == "N/A":
        return "N/A"
    try:
        val = float(val)
        metric_lower = (metric_name or "").lower()
        if unit == "ratio" or "margin" in metric_lower or "growth" in metric_lower or "equity" in metric_lower and "/" in metric_lower:
            return f"{val * 100:.2f}%"
        elif unit == "USD":
            return format_currency(val)
        else:
            return f"{val:,.2f} {unit}"
    except ValueError:
        return str(val)

# Title Block
st.markdown('<div class="dashboard-title">Comparative Financial Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">SEC Filing Auditable Traceability & Performance Dashboard (NVIDIA • AMD • Intel)</div>', unsafe_allow_html=True)

# Define Tabs
tab_overview, tab_charts, tab_evidence, tab_insights, tab_rag = st.tabs([
    "📊 Overview & Scope",
    "📈 Comparative Analysis",
    "🔍 Metric Evidence & Traceability",
    "💡 Automated Insights",
    "💬 RAG Chat Assistant"
])

# ==================== TAB 1: OVERVIEW & SCOPE ====================
with tab_overview:
    st.markdown("### Executive Summary")
    st.write(
        "This platform analyzes and compares the audited financial performance of "
        "NVIDIA (`NVDA`), Advanced Micro Devices (`AMD`), and Intel Corporation (`INTC`) "
        "for the fiscal years 2022, 2023, and 2024. All raw metrics are parsed from inline XBRL "
        "facts inside their official SEC Form 10-K documents, ensuring absolute auditability."
    )
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="premium-card">'
            f'<div class="card-title">Companies Monitored</div>'
            f'<div class="card-value card-accent">{len(sources["ticker"].unique())}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="premium-card">'
            f'<div class="card-title">Cataloged SEC Filings</div>'
            f'<div class="card-value">{len(sources)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="premium-card">'
            f'<div class="card-title">Audited Facts Extracted</div>'
            f'<div class="card-value">{len(extracted)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f'<div class="premium-card">'
            f'<div class="card-title">Extraction Gaps Cataloged</div>'
            f'<div class="card-value" style="color: #EF5350;">{len(issues)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # 2-column breakdown
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown("### Cataloged Document Sources")
        st.write("Source meta-information parsed from EDGAR:")
        
        # Display sources as dataframe with styled URL links
        sources_display = sources.copy()
        sources_display.columns = [c.replace("_", " ").title() for c in sources_display.columns]
        st.dataframe(
            sources_display,
            column_config={
                "Sec Url": st.column_config.LinkColumn("SEC Link", display_text="Open EDGAR URL"),
                "Local Path": None  # Hide local path for cleaner layout
            },
            use_container_width=True,
            hide_index=True
        )
        
    with right_col:
        st.markdown("### Known Extraction Gaps & Limitations")
        st.write(
            "Financial statements are highly custom; some metrics are not tagged uniformly. "
            "Rather than inventing or guessing figures, the system catalogs extraction gaps transparently:"
        )
        
        # Formatted issues table
        if len(issues) > 0:
            formatted_issues = issues.copy()
            formatted_issues.columns = [c.replace("_", " ").title() for c in formatted_issues.columns]
            st.dataframe(
                formatted_issues,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No extraction gaps cataloged. Clean tags found across all documents.")

# ==================== TAB 2: COMPARATIVE ANALYSIS ====================
with tab_charts:
    st.markdown("### Comparative Performance & Financial Ratios")
    st.write("Benchmark performance, profitability, growth, and leverage dynamically across competitors and years.")
    
    # Filter selection
    tickers = list(extracted["ticker"].unique())
    selected_tickers = st.multiselect("Filter Companies:", tickers, default=tickers)
    
    # Pivot extracted data for charts
    chart_extracted = extracted[extracted["ticker"].isin(selected_tickers)].copy()
    chart_derived = derived[derived["ticker"].isin(selected_tickers)].copy()
    
    # Prepare DataFrames for plotting
    # 1. Revenue Plot
    rev_df = chart_extracted[chart_extracted["metric"] == "revenue"].copy()
    rev_df["value_billions"] = rev_df["value"] / 1e9
    
    # 2. Free Cash Flow
    fcf_df = chart_derived[chart_derived["metric"] == "free_cash_flow"].copy()
    fcf_df["value_billions"] = fcf_df["value"] / 1e9
    
    # 3. Margins
    margin_df = chart_derived[chart_derived["metric"].isin(["gross_margin", "operating_margin", "net_margin"])].copy()
    margin_df["value_percent"] = margin_df["value"] * 100
    
    # 4. Growth
    growth_df = chart_derived[chart_derived["metric"] == "revenue_growth_yoy"].copy()
    growth_df["value_percent"] = growth_df["value"] * 100
    
    # 5. Leverage
    leverage_df = chart_derived[chart_derived["metric"] == "liabilities_to_equity"].copy()
    
    # Layout Grid for Charts
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        # Chart 1: Revenue
        if not rev_df.empty:
            fig_rev = px.bar(
                rev_df,
                x="fiscal_year",
                y="value_billions",
                color="company",
                barmode="group",
                title="Revenue Comparison ($ Billions)",
                labels={"value_billions": "Revenue ($ B)", "fiscal_year": "Fiscal Year", "company": "Company"},
                color_discrete_sequence=px.colors.qualitative.Plotly,
                text_auto='.1f'
            )
            fig_rev.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rev, use_container_width=True)
            
        # Chart 2: Margins
        if not margin_df.empty:
            # Let the user pick a margin type
            margin_choice = st.selectbox("Select Margin Ratio to Chart:", ["operating_margin", "gross_margin", "net_margin"])
            margin_sub = margin_df[margin_df["metric"] == margin_choice]
            
            fig_margin = px.bar(
                margin_sub,
                x="fiscal_year",
                y="value_percent",
                color="company",
                barmode="group",
                title=f"{margin_choice.replace('_', ' ').title()} (%)",
                labels={"value_percent": "Margin (%)", "fiscal_year": "Fiscal Year", "company": "Company"},
                text_auto='.1f'
            )
            fig_margin.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_margin, use_container_width=True)
            
    with c_col2:
        # Chart 3: FCF
        if not fcf_df.empty:
            fig_fcf = px.bar(
                fcf_df,
                x="fiscal_year",
                y="value_billions",
                color="company",
                barmode="group",
                title="Free Cash Flow Comparison ($ Billions)",
                labels={"value_billions": "Free Cash Flow ($ B)", "fiscal_year": "Fiscal Year", "company": "Company"},
                text_auto='.1f'
            )
            fig_fcf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_fcf, use_container_width=True)
            
        # Chart 4: YoY growth
        if not growth_df.empty:
            fig_growth = px.line(
                growth_df,
                x="fiscal_year",
                y="value_percent",
                color="company",
                markers=True,
                title="YoY Revenue Growth Rate (%)",
                labels={"value_percent": "Growth Rate (%)", "fiscal_year": "Fiscal Year", "company": "Company"}
            )
            fig_growth.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_growth, use_container_width=True)
            
    # Bottom Row: Leverage
    if not leverage_df.empty:
        fig_lev = px.bar(
            leverage_df,
            x="fiscal_year",
            y="value",
            color="company",
            barmode="group",
            title="Leverage Ratio: Liabilities-to-Equity (Higher indicates higher debt utilization)",
            labels={"value": "Total Liabilities / Stockholders' Equity", "fiscal_year": "Fiscal Year", "company": "Company"},
            text_auto='.2f'
        )
        fig_lev.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lev, use_container_width=True)

# ==================== TAB 3: METRIC EVIDENCE & TRACEABILITY ====================
with tab_evidence:
    st.markdown("### Document Audit Trail & Metric Verification")
    st.write(
        "Trace every single financial figure back to its exact location in the SEC source filing. "
        "Use this tab to audit and verify numbers."
    )
    
    # Filters
    sel_company = st.selectbox("Select Company to Audit:", extracted["company"].unique(), key="audit_company")
    sel_year = st.selectbox("Select Fiscal Year:", sorted(extracted["fiscal_year"].unique()), key="audit_year")
    
    # Get all metric options for this company & year
    company_extracted = extracted[(extracted["company"] == sel_company) & (extracted["fiscal_year"] == sel_year)]
    company_derived = derived[(derived["company"] == sel_company) & (derived["fiscal_year"] == sel_year)]
    
    all_available_metrics = list(company_extracted["metric"].unique()) + list(company_derived["metric"].unique())
    
    if len(all_available_metrics) == 0:
        st.warning("No metrics found for this company and year combination.")
    else:
        sel_metric = st.selectbox("Select Metric to Inspect:", sorted(all_available_metrics), key="audit_metric")
        
        # Display audit details
        st.markdown("#### Audit Analysis")
        
        # Check if the metric is derived or raw
        is_derived = sel_metric in list(company_derived["metric"].unique())
        
        if is_derived:
            metric_row = company_derived[company_derived["metric"] == sel_metric].iloc[0]
            st.info(f"💡 **'{sel_metric}'** is a **derived metric** computed using a reproducible formula.")
            
            # Show derived card
            cols_audit = st.columns(3)
            with cols_audit[0]:
                formatted_val = format_metric_value(metric_row["value"], metric_row["unit"], metric_row["metric"])
                st.metric("Computed Value", formatted_val)
            with cols_audit[1]:
                st.metric("Formula Used", metric_row["formula"])
            with cols_audit[2]:
                st.metric("Unit Type", metric_row["unit"])
                
            st.markdown("##### Underlying Extracted Inputs Used in Computation:")
            
            # Determine which inputs are needed
            formula_lower = str(metric_row["formula"]).lower()
            inputs_found = []
            
            # Query the raw inputs for this company/year
            for idx, raw_row in company_extracted.iterrows():
                raw_name = raw_row["metric"]
                if raw_name in formula_lower or raw_name.replace("_", "") in formula_lower.replace("_", ""):
                    inputs_found.append(raw_row)
                    
            # Fallback check for dynamic leverage calculation:
            if "assets - equity" in formula_lower:
                for idx, raw_row in company_extracted.iterrows():
                    if raw_row["metric"] in ["total_assets", "stockholders_equity"]:
                        inputs_found.append(raw_row)
            
            if inputs_found:
                for input_row in inputs_found:
                    st.markdown(f"**Input: {input_row['metric'].replace('_', ' ').title()}**")
                    col_in1, col_in2 = st.columns([1, 4])
                    with col_in1:
                        st.metric(
                            label="Raw Fact Value",
                            value=format_metric_value(input_row["value"], input_row["unit"], input_row["metric"])
                        )
                    with col_in2:
                        st.markdown(f"**Source Document:** `{input_row['source_file']}`")
                        st.markdown(f"**XBRL Tag / Context:** `{input_row['source_section']}`")
                        st.markdown(
                            f'<div class="quote-box">'
                            f'<div class="quote-header">Original Context Quote Snippet</div>'
                            f'"{input_row["source_quote"]}"'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.write("")
            else:
                st.write("Could not dynamically resolve formula input details from file.")
                
        else:
            # Raw metric auditing
            metric_row = company_extracted[company_extracted["metric"] == sel_metric].iloc[0]
            st.success(f"🔍 **'{sel_metric}'** is a **raw fact** parsed directly from the filing's inline XBRL statements.")
            
            col_a1, col_a2 = st.columns([1, 2])
            with col_a1:
                st.metric(
                    label="Audited Value",
                    value=format_metric_value(metric_row["value"], metric_row["unit"], metric_row["metric"])
                )
                st.write("**Confidence Score:**", f"`{metric_row['confidence'].upper()}`")
                st.write("**Local Document Path:**", f"`{metric_row['source_file']}`")
            with col_a2:
                st.markdown(f"**XBRL Schema Element (Fact Tag & Context):**\n`{metric_row['source_section']}`")
                st.markdown(
                    f'<div class="quote-box">'
                    f'<div class="quote-header">Original 10-K Quote Snippet</div>'
                    f'"{metric_row["source_quote"]}"'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ==================== TAB 4: AUTOMATED INSIGHTS ====================
with tab_insights:
    st.markdown("### Automated Peer Performance Benchmarking")
    st.write(
        "Grounded, rule-based observations generated automatically from the extracted financial metrics "
        "without LLM hallucination risks."
    )
    
    # Filter by year for the insights panel
    sel_insight_year = st.selectbox("Select Year for Benchmarking Insights:", sorted(derived["fiscal_year"].unique()), index=2)
    
    year_derived = derived[derived["fiscal_year"] == sel_insight_year]
    year_extracted = extracted[extracted["fiscal_year"] == sel_insight_year]
    
    if year_derived.empty or year_extracted.empty:
        st.warning(f"Insufficient metric coverage to generate benchmarks for year {sel_insight_year}.")
    else:
        st.markdown(f"#### Key Insights for Fiscal Year {sel_insight_year}")
        
        # 1. Revenue & Growth Leaders
        revs = year_extracted[year_extracted["metric"] == "revenue"]
        growths = year_derived[year_derived["metric"] == "revenue_growth_yoy"]
        
        if not revs.empty:
            max_rev_row = revs.loc[revs["value"].astype(float).idxmax()]
            min_rev_row = revs.loc[revs["value"].astype(float).idxmin()]
            
            st.markdown(
                f"📈 **Revenue Volume:** **{max_rev_row['company']}** was the largest player by revenue in {sel_insight_year}, "
                f"generating **{format_currency(max_rev_row['value'])}**. Conversely, "
                f"**{min_rev_row['company']}** recorded the lowest revenue volume in the peer group at "
                f"**{format_currency(min_rev_row['value'])}**."
            )
            
        if not growths.empty:
            max_growth_row = growths.loc[growths["value"].astype(float).idxmax()]
            min_growth_row = growths.loc[growths["value"].astype(float).idxmin()]
            
            st.markdown(
                f"🚀 **Revenue Growth Trend:** **{max_growth_row['company']}** led the peer group in year-over-year revenue expansion, "
                f"growing by **{max_growth_row['value']*100:.2f}%** compared to the prior year. "
                f"**{min_growth_row['company']}** lagged the peer group with a growth rate of "
                f"**{min_growth_row['value']*100:.2f}%**."
            )
            
        # 2. Profitability Leaders
        op_margins = year_derived[year_derived["metric"] == "operating_margin"]
        net_margins = year_derived[year_derived["metric"] == "net_margin"]
        
        if not op_margins.empty:
            max_op_row = op_margins.loc[op_margins["value"].astype(float).idxmax()]
            min_op_row = op_margins.loc[op_margins["value"].astype(float).idxmin()]
            
            st.markdown(
                f"💎 **Operating Profitability:** **{max_op_row['company']}** achieved the highest operating efficiency, "
                f"securing an operating margin of **{max_op_row['value']*100:.2f}%**. "
                f"**{min_op_row['company']}** recorded the lowest operating efficiency at "
                f"**{min_op_row['value']*100:.2f}%**."
            )
            
        # 3. Cash Flow Generation
        fcfs = year_derived[year_derived["metric"] == "free_cash_flow"]
        if not fcfs.empty:
            max_fcf_row = fcfs.loc[fcfs["value"].astype(float).idxmax()]
            min_fcf_row = fcfs.loc[fcfs["value"].astype(float).idxmin()]
            
            st.markdown(
                f"💸 **Free Cash Flow:** **{max_fcf_row['company']}** was the strongest cash generator, "
                f"producing **{format_currency(max_fcf_row['value'])}** in free cash flow (operating cash flow net of capital expenditures). "
                f"The lowest FCF generator was **{min_fcf_row['company']}** at **{format_currency(min_fcf_row['value'])}**."
            )
            
        # 4. Leverage & Solvency
        leverages = year_derived[year_derived["metric"] == "liabilities_to_equity"]
        if not leverages.empty:
            max_lev_row = leverages.loc[leverages["value"].astype(float).idxmax()]
            min_lev_row = leverages.loc[leverages["value"].astype(float).idxmin()]
            
            st.markdown(
                f"⚖️ **Leverage (Liabilities-to-Equity):** **{max_lev_row['company']}** operates with the highest leverage in the peer group "
                f"at a liabilities-to-equity ratio of **{max_lev_row['value']:.2f}** (indicating higher reliance on debt/liabilities relative to equity). "
                f"**{min_lev_row['company']}** has the most conservative capital structure with a leverage ratio of **{min_lev_row['value']:.2f}**."
            )
            
        # Summary analytical paragraph combining insights
        st.markdown("---")
        st.markdown("##### Performance Insights Matrix")
        
        # Combine into matrix table
        matrix_data = []
        for ticker in sorted(derived["ticker"].unique()):
            tick_derived = year_derived[year_derived["ticker"] == ticker]
            tick_extracted = year_extracted[year_extracted["ticker"] == ticker]
            
            row = {"Company": ticker}
            
            rev_val = tick_extracted[tick_extracted["metric"] == "revenue"]
            row["Revenue"] = format_currency(rev_val.iloc[0]["value"]) if not rev_val.empty else "N/A"
            
            growth_val = tick_derived[tick_derived["metric"] == "revenue_growth_yoy"]
            row["YoY Growth"] = f"{growth_val.iloc[0]['value']*100:.1f}%" if not growth_val.empty else "N/A"
            
            op_val = tick_derived[tick_derived["metric"] == "operating_margin"]
            row["Operating Margin"] = f"{op_val.iloc[0]['value']*100:.1f}%" if not op_val.empty else "N/A"
            
            fcf_val = tick_derived[tick_derived["metric"] == "free_cash_flow"]
            row["Free Cash Flow"] = format_currency(fcf_val.iloc[0]["value"]) if not fcf_val.empty else "N/A"
            
            lev_val = tick_derived[tick_derived["metric"] == "liabilities_to_equity"]
            row["Liabilities/Equity"] = f"{lev_val.iloc[0]['value']:.2f}x" if not lev_val.empty else "N/A"
            
            matrix_data.append(row)
            
        st.table(pd.DataFrame(matrix_data).set_index("Company"))

# ==================== TAB 5: RAG CHAT ASSISTANT ====================
with tab_rag:
    st.markdown("### Grounded SEC Filing Q&A Assistant")
    st.write(
        "Ask natural-language questions about company financial metrics or narrative commentary. "
        "The system retrieves relevant filing chunks, links them with verified metrics, and provides citations."
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can answer questions about AMD, Intel, or NVIDIA filings for 2022–2024. What would you like to investigate?"}
        ]
        
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # User input
    if prompt := st.chat_input("Ask a question about the filings (e.g. 'Why did Intel's margins drop in 2024?')"):
        # Append user message immediately to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately in chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate answer and render immediately
        with st.chat_message("assistant"):
            try:
                # We dynamically import to check if Q&A is implemented
                if 'src.rag' not in sys.modules:
                    import src.rag as rag
                else:
                    rag = sys.modules['src.rag']
                    
                with st.spinner("Analyzing filings and verifying metrics..."):
                    answer = rag.answer_question(prompt)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except NotImplementedError as e:
                # Graceful offline mode warning for Phase 5 placeholder
                offline_msg = (
                    "⚠️ **RAG Q&A Engine (Phase 5 Placeholder):**\n\n"
                    "The natural-language retrieval-augmented Q&A pipeline is currently in scaffold mode. "
                    "In the next phase (Phase 5), we will initialize the vector storage index, parse filing "
                    "chunks, and wire this chat interface to an LLM provider.\n\n"
                    f"*System logs:* `{str(e)}`"
                )
                st.markdown(offline_msg)
                st.session_state.messages.append({"role": "assistant", "content": offline_msg})
                
            except Exception as e:
                error_msg = f"An error occurred while generating a response: `{str(e)}`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to cleanly re-draw all messages above the chat input box at the bottom of the page
        st.rerun()
