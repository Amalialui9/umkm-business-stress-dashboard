import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="UMKM Decision Stress Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #06101F;
        --panel: #0E1A2B;
        --panel-2: #101F34;
        --text: #F3F7FF;
        --muted: #AAB7CF;
        --cyan: #00E5FF;
        --green: #20E3B2;
        --yellow: #F9C74F;
        --red: #FF5A7A;
        --purple: #A78BFA;
        --blue: #5EA8FF;
        --border: rgba(255,255,255,0.12);
    }
    .main .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(14,26,43,0.95), rgba(16,31,52,0.95));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.28);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button p {
        font-size: 0.92rem;
        font-weight: 650;
    }
    .hero {
        background:
            radial-gradient(circle at top left, rgba(0,229,255,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(167,139,250,0.18), transparent 28%),
            linear-gradient(135deg, rgba(14,26,43,0.98), rgba(7,16,31,0.98));
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 2.05rem;
        line-height: 1.1;
        margin: 0 0 8px 0;
        font-weight: 800;
        color: var(--text);
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 1120px;
        margin: 0;
    }
    .pill-row { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 17px; }
    .pill {
        border: 1px solid rgba(0,229,255,0.22);
        color: #DDFBFF;
        background: rgba(0,229,255,0.08);
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 0.78rem;
        font-weight: 650;
    }
    .panel {
        background: linear-gradient(145deg, rgba(14,26,43,0.96), rgba(9,19,35,0.96));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 14px 38px rgba(0,0,0,0.24);
    }
    .section-title {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 780;
        margin-bottom: 4px;
    }
    .caption {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .status-safe {
        color: #052e26; background: linear-gradient(90deg, #20E3B2, #9BF6D7);
        border-radius: 999px; padding: 7px 12px; font-weight: 800; display: inline-block;
    }
    .status-watch {
        color: #2F2300; background: linear-gradient(90deg, #F9C74F, #FFE08A);
        border-radius: 999px; padding: 7px 12px; font-weight: 800; display: inline-block;
    }
    .status-risk {
        color: #fff; background: linear-gradient(90deg, #FF5A7A, #FF8FAB);
        border-radius: 999px; padding: 7px 12px; font-weight: 800; display: inline-block;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
    }
    .table-wrap {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_TEMPLATE = "plotly_dark"
COLOR_SEQ = ["#00E5FF", "#20E3B2", "#F9C74F", "#A78BFA", "#FF5A7A", "#5EA8FF", "#F97316"]

# ==========================================================
# HELPERS
# ==========================================================

def rupiah(value: float) -> str:
    if pd.isna(value):
        return "Rp0"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}Rp{value/1_000_000_000:.2f} M"
    if value >= 1_000_000:
        return f"{sign}Rp{value/1_000_000:.2f} jt"
    if value >= 1_000:
        return f"{sign}Rp{value/1_000:.1f} rb"
    return f"{sign}Rp{value:,.0f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{value:.1f}%"


def number(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"


def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_v = s.quantile(0.02)
    max_v = s.quantile(0.98)
    if math.isclose(float(max_v - min_v), 0.0):
        out = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        out = ((s - min_v) / (max_v - min_v)).clip(0, 1) * 100
    if invert:
        out = 100 - out
    return out.fillna(50)


def clean_peak_latency(value):
    mapping = {"Low": 1, "Med": 2, "Medium": 2, "High": 3}
    return mapping.get(str(value), 2)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = Path(__file__).parent / "synthetic_umkm_data.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Monthly_Revenue", "Net_Profit_Margin (%)", "Burn_Rate_Ratio",
        "Transaction_Count", "Avg_Historical_Rating", "Review_Volatility",
        "Business_Tenure_Months", "Repeat_Order_Rate (%)", "Digital_Adoption_Score",
        "Peak_Hour_Latency", "Location_Competitiveness", "Sentiment_Score", "Class"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "Monthly_Revenue", "Net_Profit_Margin (%)", "Burn_Rate_Ratio", "Transaction_Count",
        "Avg_Historical_Rating", "Review_Volatility", "Business_Tenure_Months",
        "Repeat_Order_Rate (%)", "Digital_Adoption_Score", "Location_Competitiveness",
        "Sentiment_Score"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Estimated_Profit"] = df["Monthly_Revenue"] * df["Net_Profit_Margin (%)"] / 100
    df["Estimated_Cost"] = df["Monthly_Revenue"] - df["Estimated_Profit"]
    df["Revenue_per_Transaction"] = df["Monthly_Revenue"] / df["Transaction_Count"].replace(0, np.nan)
    df["Churn_Proxy (%)"] = (100 - df["Repeat_Order_Rate (%)"]).clip(lower=0, upper=100)
    df["Latency_Level"] = df["Peak_Hour_Latency"].apply(clean_peak_latency)

    df["Internal_Risk_Score"] = (
        0.28 * normalize(df["Burn_Rate_Ratio"]) +
        0.25 * normalize(df["Net_Profit_Margin (%)"], invert=True) +
        0.20 * normalize(df["Latency_Level"]) +
        0.17 * normalize(df["Digital_Adoption_Score"], invert=True) +
        0.10 * normalize(df["Estimated_Cost"] / df["Monthly_Revenue"].replace(0, np.nan))
    ).clip(0, 100)

    df["External_Risk_Score"] = (
        0.28 * normalize(df["Transaction_Count"], invert=True) +
        0.24 * normalize(df["Location_Competitiveness"]) +
        0.18 * normalize(df["Sentiment_Score"], invert=True) +
        0.15 * normalize(df["Avg_Historical_Rating"], invert=True) +
        0.15 * normalize(df["Review_Volatility"])
    ).clip(0, 100)

    df["Total_Risk_Score"] = (0.52 * df["Internal_Risk_Score"] + 0.48 * df["External_Risk_Score"]).clip(0, 100)
    df["Risk_Category"] = pd.cut(
        df["Total_Risk_Score"],
        bins=[-1, 35, 60, 100],
        labels=["Low Risk", "Moderate Risk", "High Risk"]
    )
    df["Digital_Maturity"] = pd.cut(
        df["Digital_Adoption_Score"],
        bins=[0, 3, 6, 10],
        labels=["Low Digital", "Developing", "Advanced Digital"],
        include_lowest=True
    )
    df["Business_Stage"] = pd.cut(
        df["Business_Tenure_Months"],
        bins=[0, 24, 60, 120, 240],
        labels=["0-2 years", "2-5 years", "5-10 years", ">10 years"],
        include_lowest=True
    )
    return df


def filter_data(df: pd.DataFrame, show_upload: bool = True):
    with st.sidebar:
        st.markdown("### Control Panel")
        st.caption("Gunakan filter ini untuk melihat segmen UMKM yang berbeda.")

        classes = sorted([x for x in df["Class"].dropna().unique().tolist()])
        selected_classes = st.multiselect("Class UMKM", classes, default=classes)

        latency_vals = sorted([x for x in df["Peak_Hour_Latency"].dropna().unique().tolist()])
        selected_latency = st.multiselect("Peak Hour Latency", latency_vals, default=latency_vals)

        maturity_vals = [str(x) for x in df["Digital_Maturity"].dropna().unique().tolist()]
        maturity_vals = sorted(maturity_vals)
        selected_maturity = st.multiselect("Digital Maturity", maturity_vals, default=maturity_vals)

        risk_appetite = st.selectbox(
            "Risk Appetite",
            ["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="Conservative menghindari kerugian; Moderate menyeimbangkan risiko-return; Aggressive mengejar return tinggi dengan risiko lebih besar.",
        )

        uploaded = None
        if show_upload:
            st.markdown("---")
            st.caption("Upload CSV lain jika ingin memakai dataset revisi.")
            uploaded = st.file_uploader("Optional CSV upload", type=["csv"], label_visibility="collapsed", key="main_csv_upload")

    work_df = df.copy()
    if selected_classes:
        work_df = work_df[work_df["Class"].isin(selected_classes)]
    if selected_latency:
        work_df = work_df[work_df["Peak_Hour_Latency"].isin(selected_latency)]
    if selected_maturity:
        work_df = work_df[work_df["Digital_Maturity"].astype(str).isin(selected_maturity)]
    return work_df, risk_appetite, uploaded


def thresholds_for_appetite(appetite: str):
    if appetite == "Conservative":
        return {"min_margin": 5.0, "max_burn": 1.00, "max_loss": 10.0, "min_profit_ratio": 0.92}
    if appetite == "Aggressive":
        return {"min_margin": -5.0, "max_burn": 1.25, "max_loss": 35.0, "min_profit_ratio": 0.55}
    return {"min_margin": 0.0, "max_burn": 1.10, "max_loss": 20.0, "min_profit_ratio": 0.75}


def evaluate_status(profit, margin, burn, thresholds, base_profit=None):
    base_profit = profit if base_profit is None else base_profit
    profit_ratio = profit / base_profit if base_profit and base_profit != 0 else 1
    if profit < 0 or margin < thresholds["min_margin"] or burn > thresholds["max_burn"] * 1.10 or profit_ratio < thresholds["min_profit_ratio"] * 0.75:
        return "Critical"
    if margin < thresholds["min_margin"] + 4 or burn > thresholds["max_burn"] or profit_ratio < thresholds["min_profit_ratio"]:
        return "Watch"
    return "Safe"


def status_badge(status: str):
    css = "status-safe" if status == "Safe" else "status-watch" if status == "Watch" else "status-risk"
    return f'<span class="{css}">{status}</span>'


def scenario_calculation(base, demand_chg, rpu_chg, cost_chg, margin_chg=0.0, repeat_chg=0.0, digital_chg=0.0, burn_chg=0.0):
    transaction = base["transactions"] * (1 + demand_chg)
    rpu = base["rpu"] * (1 + rpu_chg)
    revenue = transaction * rpu
    cost = base["cost"] * (1 + cost_chg)
    profit_from_cost = revenue - cost
    margin_adjusted_profit = revenue * ((base["margin"] + margin_chg) / 100)
    # Blend cost-based and margin-based output to keep simulation stable.
    profit = 0.72 * profit_from_cost + 0.28 * margin_adjusted_profit
    margin = profit / revenue * 100 if revenue else 0
    repeat = np.clip(base["repeat"] + repeat_chg, 0, 100)
    burn = max(0.01, base["burn"] * (1 + burn_chg))
    digital = np.clip(base["digital"] + digital_chg, 1, 10)
    return {
        "Revenue": revenue,
        "Cost": cost,
        "Profit": profit,
        "Margin (%)": margin,
        "Repeat Order (%)": repeat,
        "Burn Rate": burn,
        "Digital Score": digital,
    }


def base_metrics(df: pd.DataFrame):
    revenue = df["Monthly_Revenue"].sum()
    profit = df["Estimated_Profit"].sum()
    cost = df["Estimated_Cost"].sum()
    transactions = df["Transaction_Count"].sum()
    return {
        "revenue": revenue,
        "profit": profit,
        "cost": cost,
        "transactions": transactions,
        "margin": profit / revenue * 100 if revenue else 0,
        "rpu": revenue / transactions if transactions else df["Revenue_per_Transaction"].mean(),
        "repeat": df["Repeat_Order_Rate (%)"].mean(),
        "burn": df["Burn_Rate_Ratio"].mean(),
        "digital": df["Digital_Adoption_Score"].mean(),
        "internal_risk": df["Internal_Risk_Score"].mean(),
        "external_risk": df["External_Risk_Score"].mean(),
        "total_risk": df["Total_Risk_Score"].mean(),
    }


def chart_layout(fig, height=420, showlegend=True):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F7FF", family="Inter, Arial, sans-serif"),
        margin=dict(l=24, r=24, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if showlegend else None,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    return fig


def render_panel_title(title, caption_text=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption_text:
        st.markdown(f"<div class='caption'>{caption_text}</div>", unsafe_allow_html=True)


def make_download_df(df: pd.DataFrame):
    cols = [
        "Class", "Monthly_Revenue", "Estimated_Profit", "Estimated_Cost", "Net_Profit_Margin (%)",
        "Burn_Rate_Ratio", "Transaction_Count", "Repeat_Order_Rate (%)", "Digital_Adoption_Score",
        "Peak_Hour_Latency", "Location_Competitiveness", "Sentiment_Score",
        "Internal_Risk_Score", "External_Risk_Score", "Total_Risk_Score", "Risk_Category"
    ]
    return df[[c for c in cols if c in df.columns]].copy()

# ==========================================================
# DATA LOAD
# ==========================================================
base_df = load_data()
if base_df.empty:
    st.error("File synthetic_umkm_data.csv tidak ditemukan. Upload file CSV lewat sidebar untuk menjalankan dashboard.")
    uploaded_top = st.file_uploader("Upload dataset CSV", type=["csv"])
    if uploaded_top is not None:
        base_df = load_data(uploaded_top)
    else:
        st.stop()

filtered_df, risk_appetite, uploaded_file = filter_data(base_df)
if uploaded_file is not None:
    base_df = load_data(uploaded_file)
    filtered_df, risk_appetite, _ = filter_data(base_df, show_upload=False)

if filtered_df.empty:
    st.warning("Tidak ada data setelah filter. Longgarkan pilihan filter di sidebar.")
    st.stop()

b = base_metrics(filtered_df)
thresholds = thresholds_for_appetite(risk_appetite)

# ==========================================================
# HERO
# ==========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">UMKM Business Decision Stress Test Dashboard</div>
        <p class="hero-subtitle">
            Dashboard ini dirancang sebagai <b>decision cockpit</b> untuk mengevaluasi apakah strategi pertumbuhan UMKM tetap layak saat menghadapi ketidakpastian pasar, biaya, pelanggan, operasional, dan adopsi digital. Fokusnya bukan hanya melihat performa saat ini, tetapi menguji <b>robustness</b> keputusan bisnis sebelum implementasi.
        </p>
        <div class="pill-row">
            <span class="pill">Scenario Planning</span>
            <span class="pill">What-if Analysis</span>
            <span class="pill">Sensitivity Analysis</span>
            <span class="pill">Internal vs External Uncertainty</span>
            <span class="pill">Robust Decision</span>
            <span class="pill">Stress Testing</span>
            <span class="pill">Risk Mitigation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# TOP KPI
# ==========================================================
status_now = evaluate_status(b["profit"], b["margin"], b["burn"], thresholds, b["profit"])
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Observasi UMKM", number(len(filtered_df)), help="Jumlah baris observasi setelah filter.")
col2.metric("Total Monthly Revenue", rupiah(b["revenue"]))
col3.metric("Estimated Profit", rupiah(b["profit"]), delta=pct(b["margin"]))
col4.metric("Avg Burn Rate", f"{b['burn']:.2f}", delta=f"Tolerance {thresholds['max_burn']:.2f}")
col5.metric("Decision Status", status_now, delta=f"{risk_appetite} appetite")

st.markdown("")

# ==========================================================
# TABS
# ==========================================================
tabs = st.tabs([
    "1. Executive Cockpit",
    "2. Internal vs External Risk",
    "3. Scenario Planning",
    "4. What-if Simulator",
    "5. Sensitivity Analysis",
    "6. Robust Decision",
    "7. Stress Test & Mitigation",
    "8. Method & Data Dictionary",
])

# ==========================================================
# TAB 1: EXECUTIVE COCKPIT
# ==========================================================
with tabs[0]:
    left, right = st.columns([1.1, 0.9])
    with left:
        render_panel_title("Portfolio overview by business class", "Melihat komposisi class dan kontribusi revenue/profit untuk memahami posisi portofolio UMKM.")
        class_summary = filtered_df.groupby("Class", dropna=False).agg(
            Observations=("Class", "size"),
            Revenue=("Monthly_Revenue", "sum"),
            Profit=("Estimated_Profit", "sum"),
            Avg_Margin=("Net_Profit_Margin (%)", "mean"),
            Avg_Risk=("Total_Risk_Score", "mean"),
        ).reset_index().sort_values("Revenue", ascending=False)
        fig = px.bar(
            class_summary,
            x="Class",
            y=["Revenue", "Profit"],
            barmode="group",
            text_auto=".2s",
            color_discrete_sequence=COLOR_SEQ,
            title="Revenue vs profit by class",
        )
        fig.update_yaxes(title="Amount")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with right:
        render_panel_title("Class composition", "Komposisi ini membantu membaca apakah portofolio didominasi kelas Growth, Stable, Risky, atau Struggling.")
        count_summary = filtered_df["Class"].value_counts(dropna=False).reset_index()
        count_summary.columns = ["Class", "Count"]
        fig = px.pie(
            count_summary,
            names="Class",
            values="Count",
            hole=0.58,
            color_discrete_sequence=COLOR_SEQ,
            title="Share of observations",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_layout(fig, height=420, showlegend=False), use_container_width=True)

    col_a, col_b = st.columns([0.85, 1.15])
    with col_a:
        render_panel_title("Key profit drivers", "Korelasi sederhana terhadap estimated profit. Ini dipakai sebagai petunjuk awal, bukan bukti kausal.")
        numeric = filtered_df.select_dtypes(include=[np.number]).copy()
        corr = numeric.corr(numeric_only=True)["Estimated_Profit"].dropna().sort_values()
        corr = corr.drop(labels=["Estimated_Profit", "ID"], errors="ignore")
        driver_df = corr.abs().sort_values(ascending=False).head(10).reset_index()
        driver_df.columns = ["Variable", "Impact Strength"]
        fig = px.bar(
            driver_df.sort_values("Impact Strength", ascending=True),
            x="Impact Strength",
            y="Variable",
            orientation="h",
            color="Impact Strength",
            color_continuous_scale="Tealgrn",
            title="Top correlated variables",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart_layout(fig, height=440, showlegend=False), use_container_width=True)

    with col_b:
        render_panel_title("Revenue-margin business map", "Setiap titik adalah observasi UMKM. Kuadran kanan-atas menunjukkan revenue besar dan margin tinggi.")
        sample_n = min(len(filtered_df), 6000)
        sample_df = filtered_df.sample(sample_n, random_state=42) if len(filtered_df) > sample_n else filtered_df
        fig = px.scatter(
            sample_df,
            x="Monthly_Revenue",
            y="Net_Profit_Margin (%)",
            color="Class",
            size="Transaction_Count",
            hover_data=["Burn_Rate_Ratio", "Digital_Adoption_Score", "Repeat_Order_Rate (%)", "Location_Competitiveness"],
            color_discrete_sequence=COLOR_SEQ,
            title="Monthly revenue vs net profit margin",
        )
        fig.add_hline(y=thresholds["min_margin"], line_dash="dash", line_color="#F9C74F", annotation_text="Minimum tolerance")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)

    stage_summary = filtered_df.groupby("Business_Stage", observed=False).agg(
        Avg_Revenue=("Monthly_Revenue", "mean"),
        Avg_Profit=("Estimated_Profit", "mean"),
        Avg_Digital=("Digital_Adoption_Score", "mean"),
        Avg_Risk=("Total_Risk_Score", "mean"),
    ).reset_index()
    render_panel_title("Business maturity curve", "Karena dataset tidak memiliki kolom tanggal, grafik ini memakai tenure sebagai proksi tahap kematangan bisnis, bukan tren waktu kalender.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stage_summary["Business_Stage"].astype(str), y=stage_summary["Avg_Revenue"], mode="lines+markers", name="Avg revenue", yaxis="y1", line=dict(width=4, color="#00E5FF")))
    fig.add_trace(go.Scatter(x=stage_summary["Business_Stage"].astype(str), y=stage_summary["Avg_Profit"], mode="lines+markers", name="Avg profit", yaxis="y1", line=dict(width=4, color="#20E3B2")))
    fig.add_trace(go.Scatter(x=stage_summary["Business_Stage"].astype(str), y=stage_summary["Avg_Risk"], mode="lines+markers", name="Avg risk score", yaxis="y2", line=dict(width=3, color="#FF5A7A", dash="dot")))
    fig.update_layout(yaxis=dict(title="Revenue / profit"), yaxis2=dict(title="Risk score", overlaying="y", side="right", range=[0, 100]), title="Average performance by business tenure bucket")
    st.plotly_chart(chart_layout(fig, height=430), use_container_width=True)

# ==========================================================
# TAB 2: INTERNAL VS EXTERNAL RISK
# ==========================================================
with tabs[1]:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Uncertainty source diagnosis</div>
            <div class="caption">
                Bagian ini memisahkan risiko internal dan eksternal. Internal risk memakai indikator biaya, margin, latency, burn rate, dan kesiapan digital. External risk memakai indikator demand, kompetisi lokasi, rating, sentiment, dan volatilitas review.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Internal Risk Score", f"{b['internal_risk']:.1f}/100")
    c2.metric("External Risk Score", f"{b['external_risk']:.1f}/100")
    dominant = "External" if b["external_risk"] > b["internal_risk"] else "Internal"
    c3.metric("Dominant Uncertainty", dominant, delta=f"Gap {abs(b['external_risk']-b['internal_risk']):.1f}")

    left, right = st.columns([0.95, 1.05])
    with left:
        risk_source = pd.DataFrame({
            "Source": ["Internal", "External", "Total"],
            "Risk Score": [b["internal_risk"], b["external_risk"], b["total_risk"]]
        })
        fig = px.bar(risk_source, x="Source", y="Risk Score", color="Source", text="Risk Score", color_discrete_sequence=COLOR_SEQ, title="Internal vs external uncertainty score")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(chart_layout(fig, height=440, showlegend=False), use_container_width=True)

    with right:
        factor_df = pd.DataFrame({
            "Factor": [
                "Burn rate pressure", "Low margin", "Operational latency", "Low digital readiness", "Cost intensity",
                "Weak demand", "Location competition", "Low sentiment", "Low rating", "Review volatility"
            ],
            "Category": ["Internal"] * 5 + ["External"] * 5,
            "Score": [
                normalize(filtered_df["Burn_Rate_Ratio"]).mean(),
                normalize(filtered_df["Net_Profit_Margin (%)"], invert=True).mean(),
                normalize(filtered_df["Latency_Level"]).mean(),
                normalize(filtered_df["Digital_Adoption_Score"], invert=True).mean(),
                normalize(filtered_df["Estimated_Cost"] / filtered_df["Monthly_Revenue"].replace(0, np.nan)).mean(),
                normalize(filtered_df["Transaction_Count"], invert=True).mean(),
                normalize(filtered_df["Location_Competitiveness"]).mean(),
                normalize(filtered_df["Sentiment_Score"], invert=True).mean(),
                normalize(filtered_df["Avg_Historical_Rating"], invert=True).mean(),
                normalize(filtered_df["Review_Volatility"]).mean(),
            ]
        }).sort_values("Score", ascending=True)
        fig = px.bar(factor_df, x="Score", y="Factor", color="Category", orientation="h", text="Score", color_discrete_sequence=["#00E5FF", "#FF5A7A"], title="Detailed uncertainty drivers")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(chart_layout(fig, height=440), use_container_width=True)

    left2, right2 = st.columns([1.05, 0.95])
    with left2:
        fig = px.scatter(
            filtered_df.sample(min(len(filtered_df), 7000), random_state=7),
            x="External_Risk_Score",
            y="Internal_Risk_Score",
            color="Risk_Category",
            size="Monthly_Revenue",
            hover_data=["Class", "Net_Profit_Margin (%)", "Burn_Rate_Ratio", "Digital_Adoption_Score"],
            color_discrete_sequence=COLOR_SEQ,
            title="Risk quadrant: internal vs external exposure",
        )
        fig.add_vline(x=60, line_dash="dash", line_color="#F9C74F")
        fig.add_hline(y=60, line_dash="dash", line_color="#F9C74F")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)

    with right2:
        radar_data = pd.DataFrame({
            "Metric": ["Demand", "Competition", "Sentiment", "Rating", "Review Stability", "Digital Readiness", "Cost Control", "Margin Health", "Latency Control"],
            "Score": [
                100 - normalize(filtered_df["Transaction_Count"], invert=True).mean(),
                100 - normalize(filtered_df["Location_Competitiveness"]).mean(),
                100 - normalize(filtered_df["Sentiment_Score"], invert=True).mean(),
                100 - normalize(filtered_df["Avg_Historical_Rating"], invert=True).mean(),
                100 - normalize(filtered_df["Review_Volatility"]).mean(),
                100 - normalize(filtered_df["Digital_Adoption_Score"], invert=True).mean(),
                100 - normalize(filtered_df["Estimated_Cost"] / filtered_df["Monthly_Revenue"].replace(0, np.nan)).mean(),
                100 - normalize(filtered_df["Net_Profit_Margin (%)"], invert=True).mean(),
                100 - normalize(filtered_df["Latency_Level"]).mean(),
            ]
        })
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar_data["Score"], theta=radar_data["Metric"], fill="toself", name="Business resilience", line=dict(color="#20E3B2", width=3)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Business resilience radar")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)

# ==========================================================
# TAB 3: SCENARIO PLANNING
# ==========================================================
with tabs[2]:
    render_panel_title("Scenario planning matrix", "Skenario dibuat untuk menjawab: apakah keputusan pertumbuhan UMKM tetap beneficial pada kondisi best, base, dan worst?")
    scenario_matrix = pd.DataFrame({
        "Scenario": ["Best Case", "Base Case", "Worst Case"],
        "Market Demand": ["+30%", "+10%", "-20%"],
        "Revenue per Transaction": ["+10%", "0%", "-8%"],
        "Cost Shock": ["-5%", "0%", "+20%"],
        "Repeat Order": ["+8 pts", "+3 pts", "-10 pts"],
        "Digital Adoption": ["+1.0", "+0.4", "-0.8"],
        "External Pressure": ["Low", "Normal", "High"],
    })
    st.dataframe(scenario_matrix, use_container_width=True, hide_index=True)

    scenarios = {
        "Best Case": scenario_calculation(b, 0.30, 0.10, -0.05, margin_chg=3.0, repeat_chg=8.0, digital_chg=1.0, burn_chg=-0.08),
        "Base Case": scenario_calculation(b, 0.10, 0.00, 0.00, margin_chg=1.0, repeat_chg=3.0, digital_chg=0.4, burn_chg=0.00),
        "Worst Case": scenario_calculation(b, -0.20, -0.08, 0.20, margin_chg=-4.0, repeat_chg=-10.0, digital_chg=-0.8, burn_chg=0.15),
    }
    scenario_df = pd.DataFrame(scenarios).T.reset_index().rename(columns={"index": "Scenario"})
    scenario_df["Status"] = scenario_df.apply(lambda r: evaluate_status(r["Profit"], r["Margin (%)"], r["Burn Rate"], thresholds, b["profit"]), axis=1)

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        fig = px.bar(
            scenario_df,
            x="Scenario",
            y=["Revenue", "Cost", "Profit"],
            barmode="group",
            text_auto=".2s",
            color_discrete_sequence=COLOR_SEQ,
            title="KPI impact across scenarios",
        )
        fig.update_yaxes(title="Amount")
        st.plotly_chart(chart_layout(fig, height=470), use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=float(scenario_df.loc[scenario_df["Scenario"] == "Worst Case", "Profit"].iloc[0] / max(abs(b["profit"]), 1) * 100),
            title={"text": "Worst-case profit vs current"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [-150, 150]},
                "bar": {"color": "#FF5A7A"},
                "steps": [
                    {"range": [-150, 0], "color": "rgba(255,90,122,0.28)"},
                    {"range": [0, 75], "color": "rgba(249,199,79,0.25)"},
                    {"range": [75, 150], "color": "rgba(32,227,178,0.25)"},
                ],
            },
        ))
        st.plotly_chart(chart_layout(fig, height=470, showlegend=False), use_container_width=True)

    display_df = scenario_df.copy()
    for c in ["Revenue", "Cost", "Profit"]:
        display_df[c] = display_df[c].apply(rupiah)
    display_df["Margin (%)"] = display_df["Margin (%)"].apply(pct)
    display_df["Repeat Order (%)"] = display_df["Repeat Order (%)"].apply(pct)
    display_df["Burn Rate"] = display_df["Burn Rate"].map(lambda x: f"{x:.2f}")
    display_df["Digital Score"] = display_df["Digital Score"].map(lambda x: f"{x:.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==========================================================
# TAB 4: WHAT-IF SIMULATOR
# ==========================================================
with tabs[3]:
    render_panel_title("Interactive what-if simulator", "Ubah asumsi bisnis dan lihat dampaknya terhadap revenue, cost, profit, margin, burn rate, dan feasibility.")
    s1, s2, s3 = st.columns(3)
    with s1:
        demand_change = st.slider("Demand / transaction change (%)", -50, 60, -10, step=5) / 100
        rpu_change = st.slider("Revenue per transaction change (%)", -30, 40, 0, step=5) / 100
    with s2:
        cost_change = st.slider("Operational cost escalation (%)", -30, 60, 10, step=5) / 100
        margin_change = st.slider("Margin efficiency change (percentage point)", -15, 15, 0, step=1)
    with s3:
        repeat_change = st.slider("Repeat order change (percentage point)", -25, 25, -5, step=1)
        digital_change = st.slider("Digital adoption score change", -3.0, 3.0, 0.5, step=0.1)
        burn_change = st.slider("Burn rate change (%)", -30, 50, 10, step=5) / 100

    result = scenario_calculation(b, demand_change, rpu_change, cost_change, margin_chg=margin_change, repeat_chg=repeat_change, digital_chg=digital_change, burn_chg=burn_change)
    whatif_status = evaluate_status(result["Profit"], result["Margin (%)"], result["Burn Rate"], thresholds, b["profit"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("New Revenue", rupiah(result["Revenue"]), delta=rupiah(result["Revenue"] - b["revenue"]))
    k2.metric("New Profit", rupiah(result["Profit"]), delta=rupiah(result["Profit"] - b["profit"]))
    k3.metric("New Margin", pct(result["Margin (%)"]), delta=f"vs tolerance {thresholds['min_margin']:.1f}%")
    k4.markdown(f"<div class='panel'><div class='section-title'>Feasibility Status</div>{status_badge(whatif_status)}<div class='small-note' style='margin-top:10px;'>Based on selected risk appetite: <b>{risk_appetite}</b>.</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Waterfall(
            name="Profit bridge",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Current Profit", "Demand/RPU Effect", "Cost Effect", "Margin Effect", "Simulated Profit"],
            textposition="outside",
            y=[b["profit"], result["Revenue"] - b["revenue"], -(result["Cost"] - b["cost"]), result["Profit"] - (result["Revenue"] - result["Cost"]), result["Profit"]],
            connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        ))
        fig.update_layout(title="Profit bridge from current to simulated condition", yaxis_title="Profit impact")
        st.plotly_chart(chart_layout(fig, height=480, showlegend=False), use_container_width=True)
    with c2:
        bullet_df = pd.DataFrame({
            "Metric": ["Margin", "Burn Rate", "Repeat Order", "Digital Score"],
            "Value": [result["Margin (%)"], result["Burn Rate"] * 20, result["Repeat Order (%)"], result["Digital Score"] * 10],
            "Target": [thresholds["min_margin"], thresholds["max_burn"] * 20, b["repeat"], b["digital"] * 10]
        })
        fig = px.bar(bullet_df, y="Metric", x="Value", orientation="h", color="Metric", text="Value", color_discrete_sequence=COLOR_SEQ, title="Operational health under what-if")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_xaxes(range=[min(-20, bullet_df["Value"].min() - 5), max(100, bullet_df["Value"].max() + 10)])
        st.plotly_chart(chart_layout(fig, height=480, showlegend=False), use_container_width=True)

# ==========================================================
# TAB 5: SENSITIVITY ANALYSIS
# ==========================================================
with tabs[4]:
    render_panel_title("One-way sensitivity analysis", "Setiap variabel diubah satu per satu; variabel lain tetap. Tujuannya menemukan asumsi paling kritis terhadap profit.")
    sensitivity_level = st.slider("Sensitivity shock size (%)", 5, 40, 20, step=5) / 100
    sensitivity_vars = {
        "Market demand": ("demand", sensitivity_level),
        "Revenue per transaction": ("rpu", sensitivity_level),
        "Operational cost": ("cost", sensitivity_level),
        "Margin efficiency": ("margin", sensitivity_level * 100),
        "Repeat order": ("repeat", sensitivity_level * 100),
        "Burn rate": ("burn", sensitivity_level),
        "Digital adoption": ("digital", sensitivity_level * 10),
    }
    rows = []
    for label, (kind, delta) in sensitivity_vars.items():
        kwargs_low = dict(demand_chg=0, rpu_chg=0, cost_chg=0, margin_chg=0, repeat_chg=0, digital_chg=0, burn_chg=0)
        kwargs_high = kwargs_low.copy()
        if kind == "demand":
            kwargs_low["demand_chg"] = -delta; kwargs_high["demand_chg"] = delta
        elif kind == "rpu":
            kwargs_low["rpu_chg"] = -delta; kwargs_high["rpu_chg"] = delta
        elif kind == "cost":
            kwargs_low["cost_chg"] = delta; kwargs_high["cost_chg"] = -delta
        elif kind == "margin":
            kwargs_low["margin_chg"] = -delta; kwargs_high["margin_chg"] = delta
        elif kind == "repeat":
            # repeat order influences demand in this simulator.
            kwargs_low["demand_chg"] = -delta / 200; kwargs_high["demand_chg"] = delta / 200
            kwargs_low["repeat_chg"] = -delta; kwargs_high["repeat_chg"] = delta
        elif kind == "burn":
            kwargs_low["burn_chg"] = delta; kwargs_high["burn_chg"] = -delta
            kwargs_low["cost_chg"] = delta / 2; kwargs_high["cost_chg"] = -delta / 2
        elif kind == "digital":
            kwargs_low["digital_chg"] = -delta; kwargs_high["digital_chg"] = delta
            kwargs_low["cost_chg"] = delta / 100; kwargs_high["cost_chg"] = -delta / 150
            kwargs_low["rpu_chg"] = -delta / 200; kwargs_high["rpu_chg"] = delta / 200
        low = scenario_calculation(b, **kwargs_low)["Profit"]
        high = scenario_calculation(b, **kwargs_high)["Profit"]
        rows.append({"Variable": label, "Downside Profit": low, "Upside Profit": high, "Swing": high - low})
    sens_df = pd.DataFrame(rows).sort_values("Swing", ascending=True)

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sens_df["Variable"], x=sens_df["Downside Profit"] - b["profit"], orientation="h", name="Downside", marker_color="#FF5A7A"))
        fig.add_trace(go.Bar(y=sens_df["Variable"], x=sens_df["Upside Profit"] - b["profit"], orientation="h", name="Upside", marker_color="#20E3B2"))
        fig.update_layout(title="Tornado chart: profit sensitivity", xaxis_title="Profit deviation from current", barmode="overlay")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)
    with c2:
        top_var = sens_df.sort_values("Swing", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="panel">
                <div class="section-title">Most critical assumption</div>
                <div style="font-size:2rem;font-weight:850;color:#00E5FF;margin:8px 0;">{top_var['Variable']}</div>
                <div class="caption">Perubahan variabel ini menghasilkan rentang dampak profit terbesar pada ukuran shock yang dipilih.</div>
                <div class="small-note">Profit swing: <b>{rupiah(top_var['Swing'])}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        display_sens = sens_df.sort_values("Swing", ascending=False).copy()
        for c in ["Downside Profit", "Upside Profit", "Swing"]:
            display_sens[c] = display_sens[c].apply(rupiah)
        st.dataframe(display_sens, use_container_width=True, hide_index=True)

    render_panel_title("Multi-way sensitivity heatmap", "Demand dan cost diubah bersamaan untuk melihat kombinasi shock yang paling berbahaya.")
    demand_range = np.arange(-40, 41, 10) / 100
    cost_range = np.arange(-20, 51, 10) / 100
    heat_rows = []
    for d in demand_range:
        for c in cost_range:
            res = scenario_calculation(b, d, 0, c)
            heat_rows.append({"Demand Shock": f"{d:+.0%}", "Cost Shock": f"{c:+.0%}", "Profit Margin": res["Margin (%)"]})
    heat_df = pd.DataFrame(heat_rows)
    pivot = heat_df.pivot(index="Cost Shock", columns="Demand Shock", values="Profit Margin")
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn", title="Profit margin under combined demand and cost shock")
    st.plotly_chart(chart_layout(fig, height=500, showlegend=False), use_container_width=True)

# ==========================================================
# TAB 6: ROBUST DECISION
# ==========================================================
with tabs[5]:
    render_panel_title("Robust decision analysis", "Membandingkan beberapa policy. Keputusan yang paling robust bukan selalu yang profit tertinggi, tetapi yang tetap layak pada banyak skenario.")
    policies = {
        "Aggressive Digital Growth": {"demand": 0.18, "rpu": 0.05, "cost": 0.18, "margin": 1.0, "repeat": 3.0, "digital": 1.2, "burn": 0.10},
        "Cost Discipline": {"demand": -0.02, "rpu": 0.00, "cost": -0.12, "margin": 2.0, "repeat": 0.5, "digital": 0.2, "burn": -0.08},
        "Retention Focus": {"demand": 0.08, "rpu": 0.03, "cost": 0.06, "margin": 1.2, "repeat": 8.0, "digital": 0.4, "burn": 0.00},
        "Digital Operations": {"demand": 0.07, "rpu": 0.04, "cost": -0.03, "margin": 1.5, "repeat": 3.0, "digital": 1.5, "burn": -0.04},
        "Balanced Robust Strategy": {"demand": 0.10, "rpu": 0.03, "cost": 0.02, "margin": 1.5, "repeat": 5.0, "digital": 0.8, "burn": -0.02},
    }
    scenario_mods = {
        "Best": {"demand": 0.30, "rpu": 0.10, "cost": -0.05, "margin": 3.0, "repeat": 8.0, "digital": 1.0, "burn": -0.08},
        "Base": {"demand": 0.10, "rpu": 0.00, "cost": 0.00, "margin": 1.0, "repeat": 3.0, "digital": 0.4, "burn": 0.00},
        "Worst": {"demand": -0.20, "rpu": -0.08, "cost": 0.20, "margin": -4.0, "repeat": -10.0, "digital": -0.8, "burn": 0.15},
    }
    policy_rows = []
    for pol, pmod in policies.items():
        profits = []
        statuses = []
        for sc, smod in scenario_mods.items():
            res = scenario_calculation(
                b,
                smod["demand"] + pmod["demand"],
                smod["rpu"] + pmod["rpu"],
                smod["cost"] + pmod["cost"],
                margin_chg=smod["margin"] + pmod["margin"],
                repeat_chg=smod["repeat"] + pmod["repeat"],
                digital_chg=smod["digital"] + pmod["digital"],
                burn_chg=smod["burn"] + pmod["burn"],
            )
            profits.append(res["Profit"])
            statuses.append(evaluate_status(res["Profit"], res["Margin (%)"], res["Burn Rate"], thresholds, b["profit"]))
            policy_rows.append({"Policy": pol, "Scenario": sc, "Profit": res["Profit"], "Margin (%)": res["Margin (%)"], "Status": statuses[-1]})
        avg_profit = float(np.mean(profits))
        worst_profit = float(np.min(profits))
        stability_penalty = float(np.std(profits))
        negative_penalty = abs(worst_profit) * 0.35 if worst_profit < 0 else 0
        status_penalty = statuses.count("Critical") * abs(b["profit"]) * 0.28 + statuses.count("Watch") * abs(b["profit"]) * 0.08
        robust_score = avg_profit - 0.45 * stability_penalty + 0.85 * worst_profit - negative_penalty - status_penalty
        policy_rows.append({"Policy": pol, "Scenario": "Average", "Profit": avg_profit, "Margin (%)": np.nan, "Status": "-"})
        policy_rows.append({"Policy": pol, "Scenario": "Worst", "Profit": worst_profit, "Margin (%)": np.nan, "Status": "-"})
        policies[pol]["Average Profit"] = avg_profit
        policies[pol]["Worst Profit"] = worst_profit
        policies[pol]["Stability"] = stability_penalty
        policies[pol]["Robust Score"] = robust_score

    policy_eval = pd.DataFrame([
        {"Policy": k, "Average Profit": v["Average Profit"], "Worst Profit": v["Worst Profit"], "Stability Risk": v["Stability"], "Robust Score": v["Robust Score"]}
        for k, v in policies.items()
    ]).sort_values("Robust Score", ascending=False)

    best_policy = policy_eval.iloc[0]["Policy"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended Policy", best_policy)
    k2.metric("Best Robust Score", rupiah(policy_eval.iloc[0]["Robust Score"]))
    k3.metric("Worst-case Profit", rupiah(policy_eval.iloc[0]["Worst Profit"]))

    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        detailed_policy = pd.DataFrame([r for r in policy_rows if r["Scenario"] in ["Best", "Base", "Worst"]])
        fig = px.bar(detailed_policy, x="Policy", y="Profit", color="Scenario", barmode="group", text_auto=".2s", color_discrete_sequence=COLOR_SEQ, title="Policy profit across scenarios")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)
    with c2:
        fig = px.scatter(
            policy_eval,
            x="Worst Profit",
            y="Average Profit",
            size="Robust Score",
            color="Policy",
            hover_data=["Stability Risk", "Robust Score"],
            color_discrete_sequence=COLOR_SEQ,
            title="Risk-return map of policy options",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#FF5A7A")
        st.plotly_chart(chart_layout(fig, height=500), use_container_width=True)

    display_policy = policy_eval.copy()
    for c in ["Average Profit", "Worst Profit", "Stability Risk", "Robust Score"]:
        display_policy[c] = display_policy[c].apply(rupiah)
    st.dataframe(display_policy, use_container_width=True, hide_index=True)

# ==========================================================
# TAB 7: STRESS TEST & MITIGATION
# ==========================================================
with tabs[6]:
    render_panel_title("Stress testing decisions", "Stress test menguji keputusan pada kondisi ekstrem: demand collapse, cost inflation, competition shock, sentiment crash, dan digital failure.")
    stress_events = {
        "Normal Scenario": {"demand": 0.00, "rpu": 0.00, "cost": 0.00, "margin": 0.0, "repeat": 0.0, "digital": 0.0, "burn": 0.0},
        "Moderate Shock": {"demand": -0.12, "rpu": -0.04, "cost": 0.10, "margin": -2.0, "repeat": -4.0, "digital": -0.3, "burn": 0.06},
        "Severe Shock": {"demand": -0.28, "rpu": -0.08, "cost": 0.22, "margin": -5.0, "repeat": -9.0, "digital": -0.8, "burn": 0.15},
        "Extreme Shock": {"demand": -0.42, "rpu": -0.12, "cost": 0.35, "margin": -9.0, "repeat": -15.0, "digital": -1.5, "burn": 0.28},
        "Digital Failure": {"demand": -0.18, "rpu": -0.06, "cost": 0.08, "margin": -3.0, "repeat": -6.0, "digital": -2.5, "burn": 0.10},
        "Competition Price War": {"demand": -0.20, "rpu": -0.18, "cost": 0.05, "margin": -6.0, "repeat": -5.0, "digital": 0.0, "burn": 0.08},
    }
    stress_rows = []
    for event, m in stress_events.items():
        res = scenario_calculation(b, m["demand"], m["rpu"], m["cost"], margin_chg=m["margin"], repeat_chg=m["repeat"], digital_chg=m["digital"], burn_chg=m["burn"])
        status = evaluate_status(res["Profit"], res["Margin (%)"], res["Burn Rate"], thresholds, b["profit"])
        stress_rows.append({"Stress Event": event, **res, "Status": status})
    stress_df = pd.DataFrame(stress_rows)

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        fig = px.bar(stress_df, x="Stress Event", y="Profit", color="Status", text_auto=".2s", color_discrete_map={"Safe": "#20E3B2", "Watch": "#F9C74F", "Critical": "#FF5A7A"}, title="Profit under stress events")
        fig.add_hline(y=0, line_dash="dash", line_color="#FF5A7A")
        st.plotly_chart(chart_layout(fig, height=480), use_container_width=True)
    with c2:
        heat = stress_df.set_index("Stress Event")[["Revenue", "Cost", "Profit", "Margin (%)", "Repeat Order (%)", "Burn Rate", "Digital Score"]].copy()
        # Normalize columns for comparability in heatmap.
        heat_norm = heat.copy()
        for col in heat_norm.columns:
            col_s = pd.to_numeric(heat_norm[col], errors="coerce")
            if col_s.max() != col_s.min():
                heat_norm[col] = (col_s - col_s.min()) / (col_s.max() - col_s.min()) * 100
            else:
                heat_norm[col] = 50
        fig = px.imshow(heat_norm, aspect="auto", color_continuous_scale="RdYlGn", title="Stress test normalized KPI heatmap")
        st.plotly_chart(chart_layout(fig, height=480, showlegend=False), use_container_width=True)

    mitigation = []
    for _, r in stress_df.iterrows():
        event = r["Stress Event"]
        if r["Status"] == "Safe":
            action = "Maintain decision policy; monitor demand, burn rate, and customer preference signals."
            priority = "Low"
        elif event in ["Extreme Shock", "Severe Shock"] or r["Profit"] < 0:
            action = "Pause expansion, protect cash runway, cut non-essential costs, renegotiate fixed costs, and shift to retention-first campaign."
            priority = "Critical"
        elif event == "Digital Failure":
            action = "Prioritize checkout reliability, payment stability, response time, and operational automation before increasing marketing spend."
            priority = "High"
        elif event == "Competition Price War":
            action = "Avoid blind discounting; use bundle offers, loyalty rewards, niche positioning, and margin-based pricing guardrails."
            priority = "High"
        else:
            action = "Reduce campaign inefficiency, monitor CAC proxy, intensify repeat-order strategy, and prepare cost-control trigger."
            priority = "Medium"
        mitigation.append({"Stress Event": event, "Status": r["Status"], "Priority": priority, "Mitigation Action": action})
    mitigation_df = pd.DataFrame(mitigation)
    st.dataframe(mitigation_df, use_container_width=True, hide_index=True)

    display_stress = stress_df.copy()
    for c in ["Revenue", "Cost", "Profit"]:
        display_stress[c] = display_stress[c].apply(rupiah)
    for c in ["Margin (%)", "Repeat Order (%)"]:
        display_stress[c] = display_stress[c].apply(pct)
    display_stress["Burn Rate"] = display_stress["Burn Rate"].map(lambda x: f"{x:.2f}")
    display_stress["Digital Score"] = display_stress["Digital Score"].map(lambda x: f"{x:.2f}")
    st.dataframe(display_stress, use_container_width=True, hide_index=True)

# ==========================================================
# TAB 8: METHOD & DATA DICTIONARY
# ==========================================================
with tabs[7]:
    render_panel_title("Dashboard method", "Bagian ini menjelaskan bagaimana materi decision under risk diterapkan ke dataset UMKM.")
    method = pd.DataFrame({
        "Materi": [
            "Business Decision Pipeline", "Uncertainty Source", "Scenario Planning", "What-if Analysis",
            "Sensitivity Analysis", "Robust Decision", "Risk Appetite & Tolerance", "Stress Testing", "Mitigation Plan"
        ],
        "Implementasi di Dashboard": [
            "Decision cockpit menampilkan KPI, constraints, policy, dan risk evaluation.",
            "Risiko dipisah menjadi internal dan eksternal.",
            "Best, base, dan worst case dibuat dari perubahan demand, cost, margin, repeat order, dan digital adoption.",
            "Slider interaktif mengubah asumsi input dan menghitung ulang KPI.",
            "Tornado chart dan heatmap mengidentifikasi variabel paling kritis.",
            "Policy bisnis dibandingkan berdasarkan average profit, worst profit, stability, dan robust score.",
            "Risk appetite menentukan batas margin, burn rate, dan toleransi penurunan profit.",
            "Stress event ekstrem menguji feasibility keputusan.",
            "Rule-based recommendation diberikan untuk tiap kondisi risiko."
        ]
    })
    st.dataframe(method, use_container_width=True, hide_index=True)

    st.markdown("### Data dictionary and derived variables")
    dictionary = pd.DataFrame({
        "Variable": [
            "Monthly_Revenue", "Net_Profit_Margin (%)", "Burn_Rate_Ratio", "Transaction_Count", "Avg_Historical_Rating",
            "Review_Volatility", "Repeat_Order_Rate (%)", "Digital_Adoption_Score", "Peak_Hour_Latency", "Location_Competitiveness",
            "Sentiment_Score", "Estimated_Profit", "Estimated_Cost", "Churn_Proxy (%)", "Internal_Risk_Score", "External_Risk_Score"
        ],
        "Business Meaning": [
            "Pendapatan bulanan observasi UMKM.",
            "Margin laba bersih dalam persen.",
            "Tekanan pembakaran biaya terhadap kemampuan bisnis.",
            "Jumlah transaksi; dipakai sebagai proxy demand.",
            "Rata-rata rating historis pelanggan.",
            "Ketidakstabilan review pelanggan.",
            "Persentase pelanggan yang melakukan pembelian ulang.",
            "Tingkat adopsi digital UMKM.",
            "Latency operasional saat jam sibuk.",
            "Tingkat kompetisi lokasi.",
            "Skor sentimen review pelanggan.",
            "Monthly_Revenue × Net_Profit_Margin / 100.",
            "Monthly_Revenue - Estimated_Profit.",
            "100 - Repeat_Order_Rate; proxy churn karena tidak ada kolom churn eksplisit.",
            "Gabungan burn rate, margin, latency, digital readiness, dan cost intensity.",
            "Gabungan demand risk, competition, sentiment, rating, dan review volatility."
        ],
        "Decision Use": [
            "KPI utama", "Profitability", "Risk tolerance", "Market demand", "Customer preference",
            "Customer uncertainty", "Retention", "Technology readiness", "Operational capacity", "Competition",
            "Customer preference", "KPI output", "Cost proxy", "Customer churn proxy", "Internal uncertainty", "External uncertainty"
        ]
    })
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

    st.markdown("### Export analysis dataset")
    export_df = make_download_df(filtered_df)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered analysis data",
        data=csv,
        file_name="umkm_filtered_analysis_data.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Important interpretation note</div>
            <div class="caption">
                Dataset tidak memiliki kolom biaya aktual dan churn aktual. Karena itu, dashboard membuat proxy Estimated_Cost dan Churn_Proxy dari variabel yang tersedia. Interpretasi dashboard sebaiknya dibaca sebagai simulasi keputusan bisnis dan stress test, bukan audit akuntansi final.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
