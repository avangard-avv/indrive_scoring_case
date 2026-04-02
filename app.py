"""
Phase 5: Streamlit Dashboard — Merit-based Subsidy Scoring Engine.

Interactive UI for government officials to review ranked applicants,
understand score explanations, and approve/reject applications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_ingestion import load_and_process
from src.enrichment import enrich_applicants, YEARS
from src.model import (
    prepare_training_data,
    train_model,
    compute_merit_scores,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.explainer import explain_all_applicants


st.set_page_config(
    page_title="AgroScore AI — Merit Scoring Engine",
    page_icon="🌾",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner="Downloading subsidy registry data...")
def load_data():
    """Cache the data download and processing."""
    transactions, applicants = load_and_process()
    enriched = enrich_applicants(applicants)
    return transactions, enriched


@st.cache_resource(show_spinner="Training scoring model...")
def train_scoring_model(_transactions, _enriched):
    """Cache the trained model."""
    train_df = prepare_training_data(_transactions, _enriched)
    model, metrics = train_model(train_df, iterations=300)
    scores = compute_merit_scores(model, _enriched, _transactions)
    return model, metrics, scores


@st.cache_data(show_spinner="Generating SHAP explanations...")
def generate_explanations(_model, _scores, _transactions):
    """Cache SHAP explanations for all applicants."""
    return explain_all_applicants(
        _model, _scores, _transactions, ALL_FEATURES, CATEGORICAL_FEATURES
    )


def render_header():
    st.title("🌾 AgroScore AI")
    st.markdown(
        "**Merit-based Subsidy Scoring Engine** — "
        "Ranking agricultural subsidy applications by efficiency, reliability, "
        "and farm performance."
    )
    st.markdown("---")


def render_metrics_panel(metrics: dict, scores: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Applicants", f"{len(scores):,}")
    col2.metric("Model AUC-ROC", f"{metrics['auc_roc']:.3f}")
    col3.metric("Avg Merit Score", f"{scores['merit_score'].mean():.1f}")
    col4.metric("Median Score", f"{scores['merit_score'].median():.1f}")
    col5.metric("Score Std Dev", f"{scores['merit_score'].std():.1f}")


def render_score_distribution(scores: pd.DataFrame):
    fig = px.histogram(
        scores,
        x="merit_score",
        nbins=50,
        title="Merit Score Distribution",
        labels={"merit_score": "Merit Score (0-100)"},
        color_discrete_sequence=["#2E86AB"],
    )
    fig.update_layout(yaxis_title="Number of Applicants", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_region_analysis(scores: pd.DataFrame):
    region_stats = scores.groupby("primary_region").agg(
        avg_score=("merit_score", "mean"),
        count=("bin_iin", "count"),
    ).reset_index().sort_values("avg_score", ascending=True)

    fig = px.bar(
        region_stats,
        y="primary_region",
        x="avg_score",
        orientation="h",
        title="Average Merit Score by Region",
        labels={"primary_region": "Region", "avg_score": "Avg Score"},
        color="avg_score",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_applicant_table(scores: pd.DataFrame):
    st.subheader("📋 Ranked Applicant List")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_score, max_score = st.slider(
            "Merit Score Range", 0.0, 100.0, (0.0, 100.0), step=1.0,
        )
    with col2:
        regions = ["All"] + sorted(scores["primary_region"].dropna().unique().tolist())
        selected_region = st.selectbox("Region", regions)
    with col3:
        search_bin = st.text_input("Search by BIN/IIN", "")
    with col4:
        show_bankrupt = st.checkbox("Show bankrupt only", value=False)

    filtered = scores[
        (scores["merit_score"] >= min_score) &
        (scores["merit_score"] <= max_score)
    ]
    if selected_region != "All":
        filtered = filtered[filtered["primary_region"] == selected_region]
    if search_bin:
        filtered = filtered[filtered["bin_iin"].str.contains(search_bin, na=False)]
    if show_bankrupt:
        filtered = filtered[filtered["is_bankrupt"] == True]

    display_cols = [
        "bin_iin", "merit_score", "approval_rate", "total_applications",
        "total_requested", "primary_region", "primary_direction",
        "large_livestock_latest", "small_livestock_latest",
        "milk_production_latest", "annual_revenue",
        "is_bankrupt", "has_tax_debt", "active_lawsuits",
        "credit_score", "ndvi_score",
    ]
    st.dataframe(
        filtered[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "bin_iin": st.column_config.TextColumn("BIN/IIN"),
            "merit_score": st.column_config.ProgressColumn(
                "Merit Score", min_value=0, max_value=100, format="%.1f"
            ),
            "approval_rate": st.column_config.NumberColumn("Approval %", format="%.0%%"),
            "total_applications": st.column_config.NumberColumn("Apps"),
            "total_requested": st.column_config.NumberColumn("Total (₸)", format="₸%,.0f"),
            "primary_region": st.column_config.TextColumn("Region"),
            "primary_direction": st.column_config.TextColumn("Direction"),
            "large_livestock_latest": st.column_config.NumberColumn("Large Livestock"),
            "small_livestock_latest": st.column_config.NumberColumn("Small Livestock"),
            "milk_production_latest": st.column_config.NumberColumn("Milk (t)", format="%.1f"),
            "annual_revenue": st.column_config.NumberColumn("Revenue (₸)", format="₸%,.0f"),
            "is_bankrupt": st.column_config.CheckboxColumn("Bankrupt"),
            "has_tax_debt": st.column_config.CheckboxColumn("Tax Debt"),
            "active_lawsuits": st.column_config.NumberColumn("Lawsuits"),
            "credit_score": st.column_config.NumberColumn("Credit"),
            "ndvi_score": st.column_config.NumberColumn("NDVI", format="%.3f"),
        },
    )
    return filtered


def render_livestock_chart(row: pd.Series):
    """Render livestock and milk production trends for an applicant."""
    large_cols = [f"large_livestock_{y}" for y in YEARS]
    small_cols = [f"small_livestock_{y}" for y in YEARS]
    milk_cols = [f"milk_production_{y}" for y in YEARS]

    # Check if columns exist
    if not all(c in row.index for c in large_cols):
        return

    col1, col2 = st.columns(2)

    with col1:
        livestock_df = pd.DataFrame({
            "Year": YEARS * 2,
            "Count": [row[c] for c in large_cols] + [row[c] for c in small_cols],
            "Type": ["Cows/Horses"] * len(YEARS) + ["Sheep/Goats"] * len(YEARS),
        })
        fig = px.line(
            livestock_df, x="Year", y="Count", color="Type",
            title="Livestock Count (2021-2025)",
            markers=True,
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        milk_df = pd.DataFrame({
            "Year": YEARS,
            "Production (tons)": [row[c] for c in milk_cols],
        })
        fig = px.bar(
            milk_df, x="Year", y="Production (tons)",
            title="Milk Production (2021-2025)",
            color_discrete_sequence=["#2E86AB"],
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_enrichment_breakdown(enrichment_exps: list[dict]):
    """Render the enrichment score component breakdown."""
    if not enrichment_exps:
        return

    # Stacked horizontal bar showing component scores
    comp_df = pd.DataFrame(enrichment_exps)
    fig = go.Figure()
    for _, comp_row in comp_df.iterrows():
        color = "#2E86AB" if comp_row["direction"] == "positive" else "#E8553D"
        fig.add_trace(go.Bar(
            y=[comp_row["component"]],
            x=[comp_row["score"]],
            orientation="h",
            name=comp_row["component"],
            marker_color=color,
            text=f"{comp_row['score']}/{comp_row['max_score']}",
            textposition="auto",
            showlegend=False,
        ))
        # Add max indicator
        fig.add_trace(go.Bar(
            y=[comp_row["component"]],
            x=[comp_row["max_score"] - comp_row["score"]],
            orientation="h",
            marker_color="rgba(200,200,200,0.3)",
            showlegend=False,
        ))

    fig.update_layout(
        barmode="stack",
        title="Enrichment Score Components",
        xaxis_title="Points",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Text explanations
    for comp in enrichment_exps:
        icon = "🟢" if comp["direction"] == "positive" else "🔴"
        st.markdown(
            f"{icon} **{comp['component']}** ({comp['score']}/{comp['max_score']}): "
            f"{comp['text']}"
        )


def render_applicant_detail(scores: pd.DataFrame, explanations: dict):
    st.subheader("🔍 Applicant Detail — 'Why This Score?'")

    applicant_options = scores["bin_iin"].tolist()
    selected = st.selectbox("Select applicant (BIN/IIN)", applicant_options, index=0)

    if not selected:
        return

    row = scores[scores["bin_iin"] == selected].iloc[0]

    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Merit Score", f"{row['merit_score']:.1f} / 100")
    col2.metric("ML Score", f"{row.get('ml_percentile', 0):.1f}")
    col3.metric("Enrichment Score", f"{row.get('enrichment_score', 0):.1f}")
    col4.metric("Approval Rate", f"{row['approval_rate']:.0%}")
    col5.metric("Applications", int(row["total_applications"]))

    # Financial & legal row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Revenue", f"₸{row['annual_revenue']:,.0f}")
    col2.metric("Credit Score", int(row["credit_score"]))
    col3.metric("Bankrupt", "YES ⛔" if row["is_bankrupt"] else "No ✅")
    col4.metric("Tax Debt", "YES ⚠️" if row["has_tax_debt"] else "No ✅")
    col5.metric("Lawsuits", int(row["active_lawsuits"]))
    col6.metric("NDVI", f"{row['ndvi_score']:.3f}")

    st.markdown("---")

    # Livestock trends
    st.markdown("#### 🐄 Livestock & Production Trends")
    render_livestock_chart(row)

    # Trend indicators
    col1, col2, col3 = st.columns(3)
    large_trend = row.get("livestock_large_trend", 0)
    small_trend = row.get("livestock_small_trend", 0)
    milk_t = row.get("milk_trend", 0)
    col1.metric(
        "Large Livestock Trend",
        f"{large_trend:+.1%}",
        delta=f"{'Growing' if large_trend > 0 else 'Declining'}",
        delta_color="normal" if large_trend >= 0 else "inverse",
    )
    col2.metric(
        "Small Livestock Trend",
        f"{small_trend:+.1%}",
        delta=f"{'Growing' if small_trend > 0 else 'Declining'}",
        delta_color="normal" if small_trend >= 0 else "inverse",
    )
    col3.metric(
        "Milk Production Trend",
        f"{milk_t:+.1%}",
        delta=f"{'Increasing' if milk_t > 0 else 'Decreasing'}",
        delta_color="normal" if milk_t >= 0 else "inverse",
    )

    st.markdown("---")

    # Explanations
    if selected in explanations:
        exp = explanations[selected]

        st.markdown("#### 📊 Enrichment Score Breakdown (60% of Merit Score)")
        render_enrichment_breakdown(exp.get("enrichment", []))

        st.markdown("---")
        st.markdown("#### 🤖 ML Model Factors — SHAP (40% of Merit Score)")

        shap_exps = exp.get("shap", [])
        if shap_exps:
            exp_df = pd.DataFrame(shap_exps)
            exp_df["signed_impact"] = exp_df["impact_points"] * exp_df["direction"].map(
                {"positive": 1, "negative": -1}
            )
            exp_df["color"] = exp_df["direction"].map(
                {"positive": "#2E86AB", "negative": "#E8553D"}
            )

            fig = go.Figure(go.Bar(
                x=exp_df["signed_impact"],
                y=exp_df["label"],
                orientation="h",
                marker_color=exp_df["color"],
            ))
            fig.update_layout(
                title="Top ML Contributing Factors (SHAP)",
                xaxis_title="Impact (points)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            for s in shap_exps:
                icon = "🟢" if s["direction"] == "positive" else "🔴"
                st.markdown(f"{icon} {s['text']}")
    else:
        st.info("Explanation not available for this applicant. "
                "Select an applicant from the top-ranked list.")

    st.markdown("---")

    # Human-in-the-loop
    st.markdown("#### ✋ Decision (Human-in-the-Loop)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve", key=f"approve_{selected}", type="primary"):
            st.success(f"Applicant {selected} APPROVED. Decision logged.")
    with col2:
        if st.button("❌ Reject", key=f"reject_{selected}"):
            st.error(f"Applicant {selected} REJECTED. Decision logged.")
    with col3:
        if st.button("⏸️ Defer", key=f"defer_{selected}"):
            st.warning(f"Applicant {selected} DEFERRED for further review.")


def main():
    render_header()

    transactions, enriched = load_data()
    model, metrics, scores = train_scoring_model(transactions, enriched)
    explanations = generate_explanations(model, scores, transactions)

    tab1, tab2, tab3 = st.tabs([
        "📊 Overview", "📋 Applicant Rankings", "🔍 Detailed Review"
    ])

    with tab1:
        render_metrics_panel(metrics, scores)
        col1, col2 = st.columns(2)
        with col1:
            render_score_distribution(scores)
        with col2:
            render_region_analysis(scores)

        # Direction breakdown
        direction_stats = scores.groupby("primary_direction").agg(
            avg_score=("merit_score", "mean"),
            count=("bin_iin", "count"),
        ).reset_index().sort_values("avg_score", ascending=False)

        fig = px.bar(
            direction_stats,
            x="primary_direction",
            y="avg_score",
            title="Average Merit Score by Farming Direction",
            labels={"primary_direction": "Direction", "avg_score": "Avg Score"},
            color="count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Bankrupt vs non-bankrupt comparison
        st.markdown("#### Bankruptcy Impact on Scores")
        col1, col2 = st.columns(2)
        with col1:
            bankrupt_avg = scores[scores["is_bankrupt"] == True]["merit_score"].mean()
            non_bankrupt_avg = scores[scores["is_bankrupt"] == False]["merit_score"].mean()
            fig = go.Figure(go.Bar(
                x=["Not Bankrupt", "Bankrupt"],
                y=[non_bankrupt_avg, bankrupt_avg],
                marker_color=["#2E86AB", "#E8553D"],
                text=[f"{non_bankrupt_avg:.1f}", f"{bankrupt_avg:.1f}"],
                textposition="auto",
            ))
            fig.update_layout(title="Avg Merit Score: Bankrupt vs Not", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            tax_avg = scores[scores["has_tax_debt"] == True]["merit_score"].mean()
            no_tax_avg = scores[scores["has_tax_debt"] == False]["merit_score"].mean()
            fig = go.Figure(go.Bar(
                x=["No Tax Debt", "Has Tax Debt"],
                y=[no_tax_avg, tax_avg],
                marker_color=["#2E86AB", "#E8553D"],
                text=[f"{no_tax_avg:.1f}", f"{tax_avg:.1f}"],
                textposition="auto",
            ))
            fig.update_layout(title="Avg Merit Score: Tax Debt Impact", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        render_applicant_table(scores)

    with tab3:
        render_applicant_detail(scores, explanations)


if __name__ == "__main__":
    main()
