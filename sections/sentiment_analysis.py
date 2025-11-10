import streamlit as st
import pandas as pd
import plotly.express as px
from utils.file_io import load_agility_data
from utils.date_utils import get_selected_date_range
from utils.config import BRANDS

def render(mode: str = "by_company"):
    """
    Render sentiment distribution.
    mode = "by_company" → stacked bars per brand
    mode = "combined"   → one total bar for all brands
    """
    if mode not in {"by_company", "combined"}:
        st.error(f"Invalid mode '{mode}' in sentiment_analysis.render(). Use 'by_company' or 'combined'.")
        return

    st.subheader("📊 Sentiment Distribution")

    start_date, end_date = get_selected_date_range()

    if mode == "by_company":
        sentiment_summary = {}
        all_dfs = []

        for brand in BRANDS:
            df = load_agility_data(brand)
            if df is None or "Published Date" not in df.columns or "Sentiment" not in df.columns:
                continue

            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            df = df.dropna(subset=["Published Date"])
            df = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)]

            if df.empty:
                continue

            all_dfs.append(df)
            sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
            sentiment_summary[f"{brand} ({len(df)})"] = {
                "Positive": sentiment_counts.get("Positive", 0),
                "Neutral": sentiment_counts.get("Neutral", 0),
                "Negative": sentiment_counts.get("Negative", 0)
            }

        if not sentiment_summary:
            st.warning("No sentiment data available.")
            return

        # Add combined bar
        if all_dfs:
            df_all = pd.concat(all_dfs, ignore_index=True)
            sentiment_counts = df_all["Sentiment"].value_counts(normalize=True) * 100
            sentiment_summary[f"All Brands ({len(df_all)})"] = {
                "Positive": sentiment_counts.get("Positive", 0),
                "Neutral": sentiment_counts.get("Neutral", 0),
                "Negative": sentiment_counts.get("Negative", 0)
            }

        df_sent = pd.DataFrame.from_dict(sentiment_summary, orient="index").reset_index()
        df_sent = df_sent.melt(id_vars=["index"], var_name="Sentiment", value_name="Percentage")
        df_sent.columns = ["Company", "Sentiment", "Percentage"]

    else:  # mode == "combined"
        all_dfs = []

        for brand in BRANDS:
            df = load_agility_data(brand)
            if df is None or "Published Date" not in df.columns or "Sentiment" not in df.columns:
                continue

            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            df = df.dropna(subset=["Published Date"])
            df = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)]

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            st.warning("No sentiment data available.")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)
        sentiment_counts = df_all["Sentiment"].value_counts(normalize=True) * 100

        df_sent = pd.DataFrame({
            "Company": ["All Brands"] * 3,
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Percentage": [
                sentiment_counts.get("Positive", 0),
                sentiment_counts.get("Neutral", 0),
                sentiment_counts.get("Negative", 0)
            ]
        })

    # Plot
    fig = px.bar(
        df_sent,
        x="Company",
        y="Percentage",
        color="Sentiment",
        text="Percentage",
        barmode="stack",
        color_discrete_map={
            "Positive": "green",
            "Neutral": "grey",
            "Negative": "red"
        },
        title="Sentiment Distribution"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(xaxis_title="Company", yaxis_title="Percentage")

    st.plotly_chart(fig, use_container_width=True)

    # --- Added: Negative sentiment article topics for Artea (no article numbers) ---
    if "Artea" in BRANDS:
        with st.expander("🔎 Negative sentiment article topics for Artea", expanded=False):
            st.markdown(
                """
- **Client frustrations with Artea (multiple cases)**
  - Unexpected bank fees on account balances.
  - Complaints spreading on social media about poor treatment of customers.
  - These are reputational hits tied to service quality and fee transparency.
- **Broader financial/economic pressure**
  - Tax changes and economic downturn discussions reflect negatively on banks, with Artea mentioned as an example.
  - Suggests an association with systemic financial stress, not necessarily misconduct by the bank itself.
- **Fraud and scams targeting customers**
  - Several stories about scams and fraud attempts, where criminals impersonated or exploited Artea/Šiaulių bankas customers.
  - The negativity comes from reputational risk: banks seen as vectors or vulnerable points for fraud.
- **Šiaulių bankas stock market performance**
  - Coverage of declining stock value, analyst downgrades, and continued sell-offs.
  - Tone is negative because of weak market confidence and forecasts of reduced share price.
- **Artea liquidity/transaction issues**
  - At least one case of a business unable to retrieve funds for an extended period.
  - Directly undermines trust in the bank’s operations.

**In short:** Artea is criticized for poor customer service (fees, delays) and linked to fraud risk. Šiaulių bankas is framed negatively in financial press due to declining share value and negative analyst outlooks.
                """
            )
