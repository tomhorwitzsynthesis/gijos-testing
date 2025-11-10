import streamlit as st
import pandas as pd
from utils.file_io import load_agility_data
from utils.date_utils import get_selected_date_range
from utils.config import BRANDS

TOPIC_COLUMNS = ["Cluster_Topic1", "Cluster_Topic2", "Cluster_Topic3"]

def render() -> None:
    """
    Render a display of top 5 content topics from Agility data.
    Tabs: Combined + per-brand, all with article counts.
    """
    st.subheader("🧠 Key Communication Topics")
    start_date, end_date = get_selected_date_range()

    company_data = {}
    brand_counts = {}

    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is None or "Published Date" not in df.columns:
            continue

        df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
        df = df.dropna(subset=["Published Date"])
        df = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)]

        if df.empty or not all(col in df.columns for col in TOPIC_COLUMNS):
            continue

        company_data[brand] = df
        brand_counts[brand] = len(df)

    if not company_data:
        st.warning("No topic data available for the selected period.")
        return

    # Total article count
    total_articles = sum(brand_counts.values())

    tabs = st.tabs(
        [f"🌍 All Brands ({total_articles})"] +
        [f"🏢 {brand} ({brand_counts.get(brand, 0)})" for brand in BRANDS]
    )

    # Tab 0: Combined
    with tabs[0]:
        st.markdown("**Top topics across all brands.**")
        combined_counts = extract_topics(company_data)
        display_top_topics(combined_counts)

    # Per-brand tabs
    for i, brand in enumerate(BRANDS, start=1):
        with tabs[i]:
            if brand not in company_data:
                st.info(f"No topic data for {brand}.")
                continue
            st.markdown(f"**Top topics for {brand}**")
            brand_counts = extract_topics({brand: company_data[brand]})
            display_top_topics(brand_counts)


def extract_topics(dataframes_dict: dict) -> dict:
    """
    Extract and count topic mentions from cluster columns.
    """
    topic_counts = {}
    for df in dataframes_dict.values():
        for col in TOPIC_COLUMNS:
            for topic in df[col].dropna():
                topic = str(topic).strip()
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
    return topic_counts

def display_top_topics(topic_counts: dict) -> None:
    """
    Show the top 5 topics in a block format.
    """
    if not topic_counts:
        st.info("No topics found.")
        return

    total = sum(topic_counts.values())
    top_5 = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    for topic, count in top_5:
        pct = (count / total) * 100 if total > 0 else 0
        st.markdown(
            f'<div style="display: flex; justify-content: space-between; border: 1px solid #ccc; '
            f'padding: 6px 10px; border-radius: 6px; margin-bottom: 6px;">'
            f'<div>{topic}</div>'
            f'<div style="background-color: #eee; padding: 4px 8px; border-radius: 4px;">{pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
