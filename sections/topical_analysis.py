import streamlit as st
import pandas as pd
import os
from utils.config import DATA_ROOT  # <-- import here

TOPIC_INSIGHTS_PATH = os.path.join(DATA_ROOT, "agility", "topic_insights.xlsx")

def load_topic_insights():
    if not os.path.exists(TOPIC_INSIGHTS_PATH):
        st.error("topic_insights.xlsx not found in agility folder.")
        return None
    try:
        df = pd.read_excel(TOPIC_INSIGHTS_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading topic_insights.xlsx: {e}")
        return None

def render() -> None:
    st.subheader("🧠 Key Communication Topics with Examples")

    df = load_topic_insights()
    if df is None or df.empty:
        st.warning("No topic insights data available.")
        return

    brands = df["Brand"].unique()
    tabs = st.tabs([f"🏢 {brand}" for brand in brands])

    for i, brand in enumerate(brands):
        with tabs[i]:
            brand_df = df[df["Brand"] == brand]
            topics = brand_df["Topic"].unique()
            for topic in topics:
                topic_df = brand_df[brand_df["Topic"] == topic]
                examples = topic_df["One-line Insight"].tolist()
                st.markdown(
                    f"""
                    <div style="border:1px solid #ccc; border-radius:8px; padding:16px; margin-bottom:16px; background-color:#f9f9f9;">
                        <div style="font-weight:bold; font-size:1.1em; margin-bottom:8px;">{topic}</div>
                        <ul style="margin-left:18px;">
                            {''.join(f'<li>{ex}</li>' for ex in examples)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
