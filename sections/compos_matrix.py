import streamlit as st
import plotly.express as px
from utils.file_io import load_agility_data, load_agility_volume_map
from utils.config import BRANDS
import pandas as pd
from utils.date_utils import get_selected_date_range  # Add this import

def render():
    st.subheader("🏷️ Brand Archetypes: Volume vs. Quality")

    st.markdown("""
    **Note:** News articles were selected from February 1st until July 31st. Only articles where the bank was mentioned in either the title or the first paragraph were kept for analysis.
    """)

    st.markdown("""
    **Quality definition:** The Brand Mention Quality (BMQ) score is a measure of how well the brand is represented in the article. It takes into account the [PageRank]('https://en.wikipedia.org/wiki/PageRank') of the website, how often the brand is mentioned and where the brand is mentioned in the article. The BMQ score ranges from 0 to 1, where 1 is the best possible score.
    """)

    summary = {}

    volume_map = load_agility_volume_map()
    start_date, end_date = get_selected_date_range()  # Get selected date range

    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is None or df.empty:
            continue

        # Filter by selected date range
        if "Published Date" in df.columns:
            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            df = df.dropna(subset=["Published Date"])
            df = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)]

        volume = len(df)  # Use filtered count
        quality = df["BMQ"].mean() if "BMQ" in df.columns and not df.empty else 0

        if "Top Archetype" in df.columns and not df.empty:
            archetype_counts = df["Top Archetype"].value_counts(normalize=True) * 100
            top_3 = archetype_counts.nlargest(3)
            archetype_text = "<br>".join([f"{a} ({p:.1f}%)" for a, p in top_3.items()])
        else:
            archetype_text = "N/A"

        summary[brand] = {
            "Volume": volume,
            "Quality": round(quality, 2) if pd.notna(quality) else 0,
            "Archetypes": archetype_text
        }

    if not summary:
        st.warning("No archetype data found.")
        return

    df_summary = pd.DataFrame.from_dict(summary, orient="index").reset_index()
    df_summary.columns = ["Company", "Volume", "Quality", "Archetypes"]

    fig = px.scatter(
        df_summary,
        x="Volume",
        y="Quality",
        text="Company",
        hover_data=["Archetypes"],
        title="Company Positioning by Volume & Quality",
    )

    fig.update_traces(textposition="top center", marker=dict(size=10))

    for _, row in df_summary.iterrows():
        fig.add_annotation(
            x=row["Volume"],
            y=row["Quality"],
            text=f"<b>{row['Company']}</b><br>{row['Archetypes']}",
            showarrow=False,
            font=dict(size=9),
            align="center",
            bgcolor="white",
            borderpad=4
        )

    fig.update_layout(
        xaxis_title="Volume (Articles)",
        yaxis_title="Quality (Avg. BMQ)",
        margin=dict(l=40, r=40, t=40, b=40),
        dragmode=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        'Read more about brand archetypes here: [Brandtypes](https://www.comp-os.com/brandtypes)'
    )
