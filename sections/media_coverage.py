import streamlit as st
import pandas as pd
import plotly.express as px
from utils.file_io import load_agility_data
from utils.date_utils import get_selected_date_range
from utils.config import BRANDS, BRAND_COLORS, BRAND_NAME_MAPPING  # <-- add BRAND_COLORS

REGIONS = {
    "Total": None,
    "Lithuania": "Lithuania",
    "Latvia": "Latvia",
    "Estonia": "Estonia"
}

# ---- color helpers (consistent across all charts) ----
_BRAND_ORDER = list(BRAND_COLORS.keys())
_FALLBACK = "#BDBDBD"

def _present_color_map(present_labels):
    m = dict(BRAND_COLORS)
    for b in present_labels:
        if b not in m:
            m[b] = _FALLBACK
    return m
# -----------------------------------------------------

def render(mode: str = "by_brand"):
    """
    Display brand media coverage using pie charts.
    mode: 
      - "by_brand" → total brand share (one chart)
      - "by_brand_and_country" → one pie chart per country
    """
    if mode not in {"by_brand", "by_brand_and_country"}:
        st.error(f"Invalid mode '{mode}' in media_coverage.render(). Use 'by_brand' or 'by_brand_and_country'.")
        return

    st.subheader("📰 Media Mentions Coverage Share")

    st.markdown("""
    This section visualizes the share of media mentions across brands. 
    Use this to evaluate visibility or dominance in earned media.
    """)

    start_date, end_date = get_selected_date_range()

    filtered_frames = []

    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is None or "Published Date" not in df.columns:
            continue

        df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
        df = df.dropna(subset=["Published Date"])
        if df.empty:
            continue

        display_name = BRAND_NAME_MAPPING.get(brand, brand)
        df["Company"] = display_name

        df_filtered = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)].copy()
        if df_filtered.empty:
            continue

        filtered_frames.append(df_filtered)

    if not filtered_frames:
        st.warning("No data available for the selected period.")
        return

    df_all = pd.concat(filtered_frames, ignore_index=True)

    if mode == "by_brand":
        tabs = st.tabs(["📰 Coverage", "📢 Reach"])
        with tabs[0]:
            _plot_share_pie(df_all, title="Total Media Coverage Share")
        with tabs[1]:
            _plot_reach_pie(df_all, title="Total Media Reach Share (Impressions)")
    else:  # by_brand_and_country
        region_tabs = st.tabs([f"🌍 {region}" for region in REGIONS.keys()] + ["📢 Reach"])
        for i, (region_name, country_filter) in enumerate(REGIONS.items()):
            with region_tabs[i]:
                region_df = df_all if country_filter is None else df_all[df_all["Country"] == country_filter]
                if region_df.empty:
                    st.info("No data for this region.")
                    continue
                _plot_share_pie(region_df, title=f"{region_name} Media Coverage Share")
        # Reach tab (total impressions by brand)
        with region_tabs[-1]:
            _plot_reach_pie(df_all, title="Total Media Reach Share (Impressions)")

def _plot_share_pie(df: pd.DataFrame, title: str):
    counts = df.groupby("Company").size().reset_index(name="Articles")
    if counts.empty:
        st.info("No articles in the selected period.")
        return
    total = counts["Articles"].sum()
    counts["Percentage"] = (counts["Articles"] / total) * 100

    present = counts["Company"].unique()
    fig = px.pie(
        counts,
        names="Company",
        values="Articles",
        hover_data=["Percentage"],
        labels={"Percentage": "% of Total"},
        title=title,
        color="Company",
        color_discrete_map=_present_color_map(present),
        category_orders={"Company": _BRAND_ORDER},
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="%{label}: %{value} articles (%{customdata[0]:.1f}%)"
    )
    st.plotly_chart(fig, use_container_width=True)

def _plot_reach_pie(df: pd.DataFrame, title: str):
    if "Impressions" not in df.columns:
        st.info("No Impressions data available.")
        return
    reach = df.groupby("Company")["Impressions"].sum().reset_index()
    if reach.empty:
        st.info("No Impressions in the selected period.")
        return
    total_reach = reach["Impressions"].sum()
    reach["Percentage"] = (reach["Impressions"] / total_reach) * 100

    present = reach["Company"].unique()
    fig = px.pie(
        reach,
        names="Company",
        values="Impressions",
        hover_data=["Percentage"],
        labels={"Percentage": "% of Total"},
        title=title,
        color="Company",
        color_discrete_map=_present_color_map(present),
        category_orders={"Company": _BRAND_ORDER},
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="%{label}: %{value} impressions (%{customdata[0]:.1f}%)"
    )
    st.plotly_chart(fig, use_container_width=True)

