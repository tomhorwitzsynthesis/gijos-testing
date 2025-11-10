import streamlit as st
import pandas as pd
import plotly.express as px
from utils.file_io import load_agility_data, load_creativity_rankings
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

def _normalize_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = name.split("|")[0].strip()
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in base)
    return " ".join(cleaned.split())

def _format_simple_metric_card(label, val, pct=None, rank_now=None, total_ranks=None):
    rank_color = "gray"
    if rank_now is not None and total_ranks:
        if int(rank_now) == 1:
            rank_color = "green"
        elif int(rank_now) == int(total_ranks):
            rank_color = "red"
    pct_color = None
    if pct is not None:
        pct_color = "green" if pct > 0 else "red" if pct < 0 else "gray"
    pct_html = f'<p style="margin:0; color:{pct_color};">Δ {pct:.1f}%</p>' if pct is not None else ''
    rank_html = f'<p style="margin:0; color:{rank_color};">Rank {int(rank_now)}</p>' if rank_now is not None else ''
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h5 style="margin:0;">{label}</h5>
            <h3 style="margin:5px 0;">{val}</h3>
            {pct_html}
            {rank_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _compute_brand_strength_from_agility():
    strength = {}
    for brand in BRANDS:
        df_ag = load_agility_data(brand)
        if df_ag is None or df_ag.empty:
            continue
        if "Top Archetype" not in df_ag.columns:
            continue
        vc = df_ag["Top Archetype"].dropna().value_counts()
        total = vc.sum()
        if total > 0:
            display_name = BRAND_NAME_MAPPING.get(brand, brand)
            strength[display_name] = float(vc.iloc[0] / total * 100)
    return strength

_BRAND_DISPLAY_ORDER = [BRAND_NAME_MAPPING.get(b, b) for b in BRANDS]

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
    full_frames = {}

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
        full_frames[display_name] = df.copy()

        df_filtered = df[(df["Published Date"] >= start_date) & (df["Published Date"] <= end_date)].copy()
        if df_filtered.empty:
            continue

        filtered_frames.append(df_filtered)

    if not filtered_frames:
        st.warning("No data available for the selected period.")
        return

    df_all = pd.concat(filtered_frames, ignore_index=True)

    # --- Brand Summary Cards ---
    st.markdown("### Brand Summary")

    brand_strength_map = _compute_brand_strength_from_agility()
    if full_frames:
        df_cards_source = pd.concat(full_frames.values(), ignore_index=True)
    else:
        df_cards_source = pd.DataFrame(columns=["Company", "Published Date"])

    mentions_stats = {}
    mentions_total = 0
    if not df_cards_source.empty and "Published Date" in df_cards_source.columns:
        window_end = df_cards_source["Published Date"].max()
        if pd.notna(window_end):
            window_start = window_end - pd.DateOffset(months=6)
            df_window = df_cards_source[(df_cards_source["Published Date"] >= window_start) & (df_cards_source["Published Date"] <= window_end)]
            mentions_series = df_window.groupby("Company").size()
            mentions_total = len(mentions_series)
            mentions_mean = mentions_series.mean() if mentions_total else 0
            mentions_ranks = mentions_series.rank(ascending=False, method="min") if mentions_total else pd.Series(dtype=float)
            for brand_name, value in mentions_series.items():
                norm = _normalize_brand(brand_name)
                denom = mentions_mean if mentions_mean != 0 else 1
                delta = ((value - mentions_mean) / denom) * 100 if mentions_mean else 0
                rank_now = mentions_ranks.get(brand_name, None)
                mentions_stats[norm] = {
                    "value": int(value),
                    "delta": float(delta),
                    "rank": int(rank_now) if pd.notna(rank_now) else None,
                }

    bs_df = pd.DataFrame({
        "brand": list(brand_strength_map.keys()),
        "strength": list(brand_strength_map.values()),
    })
    if not bs_df.empty:
        bs_df["brand_norm"] = bs_df["brand"].apply(_normalize_brand)
        bs_df["rank"] = bs_df["strength"].rank(ascending=False, method="min")
        bs_mean = bs_df["strength"].mean() if len(bs_df) else 0
        denom = bs_mean if bs_mean != 0 else 1
        bs_df["delta_vs_mean_pct"] = ((bs_df["strength"] - bs_mean) / denom) * 100 if len(bs_df) else 0
        brand_strength_stats = {
            row["brand_norm"]: {
                "value": float(row["strength"]),
                "delta": float(row["delta_vs_mean_pct"]),
                "rank": int(row["rank"]),
            }
            for _, row in bs_df.iterrows()
        }
        bs_total = len(bs_df)
    else:
        brand_strength_stats = {}
        bs_total = 0

    creativity_df = load_creativity_rankings("pr")
    creativity_stats = {}
    creativity_total = 0
    if not creativity_df.empty:
        creativity_df["brand_norm"] = creativity_df["brand"].apply(_normalize_brand)
        creativity_df["rank"] = pd.to_numeric(creativity_df["rank"], errors="coerce")
        creativity_df["originality_score"] = pd.to_numeric(creativity_df["originality_score"], errors="coerce")
        cre_mean = creativity_df["originality_score"].mean()
        denom = cre_mean if cre_mean != 0 else 1
        creativity_df["delta_vs_mean_pct"] = ((creativity_df["originality_score"] - cre_mean) / denom) * 100
        for _, row in creativity_df.iterrows():
            if pd.isna(row["originality_score"]):
                continue
            creativity_stats[row["brand_norm"]] = {
                "score": float(row["originality_score"]),
                "delta": float(row["delta_vs_mean_pct"]) if not pd.isna(row["delta_vs_mean_pct"]) else None,
                "rank": int(row["rank"]) if not pd.isna(row["rank"]) else None,
            }
        creativity_total = len(creativity_stats)

    present_names = set(full_frames.keys()) | set(brand_strength_map.keys())
    if not creativity_df.empty:
        present_names.update(creativity_df["brand"].dropna().astype(str))

    available_brands = []
    brand_display_order = [BRAND_NAME_MAPPING.get(b, b) for b in BRANDS]
    for display in brand_display_order:
        norm = _normalize_brand(display)
        if (display in present_names) or (norm in mentions_stats) or (norm in brand_strength_stats) or (norm in creativity_stats):
            available_brands.append(display)
    for name in sorted(present_names):
        if name not in available_brands:
            available_brands.append(name)

    if available_brands:
        brand_tabs = st.tabs(available_brands)
        for idx, brand_display in enumerate(available_brands):
            norm = _normalize_brand(brand_display)
            with brand_tabs[idx]:
                col1, col2, col3 = st.columns(3)

                with col1:
                    mention_info = mentions_stats.get(norm)
                    if mention_info:
                        _format_simple_metric_card(
                            label="Media Mentions (6 months)",
                            val=f"{mention_info['value']:,}",
                            pct=mention_info["delta"],
                            rank_now=mention_info["rank"],
                            total_ranks=mentions_total if mentions_total else None,
                        )
                    else:
                        _format_simple_metric_card("Media Mentions (6 months)", "N/A")

                with col2:
                    strength_info = brand_strength_stats.get(norm)
                    if strength_info:
                        _format_simple_metric_card(
                            label="Brand Strength",
                            val=f"{strength_info['value']:.1f}%",
                            pct=strength_info["delta"],
                            rank_now=strength_info["rank"],
                            total_ranks=bs_total if bs_total else None,
                        )
                    else:
                        _format_simple_metric_card("Brand Strength", "N/A")

                with col3:
                    creativity_info = creativity_stats.get(norm)
                    if creativity_info:
                        _format_simple_metric_card(
                            label="Creativity",
                            val=f"{creativity_info['score']:.2f}",
                            pct=creativity_info["delta"],
                            rank_now=creativity_info["rank"],
                            total_ranks=creativity_total if creativity_total else None,
                        )
                    else:
                        _format_simple_metric_card("Creativity", "N/A")
    else:
        st.info("No brand metrics available to display.")

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

