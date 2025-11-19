import streamlit as st
import pandas as pd
import plotly.express as px
from utils.file_io import load_agility_data
from utils.config import BRANDS, BRAND_COLORS   # <-- long-key palette, e.g., "SEB Lietuvoje"

# --- name normalization: short -> long (matches BRAND_COLORS keys) ---
NAME_MAP = {
    "Swedbank": "Swedbank Lietuvoje",
    "SEB": "SEB Lietuvoje",
    "Luminor": "Luminor Lietuva",
    "Citadele": "Citadele bankas",
    "Artea": "Artea",
}

# --- color helpers ---
_ALL_BRANDS_LABEL = "All Brands"
_FALLBACK = "#BDBDBD"


def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Company values mapped to BRAND_COLORS keys."""
    if "Company" not in df.columns:
        return df
    out = df.copy()
    out["Company"] = out["Company"].replace(NAME_MAP)
    return out


def _present_color_map(present_labels) -> dict:
    """Color map covering present labels + neutral for 'All Brands'."""
    m = dict(BRAND_COLORS)
    if _ALL_BRANDS_LABEL in present_labels:
        m[_ALL_BRANDS_LABEL] = _FALLBACK
    for b in present_labels:
        if b not in m:
            m[b] = _FALLBACK
    return m


_CATEGORY_ORDER = list(BRAND_COLORS.keys()) + [_ALL_BRANDS_LABEL]


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Return Monday of the calendar week for the provided timestamp."""
    if pd.isna(ts):
        return ts
    return (ts - pd.to_timedelta(ts.weekday(), unit="D")).normalize()


def _format_week_axis(fig, series: pd.Series):
    ticks = sorted(pd.to_datetime(series.dropna().unique()))
    fig.update_layout(
        xaxis_title="Week Starting",
        xaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=[pd.to_datetime(t).strftime("%d %b %Y") for t in ticks],
        ),
    )


def render(mode: str = "by_company"):
    """
    Plot article volume trends by week.
    mode = "by_company" → lines per brand
    mode = "combined"   → one line summing all brands
    """
    if mode not in {"by_company", "combined"}:
        st.error(f"Invalid mode '{mode}' in volume_trends.render(). Use 'by_company' or 'combined'.")
        return

    st.subheader("Weekly Media Mention Trends")

    # Determine min/max dates from all brands' data
    all_dates = []
    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is not None and "Published Date" in df.columns:
            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            all_dates.extend(df["Published Date"].dropna().tolist())

    if not all_dates:
        st.warning("No data available for volume trends.")
        return

    min_date = pd.Timestamp(min(all_dates))
    max_date = pd.Timestamp(max(all_dates))
    start_week = _week_start(min_date)
    end_week = _week_start(max_date)
    weeks = pd.date_range(start=start_week, end=end_week, freq="W-MON")

    if weeks.empty:
        st.warning("Insufficient data to compute weekly trends.")
        return

    volume_rows, impressions_rows, bmq_rows = [], [], []

    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is None or "Published Date" not in df.columns:
            continue

        df = df.copy()
        df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
        df = df.dropna(subset=["Published Date"])
        if df.empty:
            continue

        df["Week"] = df["Published Date"].apply(_week_start)

        weekly_counts = df.groupby("Week").size().reindex(weeks, fill_value=0)

        if "Impressions" in df.columns:
            df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
            weekly_impressions = df.groupby("Week")["Impressions"].sum().reindex(weeks, fill_value=0)
        else:
            weekly_impressions = pd.Series(0, index=weeks, dtype=float)

        if "BMQ" in df.columns:
            df["BMQ"] = pd.to_numeric(df["BMQ"], errors="coerce")
            weekly_bmq = df.groupby("Week")["BMQ"].mean().reindex(weeks, fill_value=pd.NA)
        else:
            weekly_bmq = pd.Series(pd.NA, index=weeks, dtype="float")

        for week, count in weekly_counts.items():
            volume_rows.append({"Week": week, "Company": brand, "Volume": count})
        for week, impressions in weekly_impressions.items():
            impressions_rows.append({"Week": week, "Company": brand, "Impressions": impressions})
        for week, bmq in weekly_bmq.items():
            bmq_rows.append({"Week": week, "Company": brand, "BMQ": bmq})

    tab1, tab2, tab3 = st.tabs(["Volume", "Impressions", "BMQ"])

    with tab1:
        if not volume_rows:
            st.warning("No volume data found.")
        else:
            df_volume = pd.DataFrame(volume_rows)
            if mode == "combined":
                df_volume = df_volume.groupby("Week", as_index=False).agg({"Volume": "sum"})
                df_volume["Company"] = _ALL_BRANDS_LABEL
            df_volume = _normalized(df_volume)
            df_volume["Week"] = pd.to_datetime(df_volume["Week"])
            fig_volume = px.line(
                df_volume,
                x="Week",
                y="Volume",
                color="Company",
                markers=True,
                title="Weekly Trend of Media Mentions",
                color_discrete_map=_present_color_map(df_volume["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_volume, df_volume["Week"])
            fig_volume.update_layout(yaxis_title="Number of Articles")
            st.plotly_chart(fig_volume, use_container_width=True)

    with tab2:
        if not impressions_rows:
            st.warning("No impressions data found.")
        else:
            df_impressions = pd.DataFrame(impressions_rows)
            if mode == "combined":
                df_impressions = df_impressions.groupby("Week", as_index=False).agg({"Impressions": "sum"})
                df_impressions["Company"] = _ALL_BRANDS_LABEL
            df_impressions = _normalized(df_impressions)
            df_impressions["Week"] = pd.to_datetime(df_impressions["Week"])
            fig_impressions = px.line(
                df_impressions,
                x="Week",
                y="Impressions",
                color="Company",
                markers=True,
                title="Weekly Trend of Total Impressions",
                color_discrete_map=_present_color_map(df_impressions["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_impressions, df_impressions["Week"])
            fig_impressions.update_layout(yaxis_title="Total Impressions")
            st.plotly_chart(fig_impressions, use_container_width=True)

    with tab3:
        if not bmq_rows:
            st.warning("No BMQ data found.")
        else:
            df_bmq = pd.DataFrame(bmq_rows)
            if mode == "combined":
                df_bmq = df_bmq.groupby("Week", as_index=False).agg({"BMQ": "mean"})
                df_bmq["Company"] = _ALL_BRANDS_LABEL
            df_bmq = _normalized(df_bmq)
            df_bmq["BMQ"] = pd.to_numeric(df_bmq["BMQ"], errors="coerce")
            df_bmq["Week"] = pd.to_datetime(df_bmq["Week"])
            fig_bmq = px.line(
                df_bmq,
                x="Week",
                y="BMQ",
                color="Company",
                markers=True,
                title="Weekly Trend of Average Article Quality (BMQ)",
                color_discrete_map=_present_color_map(df_bmq["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_bmq, df_bmq["Week"])
            fig_bmq.update_layout(yaxis_title="Average BMQ Score")
            st.plotly_chart(fig_bmq, use_container_width=True)
