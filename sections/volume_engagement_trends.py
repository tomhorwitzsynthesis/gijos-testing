import streamlit as st
import pandas as pd
import plotly.express as px
from utils.config import BRAND_NAME_MAPPING, BRAND_COLORS
from utils.date_utils import get_selected_date_range
from utils.file_io import load_social_data

PLATFORMS = ["linkedin"]

# --- name normalization: short -> long (must match BRAND_COLORS keys) ---
NAME_MAP = BRAND_NAME_MAPPING.copy()

# --- color helpers ---
_ALL_BRANDS = "All Brands"  # not used here, but reserved if you add a combined series later
_FALLBACK = "#BDBDBD"
_CATEGORY_ORDER = list(BRAND_COLORS.keys())

def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Company" in out.columns:
        out["Company"] = out["Company"].replace(NAME_MAP)
    return out

def _present_color_map(present_labels) -> dict:
    m = dict(BRAND_COLORS)
    for b in present_labels:
        if b not in m:
            m[b] = _FALLBACK
    return m
# -----------------------------------------------------------------------

def render(selected_platforms=None):
    st.subheader("📈 Social Media Volume & Engagement Trends")

    # Default to all supported platforms if none provided
    if not selected_platforms:
        selected_platforms = PLATFORMS

    selected_platforms = [platform.lower() for platform in selected_platforms if platform.lower() in PLATFORMS]
    if not selected_platforms:
        st.info("No supported social platforms selected.")
        return

    start_date, end_date = get_selected_date_range()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if end_ts <= start_ts:
        st.info("Invalid date range selected.")
        return

    months = pd.period_range(start=start_ts, end=end_ts - pd.Timedelta(days=1), freq="M")
    if months.empty:
        st.info("No months available in the selected range.")
        return

    month_labels = [m.strftime('%b %Y') for m in months]

    for platform in selected_platforms:
        st.markdown(f"### {platform.capitalize()}")

        combined_data = {
            "Month": [],
            "Company": [],
            "Volume": [],
            "Engagement": [],
            "Engagement_Per_Follower": []
        }

        any_data_for_platform = False

        for brand_key, brand_display in BRAND_NAME_MAPPING.items():
            df = load_social_data(brand_key, platform)
            if df is None or df.empty or "Published Date" not in df.columns:
                continue

            df = df.dropna(subset=["Published Date"])
            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            df = df.dropna(subset=["Published Date"])
            if df.empty:
                continue

            # filter to the selected window
            df = df[(df["Published Date"] >= start_ts) & (df["Published Date"] < end_ts)]
            if df.empty:
                continue

            df["Month"] = df["Published Date"].dt.to_period("M")
            any_data_for_platform = True

            for period in months:
                month_df = df[df["Month"] == period]
                if month_df.empty:
                    continue

                volume = len(month_df)

                engagement = (
                    month_df.get("num_likes", pd.Series(0)).sum()
                    + month_df.get("num_comments", pd.Series(0)).sum() * 3
                )
                followers = month_df.get("user_followers", pd.Series()).dropna()

                follower_count = followers.iloc[-1] if not followers.empty else 0
                epf = engagement / follower_count if follower_count > 0 else 0

                combined_data["Month"].append(period.strftime('%b %Y'))
                combined_data["Company"].append(brand_display)  # short name for UI
                combined_data["Volume"].append(volume)
                combined_data["Engagement"].append(engagement)
                combined_data["Engagement_Per_Follower"].append(epf)

        if not any_data_for_platform or not combined_data["Month"]:
            st.info(f"No {platform.capitalize()} data found within the selected date range.")
            continue

        df_combined = pd.DataFrame(combined_data)

        tab1, tab2, tab3 = st.tabs(["📊 Volume", "🔥 Engagement", "📈 Engagement Per Follower"])

        # Volume
        with tab1:
            df_plot = _normalized(df_combined)  # normalize Company -> BRAND_COLORS keys
            fig_volume = px.line(
                df_plot,
                x="Month",
                y="Volume",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Monthly Post Volume",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            fig_volume.update_layout(xaxis=dict(categoryorder="array", categoryarray=month_labels))
            st.plotly_chart(fig_volume, use_container_width=True)

        # Engagement
        with tab2:
            df_plot = _normalized(df_combined)
            fig_engagement = px.line(
                df_plot,
                x="Month",
                y="Engagement",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Monthly Engagement Trend",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            fig_engagement.update_layout(xaxis=dict(categoryorder="array", categoryarray=month_labels))
            st.plotly_chart(fig_engagement, use_container_width=True)

        # Engagement per Follower
        with tab3:
            df_plot = _normalized(df_combined)
            fig_epf = px.line(
                df_plot,
                x="Month",
                y="Engagement_Per_Follower",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Engagement per Follower",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            fig_epf.update_layout(xaxis=dict(categoryorder="array", categoryarray=month_labels))
            st.plotly_chart(fig_epf, use_container_width=True)
