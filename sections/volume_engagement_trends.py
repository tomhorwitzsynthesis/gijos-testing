import streamlit as st
import pandas as pd
import plotly.express as px
from utils.config import BRAND_NAME_MAPPING, BRAND_COLORS, BRANDS
from utils.date_utils import get_selected_date_range
from utils.file_io import load_social_data

PLATFORMS = ["linkedin"]

# --- name normalization: short -> long (must match BRAND_COLORS keys) ---
NAME_MAP = BRAND_NAME_MAPPING.copy()

# --- color helpers ---
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


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
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


def render(selected_platforms=None):
    st.subheader("Social Media Volume & Engagement Trends")

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

    for platform in selected_platforms:
        st.markdown(f"### {platform.capitalize()}")

        weekly_rows = {
            "Week": [],
            "Company": [],
            "Volume": [],
            "Engagement": [],
            "Engagement_Per_Follower": [],
        }

        any_data_for_platform = False

        # Iterate over BRANDS and use mapping to get file name key
        for brand_display in BRANDS:
            # Find all possible keys that map to this brand
            possible_keys = [key for key, value in BRAND_NAME_MAPPING.items() if value == brand_display]
            # Prefer keys that are lowercase/hyphenated (these are more likely to be in the data file)
            # Sort by: 1) keys with hyphens first, 2) lowercase keys, 3) others
            possible_keys.sort(key=lambda k: (("-" not in k.lower(), k != k.lower(), k)))
            
            # Try each key until we find one that returns data
            df = None
            for brand_key in possible_keys:
                df = load_social_data(brand_key, platform)
                if df is not None and not df.empty:
                    break
            
            # If no key worked, try the fallback
            if df is None or df.empty:
                brand_key = brand_display.lower().replace(" ", "-")
                df = load_social_data(brand_key, platform)
            if df is None or df.empty or "Published Date" not in df.columns:
                continue

            df = df.copy()
            df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
            df = df.dropna(subset=["Published Date"])
            if df.empty:
                continue

            df = df[(df["Published Date"] >= start_ts) & (df["Published Date"] < end_ts)]
            if df.empty:
                continue

            df["Week"] = df["Published Date"].apply(_week_start)
            any_data_for_platform = True

            weekly_groups = df.groupby("Week")

            for week_start, week_df in weekly_groups:
                volume = len(week_df)
                likes = week_df.get("num_likes", pd.Series(dtype="float")).fillna(0).sum()
                comments = week_df.get("num_comments", pd.Series(dtype="float")).fillna(0).sum()
                shares = week_df.get("num_shares", pd.Series(dtype="float")).fillna(0).sum()
                engagement = likes + comments * 3 + shares

                followers = week_df.get("user_followers", pd.Series(dtype="float")).dropna()
                follower_count = followers.iloc[-1] if not followers.empty else 0
                engagement_per_follower = engagement / follower_count if follower_count else 0

                weekly_rows["Week"].append(week_start)
                weekly_rows["Company"].append(brand_display)
                weekly_rows["Volume"].append(volume)
                weekly_rows["Engagement"].append(engagement)
                weekly_rows["Engagement_Per_Follower"].append(engagement_per_follower)

        if not any_data_for_platform or not weekly_rows["Week"]:
            st.info(f"No {platform.capitalize()} data found within the selected date range.")
            continue

        df_combined = pd.DataFrame(weekly_rows)
        df_combined["Week"] = pd.to_datetime(df_combined["Week"])

        tab1, tab2, tab3 = st.tabs(["Volume", "Engagement", "Engagement Per Follower"])

        with tab1:
            df_plot = _normalized(df_combined)
            fig_volume = px.line(
                df_plot,
                x="Week",
                y="Volume",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Weekly Post Volume",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_volume, df_plot["Week"])
            fig_volume.update_layout(yaxis_title="Number of Posts")
            st.plotly_chart(fig_volume, use_container_width=True)

        with tab2:
            df_plot = _normalized(df_combined)
            fig_engagement = px.line(
                df_plot,
                x="Week",
                y="Engagement",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Weekly Engagement Trend",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_engagement, df_plot["Week"])
            fig_engagement.update_layout(yaxis_title="Total Interactions")
            st.plotly_chart(fig_engagement, use_container_width=True)

        with tab3:
            df_plot = _normalized(df_combined)
            fig_epf = px.line(
                df_plot,
                x="Week",
                y="Engagement_Per_Follower",
                color="Company",
                markers=True,
                title=f"{platform.capitalize()} - Weekly Engagement per Follower",
                color_discrete_map=_present_color_map(df_plot["Company"].unique()),
                category_orders={"Company": _CATEGORY_ORDER},
            )
            _format_week_axis(fig_epf, df_plot["Week"])
            fig_epf.update_layout(yaxis_title="Engagement / Follower")
            st.plotly_chart(fig_epf, use_container_width=True)
