import os
import hashlib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime
from typing import Dict, List, Tuple

from utils.config import BRANDS, BRAND_COLORS, BRAND_NAME_MAPPING, DATA_ROOT
from utils.date_utils import get_selected_date_range
from utils.file_io import load_ads_data


POST_TEXT_COLUMNS = ["Post", "post_text", "content", "text"]
CUSTOM_STOPWORDS = STOPWORDS.union(
    {
        "https", "http", "linkedin", "amp", "amp;", "co", "www",
        "lt", "lt.", "eu", "amp", "amp;", "amp&", "ampamp", "com",
        "video", "photo", "ampquot", "utm", "utm_source", "utm_medium", "ir"
    }
)

MIN_REACH_THRESHOLD = 1000
MAX_DURATION_DAYS = 365


@st.cache_data(ttl=0)
def _load_all_linkedin_text():
    """
    Preprocess all LinkedIn post text once and cache it.
    Returns a dict: {brand: [(date, text), ...]}
    This avoids reloading Excel files on every render.
    Uses the same logic as load_social_data to ensure compatibility.
    """
    brand_texts: Dict[str, List[Tuple[datetime, str]]] = {}
    
    # Load the main LinkedIn Excel file once
    path = os.path.join(DATA_ROOT, "social_media", "linkedin_posts.xlsx")
    if not os.path.exists(path):
        return brand_texts
    
    try:
        df_all = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        st.error(f"Error loading LinkedIn data: {e}")
        return brand_texts
    
    if "user_id" not in df_all.columns:
        return brand_texts
    
    # Find the text column
    text_col = None
    for col in POST_TEXT_COLUMNS:
        if col in df_all.columns:
            text_col = col
            break
    
    if text_col is None:
        return brand_texts
    
    # Normalize dates - use same logic as load_social_data
    if "Published Date" in df_all.columns:
        df_all["Published Date"] = pd.to_datetime(df_all["Published Date"], utc=True, errors="coerce")
    elif "date_posted" in df_all.columns:
        df_all["Published Date"] = pd.to_datetime(df_all["date_posted"], utc=True, errors="coerce")
    elif "timestamp" in df_all.columns:
        df_all["Published Date"] = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")
    else:
        return brand_texts
    
    df_all["Published Date"] = df_all["Published Date"].dt.tz_localize(None)
    
    # Process each brand using the same matching logic as load_social_data
    for brand_display in BRANDS:
        # Find all possible keys that map to this brand (including canonical name)
        possible_keys = [brand_display]  # Start with canonical name
        possible_keys.extend([key for key, value in BRAND_NAME_MAPPING.items() if value == brand_display])
        
        # Try to match user_id with any variation (same logic as load_social_data)
        mask = pd.Series([False] * len(df_all), index=df_all.index)
        
        for key in possible_keys:
            key_lower = key.lower().strip()
            # Try exact match
            mask |= df_all["user_id"].astype(str).str.strip().str.lower() == key_lower
        
        if not mask.any():
            continue
        
        brand_df = df_all[mask].copy()
        brand_df = brand_df.dropna(subset=["Published Date", text_col])
        
        if brand_df.empty:
            continue
        
        # Store (date, text) tuples
        texts = []
        for _, row in brand_df.iterrows():
            text = str(row[text_col]).strip()
            if text and text.lower() not in ["nan", "none", ""]:
                texts.append((row["Published Date"], text))
        
        if texts:
            brand_texts[brand_display] = texts
    
    return brand_texts


def _filter_texts_by_date(
    texts: List[Tuple[datetime, str]], 
    start_date: datetime, 
    end_date: datetime
) -> str:
    """Filter texts by date range and combine into single string."""
    filtered = [text for date, text in texts if start_date <= date < end_date]
    return " ".join(filtered).strip()


def _generate_wordcloud(text: str, brand_color: str | None = None):
    """Generate a prettier wordcloud with improved styling."""
    if not text or not text.strip():
        return None
    
    # Configure wordcloud with better styling
    # Use nice colormaps: 'viridis', 'plasma', 'inferno', 'magma', 'Set2', 'Set3'
    # For brands, use a color that complements the brand color
    colormap_options = [
        "viridis", "plasma", "inferno", "magma", "Set2", "Set3", 
        "tab10", "Dark2", "Paired", "Accent"
    ]
    
    if brand_color:
        # For brand-specific clouds, use a complementary colormap
        # Pick one based on brand color hash for consistency
        colormap_idx = int(hashlib.md5(brand_color.encode()).hexdigest(), 16) % len(colormap_options)
        colormap = colormap_options[colormap_idx]
    else:
        # For overview, use viridis (nice gradient)
        colormap = "viridis"
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=CUSTOM_STOPWORDS,
        max_words=200,
        relative_scaling=0.5,
        colormap=colormap,
        min_font_size=10,
        max_font_size=120,
        font_path=None,  # Use default font, or specify a path to a TTF file
        prefer_horizontal=0.7,
        collocations=True,
        contour_width=0,  # No contour for cleaner look
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white', dpi=100)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def _load_filtered_ads_data():
    """Load ads data, apply shared filters, and return cleaned DataFrame."""
    df = load_ads_data()
    if df is None or df.empty:
        return None

    start_date, end_date = get_selected_date_range()
    if "startDateFormatted" in df.columns:
        df = df.dropna(subset=["startDateFormatted"])
        df = df[(df["startDateFormatted"] >= start_date) & (df["startDateFormatted"] < end_date)]

    if df.empty:
        return None

    if "duration_days" not in df.columns:
        if "startDateFormatted" in df.columns and "endDateFormatted" in df.columns:
            df["duration_days"] = (df["endDateFormatted"] - df["startDateFormatted"]).dt.days
        else:
            return None

    df_clean = df[
        (df["reach"] > MIN_REACH_THRESHOLD)
        & df["reach"].notna()
        & (df["duration_days"] > 0)
        & (df["duration_days"] <= MAX_DURATION_DAYS)
        & df["duration_days"].notna()
    ].copy()

    return df_clean if not df_clean.empty else None


def _render_density_plots_section():
    """Render combined view of ad reach vs duration with trend line and duration slider."""
    st.subheader("Ad Reach vs Duration")
    st.caption(
        "Explore the relationship between ad reach and duration. Use the slider to filter by minimum duration."
    )
    
    df_clean = _load_filtered_ads_data()
    if df_clean is None or df_clean.empty:
        st.info("No ads data available.")
        return

    if "reach" not in df_clean.columns or "brand" not in df_clean.columns:
        st.warning("Required columns (reach, brand) not found in ads data.")
        return
    
    # Get min and max duration for slider
    min_duration = int(df_clean["duration_days"].min())
    max_duration = int(df_clean["duration_days"].max())
    
    # Slider for minimum duration
    col1, col2 = st.columns([3, 1])
    with col1:
        min_duration_filter = st.slider(
            "Minimum Duration (days)",
            min_value=min_duration,
            max_value=max_duration,
            value=min_duration,
            step=1,
            help="Filter ads by minimum duration"
        )
    with col2:
        filtered_count = len(df_clean[df_clean["duration_days"] >= min_duration_filter])
        st.metric("Ads Shown", filtered_count)
    
    # Trend line controls
    trend_col1, trend_col2 = st.columns([1, 1])
    with trend_col1:
        show_trend_line = st.checkbox("Show Trend Line", value=True)
    with trend_col2:
        trend_shape = st.radio(
            "Trend Shape",
            options=["Quadratic", "Cubic"],
            horizontal=True,
            help="Choose the polynomial degree for the trend line.",
        )
    
    # Apply duration filter
    df_filtered = df_clean[df_clean["duration_days"] >= min_duration_filter].copy()
    
    if df_filtered.empty:
        st.info(f"No ads with duration >= {min_duration_filter} days.")
        return
    
    # Create scatter plot
    fig = px.scatter(
        df_filtered,
        x="duration_days",
        y="reach",
        color="brand",
        size="reach",
        hover_data=["brand"],
        labels={
            "duration_days": "Duration (days)",
            "reach": "Reach",
            "brand": "Brand"
        },
        title="Ad Reach vs Duration",
        color_discrete_map=BRAND_COLORS,
    )
    
    # Add trend line (polynomial regression using numpy)
    if show_trend_line and len(df_filtered) > 2:
        degree = 2 if trend_shape == "Quadratic" else 3
        if len(df_filtered) > degree:
            x_values = df_filtered["duration_days"].values
            y_values = df_filtered["reach"].values
            
            # Fit polynomial regression on log-transformed reach
            coeffs = np.polyfit(x_values, np.log10(y_values + 1), degree)
            
            # Generate trend line points
            x_trend = np.linspace(df_filtered["duration_days"].min(), df_filtered["duration_days"].max(), 200)
            y_trend_log = np.polyval(coeffs, x_trend)
            y_trend = np.power(10, y_trend_log) - 1  # Convert back from log scale
            
            # Add trend line to plot
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name=f'{trend_shape} Trend',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=True,
            ))
    
    fig.update_layout(
        height=600,
        hovermode="closest",
    )
    fig.update_yaxes(
        type="log",
        range=[np.log10(1000), None],  # Start at 1,000 on log scale
        title="Reach",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_map_section():
    """Render correlation heatmap for brand-level reach and duration signals."""
    st.subheader("🔗 Reach vs Duration Correlation Map")
    st.caption("Aggregated per-brand metrics to illustrate how reach and duration move together.")

    df_clean = _load_filtered_ads_data()
    if df_clean is None or df_clean.empty:
        st.info("No ads data available for correlation analysis.")
        return

    brand_metrics = (
        df_clean.groupby("brand")
        .agg(
            total_ads=("reach", "count"),
            avg_reach=("reach", "mean"),
            median_reach=("reach", "median"),
            max_reach=("reach", "max"),
            avg_duration=("duration_days", "mean"),
            median_duration=("duration_days", "median"),
        )
        .dropna()
    )

    if brand_metrics.empty or len(brand_metrics) < 2:
        st.info("Not enough brand data points to compute correlation.")
        return

    corr_matrix = brand_metrics.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        title="Correlation Map of Brand-Level Ad Metrics",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Brand metric summary"):
        st.dataframe(
            brand_metrics.round(1).sort_values("avg_reach", ascending=False),
            use_container_width=True,
        )

    overall_corr = df_clean[["reach", "duration_days"]].corr().iloc[0, 1]
    st.caption(
        f"Overall reach vs duration correlation across filtered ads: **{overall_corr:.2f}** "
        "(positive values suggest longer campaigns generally yield higher reach)."
    )


def _render_alerts_section():
    """Render a prototype alerts panel with illustrative blinking alerts."""
    st.subheader("Prototype Alerts")
    st.caption("Demonstrates the types of automated alerts you could enable later (with blinking animations).")

    # Inject CSS for blinking animation
    st.markdown("""
    <style>
        @keyframes blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.3; }
        }
        .blinking-alert {
            animation: blink 2s infinite;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid;
        }
        .alert-warning {
            background-color: #FFF4E6;
            border-color: #FF9800;
            color: #E65100;
        }
        .alert-error {
            background-color: #FFEBEE;
            border-color: #F44336;
            color: #C62828;
        }
        .alert-info {
            background-color: #E3F2FD;
            border-color: #2196F3;
            color: #1565C0;
        }
        .alert-success {
            background-color: #E8F5E9;
            border-color: #4CAF50;
            color: #2E7D32;
        }
    </style>
    """, unsafe_allow_html=True)

    alerts = [
        (
            "warning",
            "Reach Drop Alert",
            "Ignitis reach fell 35% week-over-week — consider refreshing LinkedIn creative.",
        ),
        (
            "error",
            "Duration Spike",
            "Teltonika has 12 ads running for 60+ days. Review fatigue risk and rotation cadence.",
        ),
        (
            "info",
            "Format Opportunity",
            "Kauno Energija video ads outperform static posts by 2.4× engagement — replicate across markets.",
        ),
        (
            "success",
            "Performance Milestone",
            "Exergi reached 1M total reach this month — celebrate this achievement!",
        ),
    ]

    for alert_type, title, description in alerts:
        st.markdown(
            f'<div class="blinking-alert alert-{alert_type}">'
            f'<strong>{title}</strong><br>{description}'
            f'</div>',
            unsafe_allow_html=True
        )


def _render_wordcloud_section():
    st.subheader("LinkedIn Word Clouds")
    st.caption(
        "Explores the language used across LinkedIn posts. The overview combines all brands, "
        "while the tabs highlight brand-specific vocabularies."
    )

    start_date, end_date = get_selected_date_range()
    
    # Load preprocessed text (cached - only loads once)
    with st.spinner("Loading post text..."):
        brand_texts_all = _load_all_linkedin_text()
    
    # Fallback to original method if new method doesn't work
    if not brand_texts_all:
        # Try using the original load_all_social_data as fallback
        from utils.file_io import load_all_social_data
        brand_dfs = load_all_social_data(BRANDS, platform="linkedin")
        
        if not brand_dfs:
            st.info("No LinkedIn data available.")
            return
        
        # Extract text from DataFrames (slower but works)
        brand_texts_all = {}
        for brand, df in brand_dfs.items():
            if "Published Date" not in df.columns:
                continue
            df = df.copy()
            df = df.dropna(subset=["Published Date"])
            
            # Find text column
            text_col = None
            for col in POST_TEXT_COLUMNS:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                continue
            
            texts = []
            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                if text and text.lower() not in ["nan", "none", ""]:
                    texts.append((row["Published Date"], text))
            
            if texts:
                brand_texts_all[brand] = texts
        
        if not brand_texts_all:
            st.info("No LinkedIn data available.")
            return

    # Filter by date range (fast - no file I/O)
    combined_text = ""
    brand_text_map = {}
    
    for brand, texts in brand_texts_all.items():
        text = _filter_texts_by_date(texts, start_date, end_date)
        if text and text.strip():
            brand_text_map[brand] = text
            combined_text += f" {text}"

    if not combined_text.strip():
        st.info("No post text available for the selected date range.")
        return

    tabs = st.tabs(["Overview"] + [f"{brand}" for brand in brand_text_map.keys()])

    with tabs[0]:
        with st.spinner("Generating overview word cloud..."):
            fig = _generate_wordcloud(combined_text)
        if fig:
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Unable to generate the overview word cloud.")

    for idx, (brand, text) in enumerate(brand_text_map.items(), start=1):
        with tabs[idx]:
            with st.spinner(f"Generating word cloud for {brand}..."):
                fig = _generate_wordcloud(text, brand_color=BRAND_COLORS.get(brand))
            if fig:
                st.pyplot(fig, clear_figure=True)
            else:
                st.info(f"Unable to render the word cloud for {brand}.")


def render():
    st.title("Testing Ground")
    st.write(
        "A sandbox for experimental visuals. Currently includes density plots, correlation heatmaps, "
        "prototype alerts, and LinkedIn word clouds."
    )
    _render_density_plots_section()
    st.divider()
    _render_correlation_map_section()
    st.divider()
    _render_alerts_section()
    st.divider()
    _render_wordcloud_section()

