import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.file_io import load_agility_data, load_agility_volume_map
from utils.config import BRANDS
import pandas as pd
from utils.date_utils import get_selected_date_range  # Add this import

def render():
    st.subheader("Brand Archetypes: Volume vs. Quality")

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

        summary[brand] = {
            "Volume": volume,
            "Quality": round(quality, 2) if pd.notna(quality) else 0,
        }

    if not summary:
        st.warning("No archetype data found.")
        return

    df_summary = pd.DataFrame.from_dict(summary, orient="index").reset_index()
    df_summary.columns = ["Company", "Volume", "Quality"]

    fig = px.scatter(
        df_summary,
        x="Volume",
        y="Quality",
        hover_data=["Company"],
        title="Company Positioning by Volume & Quality",
    )

    # Use Streamlit green color (#2FB375) for all dots
    fig.update_traces(marker=dict(size=12, opacity=0.8, color="#2FB375"))

    # Calculate adaptive offset based on quality range
    quality_range = df_summary["Quality"].max() - df_summary["Quality"].min()
    quality_max = df_summary["Quality"].max()
    quality_min = df_summary["Quality"].min()
    # Use 10% of the range as offset, with a minimum of 0.02 (increased for more spacing)
    base_offset = max(quality_range * 0.10, 0.02) if quality_range > 0 else 0.04
    
    # Calculate data ranges once for rounded rectangle sizing
    x_range = df_summary["Volume"].max() - df_summary["Volume"].min()
    y_range = df_summary["Quality"].max() - df_summary["Quality"].min()
    
    # Estimate card dimensions for collision detection
    # Approximate card width (based on text length) and height
    card_padding = 0.03  # Additional padding between cards
    estimated_card_height = y_range * 0.06  # Estimated height of card in data coordinates
    estimated_card_width = x_range * 0.10   # Estimated width of card in data coordinates
    
    # First pass: calculate initial label positions
    label_positions = []
    for idx, row in df_summary.iterrows():
        x_pos = row["Volume"]
        y_pos = row["Quality"]
        label_y = y_pos + base_offset
        label_positions.append({
            'index': idx,
            'x': x_pos,
            'y': label_y,
            'company': row['Company']
        })
    
    # Collision detection and resolution
    # Sort by x position to process left to right
    label_positions.sort(key=lambda p: p['x'])
    
    # Resolve overlaps by adjusting y positions
    for i in range(len(label_positions)):
        current = label_positions[i]
        
        # Check against all previous labels
        for j in range(i):
            other = label_positions[j]
            
            # Check if cards would overlap (simple rectangular collision)
            # Consider overlap if x positions are close AND y positions are close
            x_distance = abs(current['x'] - other['x'])
            y_distance = abs(current['y'] - other['y'])
            
            # Check if cards overlap in both dimensions
            if x_distance < estimated_card_width and y_distance < estimated_card_height:
                # Overlap detected! Move current label up
                min_y_needed = other['y'] + estimated_card_height + card_padding * y_range
                if current['y'] < min_y_needed:
                    current['y'] = min_y_needed
                
                # Also check if we're getting too close to the top
                # If so, try moving both slightly
                if current['y'] > quality_max + base_offset * 2:
                    # Try moving the other one down instead
                    new_other_y = current['y'] - estimated_card_height - card_padding * y_range
                    if new_other_y > other['y']:
                        other['y'] = new_other_y
                        current['y'] = other['y'] + estimated_card_height + card_padding * y_range
    
    # Add brand names above data points with connecting lines
    for pos in label_positions:
        idx = pos['index']
        row = df_summary.iloc[idx]
        x_pos = pos['x']
        y_pos = row["Quality"]
        label_y = pos['y']  # Use collision-resolved position
        
        # Add connecting line from data point to label
        fig.add_shape(
            type="line",
            x0=x_pos,
            y0=y_pos,
            x1=x_pos,
            y1=label_y,
            line=dict(color="gray", width=1.5),
            layer="below"
        )
        
        # Add brand name annotation above the data point with normal rectangle
        fig.add_annotation(
            x=x_pos,
            y=label_y,
            text=row['Company'],  # Removed bold tags
            showarrow=False,
            font=dict(size=20, color="black"),  # Increased font size, removed bold
            align="center",
            bgcolor="white",
            borderpad=6,
            bordercolor="lightgray",
            borderwidth=1,
            yanchor="middle"
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
