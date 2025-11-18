# sections/audience_affinity.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.config import BRAND_NAME_MAPPING, BRAND_COLORS
from utils.file_io import load_audience_affinity_outputs

def format_percentage(series):
    return series.round(0).astype(int).astype(str) + "%"

def prettify_column(col):
    # Remove '_%High' and prettify
    col = col.replace("_%High", "")
    col = col.replace("_", " ")
    col = col.replace("Customer", "Customer")
    col = col.replace("Talent", "Talent")
#    col = col.replace("Pro", "Professional")
    col = col.replace("Investor", "Investor")
    col = col.replace("Employer Branding", "Employer Branding")
    col = col.replace("Career Growth", "Career Growth")
    col = col.replace("Market Impact", "Market Impact")
    col = col.replace("Problem Solving", "Problem Solving")
    col = col.replace("Clarity Offerings", "Clarity of Offerings")
    col = col.replace("Innovation", "Innovation")
    col = col.replace("Expertise", "Expertise")
    col = col.replace("Industry Relevance", "Industry Relevance")
    col = col.replace("Long Term", "Long-Term Vision")
    col = col.replace("Positioning", "Positioning")
    col = col.replace("Market Influence", "Market Influence")
    return col

def render():
    try:
        affinity_data = load_audience_affinity_outputs()
        if affinity_data is None:
            st.error("❌ No audience affinity data available.")
            return

        summary_df = affinity_data.get("summary_df")
        gpt_summary = affinity_data.get("gpt_summary")

        if summary_df is None or summary_df.empty:
            st.error("❌ No summary data available.")
            return

        summary_df["Brand"] = summary_df["Brand"].map(
            lambda x: BRAND_NAME_MAPPING.get(x, x)
        )

        st.subheader("🔍 Audience Averages View")

        # Overview text
        st.markdown(
            "Percentages below represent the share of respondents who rated each aspect in the top 2 boxes (6 or 7) on a 1–7 scale. "
            "This metric helps identify which brands resonate most strongly with different audience segments across key dimensions. "
            "Higher percentages indicate stronger alignment and positive perception within each audience group."
        )

        audience_map = {
            "Customers & End Users": {
                "pct_col": "Customers & End Users_%High",
                "detail_cols": ["Customer_Problem_Solving", "Customer_Clarity_Offerings", "Customer_Innovation"]
            },
            "Job Seekers & Talent": {
                "pct_col": "Job Seekers & Talent_%High",
                "detail_cols": ["Talent_Employer_Branding", "Talent_Career_Growth", "Talent_Market_Impact"]
            },
            "Professionals": {
                "pct_col": "Professionals_%High",
                "detail_cols": ["Pro_Expertise", "Pro_Industry_Relevance", "Pro_Innovation"]
            },
            "Decision Makers & Investors": {
                "pct_col": "Decision Makers & Investors_%High",
                "detail_cols": ["Investor_Long_Term", "Investor_Positioning", "Investor_Market_Influence"]
            }
        }

        # Create tabs for each audience plus a full table tab
        audience_tabs = st.tabs([
            "Customers & End Users",
            "Job Seekers & Talent",
            "Professionals",
            "Decision Makers & Investors",
            "Full Table"
        ])
        
        # Process each audience tab
        for tab_idx, (audience_name, tab) in enumerate(zip(
            ["Customers & End Users", "Job Seekers & Talent", "Professionals", "Decision Makers & Investors"],
            audience_tabs[:4]
        )):
            with tab:
                mapping = audience_map.get(audience_name, {})
                detail_cols = mapping.get("detail_cols", [])
                
                # Show detailed breakdown for each company (horizontal orientation)
                if detail_cols:
                    # Create data for detailed metrics
                    detail_data = []
                    for _, row in summary_df.iterrows():
                        brand = row["Brand"]
                        for detail_col in detail_cols:
                            if detail_col in summary_df.columns:
                                detail_data.append({
                                    "Brand": brand,
                                    "Metric": prettify_column(detail_col),
                                    "Percentage": row[detail_col]
                                })
                    
                    if detail_data:
                        detail_df = pd.DataFrame(detail_data)
                        
                        # Create horizontal grouped bar chart (rotated 90 degrees)
                        fig_detail = px.bar(
                            detail_df,
                            x="Percentage",
                            y="Brand",
                            color="Metric",
                            orientation="h",
                            barmode="group",
                            labels={"Percentage": "Percentage (%)", "Brand": "Company"},
                            title=f"{audience_name} - Detailed Metrics Breakdown",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                        )
                        fig_detail.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            height=max(500, len(summary_df) * 90),  # Much more height per row to prevent overlap
                            hovermode="y unified",
                            bargap=0.3,  # Spacing between groups of bars (companies) - reduce for closer, increase for more space
                            bargroupgap=0.5,  # Spacing between bars within a group (metrics) - reduce for closer, increase for more space
                        )
                        # Remove decimals from percentages and ensure bars don't overlap
                        fig_detail.update_traces(
                            texttemplate="%{x:.0f}%",  # Remove decimals, round to whole numbers
                            textposition="outside",
                            textfont=dict(size=14),  # Increase percentage font size
                            width=0.2,  # Smaller bar width to prevent overlap within groups
                        )
                        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Full table tab with sub-tabs
        with audience_tabs[4]:
            # Create sub-tabs within Full Table tab
            table_tabs = st.tabs([
                "Averages",
                "Customers & End Users",
                "Job Seekers & Talent",
                "Professionals",
                "Decision Makers & Investors"
            ])
            
            # Averages sub-tab
            with table_tabs[0]:
                cols_to_show = [col for col in summary_df.columns if col.endswith("_%High")]
                display_df = summary_df[["Brand"] + cols_to_show].copy()
                # Format all percentage columns
                for col in cols_to_show:
                    display_df[col] = format_percentage(display_df[col])
                
                # Prettify column names
                display_df.columns = [prettify_column(col) if col != "Brand" else "Brand" for col in display_df.columns]
                st.dataframe(display_df.set_index("Brand"), use_container_width=True)
            
            # Individual audience sub-tabs
            for sub_tab_idx, audience_name in enumerate([
                "Customers & End Users",
                "Job Seekers & Talent",
                "Professionals",
                "Decision Makers & Investors"
            ], start=1):
                with table_tabs[sub_tab_idx]:
                    mapping = audience_map.get(audience_name, {})
                    pct_col = mapping.get("pct_col")
                    detail_cols = mapping.get("detail_cols", [])
                    
                    if pct_col and pct_col in summary_df.columns:
                        cols_to_show = [pct_col] + detail_cols
                        display_df = summary_df[["Brand"] + cols_to_show].copy()
                        display_df = display_df.rename(columns={pct_col: "Average Value"})
                        
                        # Format all percentage columns (including detail columns)
                        for col in ["Average Value"] + detail_cols:
                            if col in display_df.columns:
                                display_df[col] = format_percentage(display_df[col])
                        
                        # Prettify column names
                        display_df.columns = [prettify_column(col) if col != "Brand" else "Brand" for col in display_df.columns]
                        st.dataframe(display_df.set_index("Brand"), use_container_width=True)
                    else:
                        st.warning(f"No data available for {audience_name}")

        # GPT Summary with green box styling
        if gpt_summary:
            st.markdown("---")
            st.markdown("""
            <style>
                .summary-box {
                    background-color: #E8F5E9;
                    border: 2px solid #2FB375;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                }
            </style>
            """, unsafe_allow_html=True)
            st.subheader("🧠 Summary")
            st.markdown(f'<div class="summary-box">{gpt_summary}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error("🚨 Failed to load audience affinity data.")
        st.exception(e)
