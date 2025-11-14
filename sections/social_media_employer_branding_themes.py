import os
import streamlit as st
import pandas as pd
import html
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from utils.config import BRAND_NAME_MAPPING, DATA_ROOT, BRAND_COLORS

BRAND_ORDER = list(BRAND_COLORS.keys())


@st.cache_data(ttl=0)
def _load_employer_branding_themes_data():
    """Load employer branding themes data from employer_branding_posts.xlsx and return dict of company -> list of top 5 themes with percentages and examples."""
    path = os.path.join(DATA_ROOT, "social_media", "employer_branding", "employer_branding_posts.xlsx")
    if not os.path.exists(path):
        return {}
    
    try:
        df = pd.read_excel(path)
        
        # Check for required columns - try different possible column names for company
        company_col = None
        for col in ['user_id', 'pageName', 'company', 'brand']:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None or 'theme_1' not in df.columns or 'post_text' not in df.columns:
            return {}
        
        # Filter out rows with missing theme_1 (at least one theme should exist)
        df = df[df['theme_1'].notna()].copy()
        
        if df.empty:
            return {}
        
        # Track used post texts to ensure no duplicates across all companies
        used_post_texts = set()
        
        # Group by company and count all themes (from theme_1, theme_2, theme_3)
        themes_by_company = {}
        
        for company in df[company_col].unique():
            if pd.isna(company):
                continue
            
            company_df = df[df[company_col] == company]
            
            # Collect all themes from theme_1, theme_2, theme_3
            all_themes = []
            for col in ['theme_1', 'theme_2', 'theme_3']:
                if col in company_df.columns:
                    themes = company_df[col].dropna().tolist()
                    all_themes.extend(themes)
            
            if not all_themes:
                continue
            
            # Count theme occurrences
            theme_counts = Counter(all_themes)
            total = len(all_themes)
            
            # Normalize company name using BRAND_NAME_MAPPING
            company_normalized = BRAND_NAME_MAPPING.get(str(company), str(company))
            
            # Calculate percentages and get top 5
            top_5 = []
            for theme, count in theme_counts.most_common(5):
                percentage = (count / total) * 100 if total > 0 else 0
                
                # Get 2 unique examples for this theme
                examples = []
                # Check all theme columns to find posts with this theme
                for theme_col in ['theme_1', 'theme_2', 'theme_3']:
                    if theme_col not in company_df.columns:
                        continue
                    theme_df = company_df[company_df[theme_col] == theme]
                    # Filter out rows with missing post_text
                    theme_df = theme_df[theme_df['post_text'].notna()].copy()
                    
                    for _, row in theme_df.iterrows():
                        if len(examples) >= 2:
                            break
                        
                        post_text = str(row['post_text']).strip()
                        if not post_text or post_text == 'nan':
                            continue
                        
                        # Check if this exact text has been used before
                        if post_text not in used_post_texts:
                            # Truncate to 150 characters (longer than before)
                            if len(post_text) > 150:
                                post_text_display = post_text[:150] + "..."
                            else:
                                post_text_display = post_text
                            
                            # Get URL if available
                            post_url = None
                            if 'url' in row and pd.notna(row['url']):
                                post_url = str(row['url']).strip()
                            
                            examples.append({
                                'text': post_text_display,
                                'url': post_url
                            })
                            used_post_texts.add(post_text)
                    
                    if len(examples) >= 2:
                        break
                
                top_5.append({
                    'theme': str(theme),
                    'percentage': percentage,
                    'count': int(count),
                    'examples': examples
                })
            
            if top_5:
                themes_by_company[company_normalized] = top_5
        
        return themes_by_company
    except Exception as e:
        st.error(f"Error loading employer branding themes data: {e}")
        return {}


@st.cache_data(ttl=0)
def _load_theme_distribution_data():
    """Load employer branding themes data and return dict of theme -> list of (company, count) tuples."""
    path = os.path.join(DATA_ROOT, "social_media", "employer_branding", "employer_branding_posts.xlsx")
    if not os.path.exists(path):
        return {}
    
    try:
        df = pd.read_excel(path)
        
        # Check for required columns - try different possible column names for company
        company_col = None
        for col in ['user_id', 'pageName', 'company', 'brand']:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None or 'theme_1' not in df.columns:
            return {}
        
        # Filter out rows with missing theme_1 (at least one theme should exist)
        df = df[df['theme_1'].notna()].copy()
        
        if df.empty:
            return {}
        
        # Get all unique themes
        all_themes = set()
        for col in ['theme_1', 'theme_2', 'theme_3']:
            if col in df.columns:
                themes = df[col].dropna().unique().tolist()
                all_themes.update(themes)
        
        # For each theme, count posts per company (checking all 3 theme columns)
        theme_distribution = {}
        
        for theme in all_themes:
            if pd.isna(theme):
                continue
            
            company_counts = Counter()
            
            # Check each row to see if this theme appears in any of the theme columns
            for _, row in df.iterrows():
                company = row[company_col]
                if pd.isna(company):
                    continue
                
                # Check if this theme appears in theme_1, theme_2, or theme_3
                theme_found = False
                for theme_col in ['theme_1', 'theme_2', 'theme_3']:
                    if theme_col in df.columns:
                        if pd.notna(row[theme_col]) and str(row[theme_col]).strip() == str(theme).strip():
                            theme_found = True
                            break
                
                if theme_found:
                    # Normalize company name using BRAND_NAME_MAPPING
                    company_normalized = BRAND_NAME_MAPPING.get(str(company), str(company))
                    company_counts[company_normalized] += 1
            
            if company_counts:
                theme_distribution[str(theme)] = dict(company_counts)
        
        return theme_distribution
    except Exception as e:
        st.error(f"Error loading theme distribution data: {e}")
        return {}


def _render_theme_distribution_charts():
    """Render pie charts showing company distribution for each theme."""
    theme_distribution = _load_theme_distribution_data()
    
    if not theme_distribution:
        st.info("No theme distribution data available.")
        return
    
    st.markdown("### Theme Distribution by Company")
    st.markdown("Pie charts showing how many posts each company made for each employer branding theme.")
    
    # Get all themes and sort them
    all_themes = sorted(theme_distribution.keys())
    
    if not all_themes:
        st.info("No themes available.")
        return
    
    # Create tabs for each theme
    tabs = st.tabs(all_themes)
    
    for idx, theme in enumerate(all_themes):
        with tabs[idx]:
            company_counts = theme_distribution[theme]
            
            if not company_counts:
                st.info(f"No data available for theme: {theme}")
                continue
            
            # Prepare data for pie chart
            companies = list(company_counts.keys())
            counts = list(company_counts.values())
            
            # Create DataFrame for plotly
            chart_df = pd.DataFrame({
                'Company': companies,
                'Posts': counts
            })
            
            # Sort by count descending
            chart_df = chart_df.sort_values('Posts', ascending=False)
            
            # Create color mapping based on BRAND_COLORS for consistent coloring
            color_map = {}
            for company in chart_df['Company']:
                color_map[company] = BRAND_COLORS.get(company, '#BDBDBD')  # Default gray if not found
            
            # Create pie chart with consistent colors
            fig = px.pie(
                chart_df,
                values='Posts',
                names='Company',
                title=f'Distribution of "{theme}" Posts by Company',
                color='Company',
                color_discrete_map=color_map
            )
            
            # Update layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Posts: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                showlegend=True,
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=0)
def _load_company_theme_counts():
    """Load employer branding themes data and return DataFrame with company, theme, and count."""
    path = os.path.join(DATA_ROOT, "social_media", "employer_branding", "employer_branding_posts.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(path)
        
        # Check for required columns - try different possible column names for company
        company_col = None
        for col in ['user_id', 'pageName', 'company', 'brand']:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None or 'theme_1' not in df.columns:
            return pd.DataFrame()
        
        # Filter out rows with missing theme_1 (at least one theme should exist)
        df = df[df['theme_1'].notna()].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        # Collect all data: company, theme pairs
        data_rows = []
        
        for _, row in df.iterrows():
            company = row[company_col]
            if pd.isna(company):
                continue
            
            # Normalize company name
            company_normalized = BRAND_NAME_MAPPING.get(str(company), str(company))
            
            # Check all theme columns
            for theme_col in ['theme_1', 'theme_2', 'theme_3']:
                if theme_col in df.columns and pd.notna(row[theme_col]):
                    theme = str(row[theme_col]).strip()
                    if theme and theme != 'nan':
                        data_rows.append({
                            'Company': company_normalized,
                            'Theme': theme
                        })
        
        if not data_rows:
            return pd.DataFrame()
        
        # Create DataFrame and count occurrences
        data_df = pd.DataFrame(data_rows)
        counts_df = data_df.groupby(['Company', 'Theme']).size().reset_index(name='Count')
        
        return counts_df
    except Exception as e:
        st.error(f"Error loading company theme counts: {e}")
        return pd.DataFrame()


def _render_company_theme_stacked_bar():
    """Render stacked bar chart showing number of posts per company, stacked by theme."""
    counts_df = _load_company_theme_counts()
    
    if counts_df.empty:
        st.info("No data available for stacked bar chart.")
        return
    
    st.markdown("### Employer Branding Posts by Company and Theme")
    st.markdown("Stacked bar chart showing total posts per company, divided by theme.")
    
    # Pivot the data to have companies as rows and themes as columns
    pivot_df = counts_df.pivot(index='Company', columns='Theme', values='Count').fillna(0)
    
    # Sort companies by total posts (descending)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    pivot_df = pivot_df.drop('Total', axis=1)
    
    # Get all themes and create a color map
    all_themes = sorted(pivot_df.columns.tolist())
    
    # Create a color palette for themes (using a distinct color scheme)
    # Use a larger color palette that cycles if needed
    theme_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark2
    theme_color_map = {theme: theme_colors[i % len(theme_colors)] for i, theme in enumerate(all_themes)}
    
    # Create stacked bar chart
    # Reset index to make Company a column
    plot_df = pivot_df.reset_index()
    
    # Create the figure with go.Bar for better control
    fig = go.Figure()
    
    # Add a trace for each theme
    for theme in all_themes:
        fig.add_trace(go.Bar(
            name=theme,
            x=plot_df['Company'],
            y=plot_df[theme],
            marker_color=theme_color_map[theme]
        ))
    
    # Update layout for stacked bars
    fig.update_layout(
        barmode='stack',
        title='Number of Employer Branding Posts by Company and Theme',
        xaxis_title='Company',
        yaxis_title='Number of Posts',
        height=500,
        margin=dict(l=20, r=20, t=50, b=100),
        legend=dict(
            title='Theme',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        xaxis=dict(tickangle=-45)
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Company: %{x}<br>Posts: %{y}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary table
    st.markdown("#### Summary Table")
    summary_df = pivot_df.copy()
    summary_df['Total Posts'] = summary_df.sum(axis=1)
    # Calculate percentages for each theme
    for theme in all_themes:
        summary_df[f'{theme} %'] = (summary_df[theme] / summary_df['Total Posts'] * 100).round(1)
    
    # Reorder columns: Total, then themes with counts, then themes with percentages
    display_cols = ['Total Posts'] + all_themes + [f'{t} %' for t in all_themes]
    summary_df = summary_df[display_cols]
    st.dataframe(summary_df, use_container_width=True)


def render():
    """Render employer branding themes overview with tabs per company showing top 5 themes."""
    # First render the stacked bar chart
    _render_company_theme_stacked_bar()
    
    st.markdown("---")
    
    # Then render the pie charts
    _render_theme_distribution_charts()
    
    st.markdown("---")
    
    # Then render the company overview
    themes_data = _load_employer_branding_themes_data()
    
    if not themes_data:
        st.info("No employer branding themes data available.")
        return
    
    st.markdown("### Employer Branding Themes Overview by Company")
    
    # Get company names and order them (prefer BRAND_ORDER if available)
    companies = list(themes_data.keys())
    ordered_companies = [c for c in BRAND_ORDER if c in companies]
    extra_companies = sorted([c for c in companies if c not in BRAND_ORDER])
    tab_labels = ordered_companies + extra_companies
    
    if not tab_labels:
        st.info("No company data available.")
        return
    
    tabs = st.tabs(tab_labels)
    
    for idx, company in enumerate(tab_labels):
        with tabs[idx]:
            themes = themes_data.get(company, [])
            
            if not themes:
                st.info(f"No themes data available for {company}.")
                continue
            
            st.subheader(f"{company} Employer Branding Themes")
            
            # Display each theme - category and examples on left, percentage on right
            for theme_item in themes:
                theme_name = html.escape(str(theme_item['theme']))
                percentage = theme_item['percentage']
                count = theme_item['count']
                examples = theme_item.get('examples', [])
                
                # Create two columns: left for category + examples, right for percentage
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Box for theme name (same width as examples)
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#f9f9f9; box-shadow:0 2px 4px rgba(0,0,0,0.08);">
                            <h5 style="margin:0; color:#333; font-weight:500;">{theme_name}</h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display examples below the theme (same width as category box)
                    if examples:
                        for example in examples:
                            # Handle both dict format (new) and string format (old, for backward compatibility)
                            if isinstance(example, dict):
                                example_text = example.get('text', '')
                                example_url = example.get('url')
                            else:
                                # Backward compatibility with old string format
                                example_text = str(example)
                                example_url = None
                            
                            # Escape HTML to prevent injection
                            escaped_text = html.escape(example_text)
                            
                            # Create link if URL is available
                            if example_url:
                                escaped_url = html.escape(str(example_url))
                                example_html = f'<a href="{escaped_url}" target="_blank" style="color:#2FB375; text-decoration:none;">"{escaped_text}"</a>'
                            else:
                                example_html = f'"{escaped_text}"'
                            
                            st.markdown(
                                f"""
                                <div style="border:1px solid #e0e0e0; border-radius:10px; padding:12px; margin-bottom:10px; background-color:#fafafa; font-style:italic; color:#555;">
                                    <p style="margin:0; font-size:0.9em;">{example_html}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                with col2:
                    # Calculate number of boxes on left (1 category + examples)
                    num_boxes = 1 + len(examples)
                    # Box for percentage - use flexbox to match height
                    st.markdown(
                        f"""
                        <div style="border:1px solid #2FB375; border-radius:10px; padding:15px; background-color:#F5FFF9; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.08); display:flex; flex-direction:column; justify-content:center; min-height:{num_boxes * 60}px;">
                            <h3 style="margin:0; color:#2FB375; font-weight:bold;">{percentage:.1f}%</h3>
                            <p style="margin:4px 0 0; color:#666; font-size:0.85em;">{count} posts</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

