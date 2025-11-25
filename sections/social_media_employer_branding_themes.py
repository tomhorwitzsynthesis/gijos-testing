import os
import streamlit as st
import pandas as pd
import html
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from difflib import SequenceMatcher
from utils.config import (
    BRAND_NAME_MAPPING,
    DATA_ROOT,
    BRAND_COLORS,
    PRIMARY_ACCENT_COLOR,
    POSITIVE_HIGHLIGHT_COLOR,
)

BRAND_ORDER = list(BRAND_COLORS.keys())


def _is_similar_to_any(text: str, existing_texts: list[str], threshold: float = 0.8) -> bool:
    if not text or not existing_texts:
        return False
    norm_text = str(text).strip().lower()
    for other in existing_texts:
        if not other:
            continue
        other_norm = str(other).strip().lower()
        if not other_norm:
            continue
        if SequenceMatcher(None, norm_text, other_norm).ratio() >= threshold:
            return True
    return False


MIN_POSTS_FOR_OVERVIEW = 8


@st.cache_data(ttl=0)
def _load_all_employer_branding_data():
    """Load employer branding data once and compute all needed structures in a single pass."""
    path = os.path.join(DATA_ROOT, "social_media", "employer_branding", "employer_branding_posts.xlsx")
    if not os.path.exists(path):
        return {}, {}, pd.DataFrame()
    
    try:
        df = pd.read_excel(path)
        
        # Check for required columns - try different possible column names for company
        company_col = None
        for col in ['user_id', 'pageName', 'company', 'brand']:
            if col in df.columns:
                company_col = col
                break
        
        if company_col is None or 'theme_1' not in df.columns:
            return {}, {}, pd.DataFrame()
        
        # Filter out rows with missing theme_1 (at least one theme should exist)
        df = df[df['theme_1'].notna()].copy()
        
        if df.empty:
            return {}, {}, pd.DataFrame()
        
        # Initialize data structures
        company_theme_pairs = []  # For company-theme counts DataFrame
        theme_company_counts = {}  # For theme -> company counts (theme_distribution)
        company_theme_examples = {}  # For storing examples by company and theme
        posts_per_company = Counter()
        
        # Process each row once - single pass
        for _, row in df.iterrows():
            company = row[company_col]
            if pd.isna(company):
                continue
            
            company_normalized = BRAND_NAME_MAPPING.get(str(company), str(company))
            posts_per_company[company_normalized] += 1
            post_text = str(row.get('post_text', '')).strip() if 'post_text' in row and pd.notna(row.get('post_text')) else ""
            post_url = str(row.get('url', '')).strip() if 'url' in row and pd.notna(row.get('url')) else None
            
            # Process all theme columns
            for theme_col in ['theme_1', 'theme_2', 'theme_3']:
                if theme_col not in df.columns:
                    continue
                if pd.isna(row[theme_col]):
                    continue
                
                theme = str(row[theme_col]).strip()
                if not theme or theme == 'nan':
                    continue
                
                # Add to company-theme pairs for DataFrame
                company_theme_pairs.append({
                    'Company': company_normalized,
                    'Theme': theme
                })
                
                # Update theme distribution
                if theme not in theme_company_counts:
                    theme_company_counts[theme] = Counter()
                theme_company_counts[theme][company_normalized] += 1
                
                # Store examples if post_text is available
                if post_text:
                    key = (company_normalized, theme)
                    if key not in company_theme_examples:
                        company_theme_examples[key] = []
                    if len(company_theme_examples[key]) < 8:
                        company_theme_examples[key].append({
                            'text': post_text[:150] + "..." if len(post_text) > 150 else post_text,
                            'raw_text': post_text,
                            'url': post_url
                        })
        
        # Convert theme_company_counts to dict format
        theme_distribution_dict = {theme: dict(counts) for theme, counts in theme_company_counts.items()}
        
        # Create counts DataFrame
        if company_theme_pairs:
            data_df = pd.DataFrame(company_theme_pairs)
            counts_df = data_df.groupby(['Company', 'Theme']).size().reset_index(name='Count')
        else:
            counts_df = pd.DataFrame()
        
        # Build themes_by_company with top 5 and examples
        themes_by_company = {}
        company_example_memory: dict[str, list[str]] = {}
        for company_normalized in set(pair['Company'] for pair in company_theme_pairs):
            # Get all themes for this company
            company_themes = [p['Theme'] for p in company_theme_pairs if p['Company'] == company_normalized]
            if not company_themes:
                continue
            
            # Count theme occurrences
            theme_counts = Counter(company_themes)
            total = len(company_themes)
            
            # Get top 5 themes
            top_5 = []
            company_example_memory.setdefault(company_normalized, [])
            for theme, count in theme_counts.most_common(5):
                percentage = (count / total) * 100 if total > 0 else 0
                
                # Get examples from stored examples
                examples = []
                key = (company_normalized, theme)
                candidates = company_theme_examples.get(key, [])
                selected_raws = []
                for ex in candidates:
                    raw_text = ex.get('raw_text') or ex.get('text')
                    if _is_similar_to_any(raw_text, selected_raws):
                        continue
                    if _is_similar_to_any(raw_text, company_example_memory[company_normalized]):
                        continue
                    examples.append({
                        'text': ex.get('text', ''),
                        'url': ex.get('url')
                    })
                    selected_raws.append(raw_text)
                    company_example_memory[company_normalized].append(raw_text)
                    if len(examples) >= 2:
                        break
                if len(examples) < 2:
                    for ex in candidates:
                        raw_text = ex.get('raw_text') or ex.get('text')
                        if raw_text in selected_raws:
                            continue
                        examples.append({
                            'text': ex.get('text', ''),
                            'url': ex.get('url')
                        })
                        selected_raws.append(raw_text)
                        company_example_memory[company_normalized].append(raw_text)
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
        
        return themes_by_company, theme_distribution_dict, counts_df, dict(posts_per_company)
    except Exception as e:
        st.error(f"Error loading employer branding data: {e}")
        return {}, {}, pd.DataFrame(), {}


def _render_theme_distribution_charts(theme_distribution):
    """Render pie charts showing company distribution for each theme."""
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


def _render_company_theme_stacked_bar(counts_df):
    """Render employer branding posts charts (total + 100% stacked)."""
    if counts_df.empty:
        st.info("No data available for stacked bar chart.")
        return
    
    st.markdown("### Employer Branding Posts by Company and Theme")
    st.markdown("Compare total post volume and theme mix for each company.")
    st.caption(
        "Posts in this section were selected when they directly or indirectly "
        "address employer branding topics. The three charts below focus solely "
        "on these posts to show total volumes, theme mixes, and the split by theme."
    )
    
    pivot_df = counts_df.pivot(index='Company', columns='Theme', values='Count').fillna(0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    totals = pivot_df['Total'].copy()
    pivot_df = pivot_df.drop(columns='Total')
    
    all_themes = sorted(pivot_df.columns.tolist())
    theme_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark2
    theme_color_map = {theme: theme_colors[i % len(theme_colors)] for i, theme in enumerate(all_themes)}
    
    tabs = st.tabs(["Total Volume", "Theme Mix (100%)"])
    
    with tabs[0]:
        st.markdown("#### Total Posts per Company")
        fig_total = px.bar(
            x=totals.index,
            y=totals.values,
            labels={'x': 'Company', 'y': 'Posts'},
            color_discrete_sequence=[PRIMARY_ACCENT_COLOR]
        )
        fig_total.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=100),
            xaxis=dict(tickangle=-45)
        )
        fig_total.update_traces(hovertemplate='<b>%{x}</b><br>Posts: %{y}<extra></extra>')
        st.plotly_chart(fig_total, use_container_width=True)
    
    with tabs[1]:
        st.markdown("#### Theme Mix (Share of Posts)")
        percent_df = pivot_df.div(pivot_df.sum(axis=1).replace(0, 1), axis=0) * 100
        plot_df = percent_df.reset_index()
        fig_mix = go.Figure()
        for theme in all_themes:
            fig_mix.add_trace(go.Bar(
                name=theme,
                x=plot_df['Company'],
                y=plot_df[theme],
                marker_color=theme_color_map[theme],
                customdata=pivot_df[theme].values,
                hovertemplate='<b>%{x}</b><br>Theme: %{fullData.name}<br>Share: %{y:.1f}%<br>Posts: %{customdata}<extra></extra>'
            ))
        fig_mix.update_layout(
            barmode='stack',
            title='Theme Mix per Company (100% stacked)',
            xaxis_title='Company',
            yaxis_title='Share of Posts (%)',
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
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_mix, use_container_width=True)


def render():
    """Render employer branding themes overview with tabs per company showing top 5 themes."""
    themes_data, theme_distribution, counts_df, posts_per_company = _load_all_employer_branding_data()
    
    # First render the stacked bar chart
    _render_company_theme_stacked_bar(counts_df)
    
    st.markdown("---")
    
    # Then render the pie charts
    _render_theme_distribution_charts(theme_distribution)
    
    st.markdown("---")
    
    # Then render the company overview
    if not themes_data:
        st.info("No employer branding themes data available.")
        return
    
    st.markdown("### Employer Branding Themes Overview by Company")
    st.caption(f"Only companies with at least {MIN_POSTS_FOR_OVERVIEW} employer branding posts are shown below.")
    
    # Get company names and order them (prefer BRAND_ORDER if available)
    companies = [c for c in themes_data.keys() if posts_per_company.get(c, 0) >= MIN_POSTS_FOR_OVERVIEW]
    ordered_companies = [c for c in BRAND_ORDER if c in companies]
    extra_companies = sorted([c for c in companies if c not in BRAND_ORDER])
    tab_labels = ordered_companies + extra_companies
    
    if not tab_labels:
        st.info(f"No companies have at least {MIN_POSTS_FOR_OVERVIEW} employer branding posts to show detailed themes.")
        return
    
    tabs = st.tabs(tab_labels)
    
    for idx, company in enumerate(tab_labels):
        with tabs[idx]:
            themes = themes_data.get(company, [])
            
            if not themes:
                st.info(f"No themes data available for {company}.")
                continue
            
            total_posts = posts_per_company.get(company, 0)
            st.subheader(f"{company} Employer Branding Themes")
            st.markdown(f"<p style='color:#5F6B7C;'>Total posts analyzed: {total_posts}</p>", unsafe_allow_html=True)
            
            # Display each theme with metrics and examples stacked vertically
            for theme_item in themes:
                theme_name = html.escape(str(theme_item['theme']))
                percentage = theme_item['percentage']
                count = theme_item['count']
                examples = theme_item.get('examples', [])
                
                example_cards = []
                if examples:
                    for example in examples:
                        if isinstance(example, dict):
                            example_text = example.get('text', '')
                            example_url = example.get('url')
                        else:
                            example_text = str(example)
                            example_url = None
                        
                        escaped_text = html.escape(example_text)
                        
                        if example_url:
                            escaped_url = html.escape(str(example_url))
                            example_html = f'<a href="{escaped_url}" target="_blank" style="color:{PRIMARY_ACCENT_COLOR}; text-decoration:none;">"{escaped_text}"</a>'
                        else:
                            example_html = f'"{escaped_text}"'
                        
                        example_cards.append(
                            f'<div style="border:1px solid #e0e0e0; border-radius:10px; padding:12px; margin-bottom:10px; '
                            f'background-color:#fafafa; font-style:italic; color:#555;">\n'
                            f'    <p style="margin:0; font-size:0.9em;">{example_html}</p>\n'
                            f'</div>'
                        )
                else:
                    example_cards.append("<p style='margin:4px 0 0; color:#666;'>No examples available.</p>")
                
                example_cards_html = "\n".join(example_cards)
                
                card_html = (
                    f'<div style="border:1px solid #ddd; border-radius:12px; padding:18px; margin-bottom:16px; '
                    f'background-color:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.06);">\n'
                    f'    <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">\n'
                    f'        <div>\n'
                    f'            <h5 style="margin:0; color:{PRIMARY_ACCENT_COLOR}; font-weight:600;">{theme_name}</h5>\n'
                    f'        </div>\n'
                    f'        <div style="text-align:right;">\n'
                    f'            <p style="margin:0; font-size:1.8em; color:{POSITIVE_HIGHLIGHT_COLOR}; font-weight:700;">{percentage:.1f}%</p>\n'
                    f'            <p style="margin:4px 0 0; color:#666; font-size:0.9em;">{count} posts</p>\n'
                    f'        </div>\n'
                    f'    </div>\n'
                    f'    <div style="margin-top:12px;">\n'
                    f'{example_cards_html}\n'
                    f'    </div>\n'
                    f'</div>'
                )
                
                st.markdown(card_html, unsafe_allow_html=True)

