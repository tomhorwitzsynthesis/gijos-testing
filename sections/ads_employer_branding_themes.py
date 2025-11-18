import os
import streamlit as st
import pandas as pd
import html
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from difflib import SequenceMatcher
from utils.config import BRAND_NAME_MAPPING, DATA_ROOT, BRAND_COLORS

BRAND_ORDER = list(BRAND_COLORS.keys())


def _normalize_brand(name: str) -> str:
    """Normalize brand name to match ads dashboard normalization."""
    if not isinstance(name, str):
        return ""
    base = name.split("|")[0].strip()
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in base)
    return " ".join(cleaned.split())


def _normalize_company_name(company: str) -> str:
    """Normalize company name using BRAND_NAME_MAPPING, matching ads dashboard approach."""
    if not isinstance(company, str):
        return ""
    company_str = str(company).strip()
    
    # Use case-insensitive lookup like file_io.py does
    brand_lookup = {k.lower(): v for k, v in BRAND_NAME_MAPPING.items()}
    company_lower = company_str.lower()
    
    # Try case-insensitive match first (this is what file_io.py does)
    if company_lower in brand_lookup:
        return brand_lookup[company_lower]
    
    # Try exact match
    if company_str in BRAND_NAME_MAPPING:
        return BRAND_NAME_MAPPING[company_str]
    
    # Try normalized version
    normalized = _normalize_brand(company_str)
    if normalized in brand_lookup:
        return brand_lookup[normalized]
    
    # Try all variations in mapping (case-insensitive)
    for key, value in BRAND_NAME_MAPPING.items():
        if key.lower() == company_lower or _normalize_brand(key) == normalized:
            return value
    
    # If no match, try to find in BRAND_ORDER by normalized name
    for brand in BRAND_ORDER:
        if _normalize_brand(brand) == normalized:
            return brand
    
    # Return original if no match found
    return company_str


def _is_similar_to_any(text: str, existing_texts: list[str], threshold: float = 0.8) -> bool:
    """Return True if text is at least threshold similar to any string in existing_texts."""
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


@st.cache_data(ttl=0)
def _load_all_employer_branding_data():
    """Load employer branding ads once and prepare derived datasets."""
    path = os.path.join(DATA_ROOT, "ads", "employer_branding", "employer_branding_ads.xlsx")
    if not os.path.exists(path):
        return {}, {}, pd.DataFrame()
    
    try:
        df = pd.read_excel(path)
        
        company_col = None
        for col in ['pageName', 'user_id', 'company', 'brand']:
            if col in df.columns:
                company_col = col
                break
        
        text_col = None
        for col in ['snapshot/body/text', 'snapshot/body']:
            if col in df.columns:
                text_col = col
                break
        
        theme_columns = [col for col in ['theme_1', 'theme_2', 'theme_3'] if col in df.columns]
        
        if company_col is None or not theme_columns:
            return {}, {}, pd.DataFrame()
        
        df = df[df['theme_1'].notna()].copy()
        if df.empty:
            return {}, {}, pd.DataFrame()
        
        company_theme_pairs = []
        theme_company_counts = {}
        company_theme_examples = {}
        
        for _, row in df.iterrows():
            company = row[company_col]
            if pd.isna(company):
                continue
            company_normalized = _normalize_company_name(str(company))
            
            ad_text = ""
            if text_col and text_col in row and pd.notna(row[text_col]):
                ad_text = str(row[text_col]).strip()
            ad_id = None
            if 'adArchiveID' in row and pd.notna(row['adArchiveID']):
                try:
                    ad_id = str(int(row['adArchiveID']))
                except Exception:
                    ad_id = str(row['adArchiveID'])
            
            for theme_col in theme_columns:
                theme_value = row[theme_col]
                if pd.isna(theme_value):
                    continue
                theme = str(theme_value).strip()
                if not theme or theme == 'nan':
                    continue
                
                company_theme_pairs.append({'Company': company_normalized, 'Theme': theme})
                theme_company_counts.setdefault(theme, Counter())[company_normalized] += 1
                
                if ad_text:
                    key = (company_normalized, theme)
                    company_theme_examples.setdefault(key, [])
                    if len(company_theme_examples[key]) < 8:
                        display_text = ad_text[:150] + "..." if len(ad_text) > 150 else ad_text
                        company_theme_examples[key].append({
                            'text': display_text,
                            'raw_text': ad_text,
                            'ad_id': ad_id
                        })
        
        if not company_theme_pairs:
            return {}, {}, pd.DataFrame()
        
        data_df = pd.DataFrame(company_theme_pairs)
        counts_df = data_df.groupby(['Company', 'Theme']).size().reset_index(name='Count')
        
        themes_by_company = {}
        company_example_memory: dict[str, list[str]] = {}
        for company in data_df['Company'].unique():
            company_rows = data_df[data_df['Company'] == company]
            theme_counts = Counter(company_rows['Theme'])
            total = sum(theme_counts.values())
            if total == 0:
                continue
            
            top_items = []
            company_example_memory.setdefault(company, [])
            for theme, count in theme_counts.most_common(5):
                examples = []
                key = (company, theme)
                candidates = company_theme_examples.get(key, [])
                selected_raws = []
                # Primary pass: prefer examples that differ from existing selections and other themes
                for ex in candidates:
                    raw_text = ex.get('raw_text') or ex.get('text')
                    if _is_similar_to_any(raw_text, selected_raws):
                        continue
                    if _is_similar_to_any(raw_text, company_example_memory[company]):
                        continue
                    examples.append({
                        'text': ex.get('text', ''),
                        'ad_id': ex.get('ad_id')
                    })
                    selected_raws.append(raw_text)
                    company_example_memory[company].append(raw_text)
                    if len(examples) >= 2:
                        break
                # Fallback: allow similar ones if not enough unique options
                if len(examples) < 2:
                    for ex in candidates:
                        raw_text = ex.get('raw_text') or ex.get('text')
                        if raw_text in selected_raws:
                            continue
                        examples.append({
                            'text': ex.get('text', ''),
                            'ad_id': ex.get('ad_id')
                        })
                        selected_raws.append(raw_text)
                        company_example_memory[company].append(raw_text)
                        if len(examples) >= 2:
                            break
                top_items.append({
                    'theme': str(theme),
                    'percentage': (count / total) * 100 if total > 0 else 0,
                    'count': int(count),
                    'examples': examples
                })
            if top_items:
                themes_by_company[company] = top_items
        
        theme_distribution = {
            str(theme): dict(counter) for theme, counter in theme_company_counts.items()
        }
        
        return themes_by_company, theme_distribution, counts_df
    except Exception as e:
        st.error(f"Error loading employer branding themes data: {e}")
        return {}, {}, pd.DataFrame()


def _render_theme_distribution_charts(theme_distribution):
    """Render pie charts showing company distribution for each theme."""
    if not theme_distribution:
        st.info("No theme distribution data available.")
        return
    
    st.markdown("### Theme Distribution by Company")
    st.markdown("Pie charts showing how many ads each company made for each employer branding theme.")
    
    all_themes = sorted(theme_distribution.keys())
    if not all_themes:
        st.info("No themes available.")
        return
    
    tabs = st.tabs(all_themes)
    
    for idx, theme in enumerate(all_themes):
        with tabs[idx]:
            company_counts = theme_distribution[theme]
            
            if not company_counts:
                st.info(f"No data available for theme: {theme}")
                continue
            
            chart_df = pd.DataFrame({
                'Company': list(company_counts.keys()),
                'Ads': list(company_counts.values())
            }).sort_values('Ads', ascending=False)
            
            color_map = {
                company: BRAND_COLORS.get(company, '#BDBDBD')
                for company in chart_df['Company']
            }
            
            fig = px.pie(
                chart_df,
                values='Ads',
                names='Company',
                title=f'Distribution of "{theme}" Ads by Company',
                color='Company',
                color_discrete_map=color_map
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Ads: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig.update_layout(
                showlegend=True,
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)


def _render_company_theme_stacked_bar(counts_df):
    """Render employer branding ads charts (total + 100% stacked)."""
    if counts_df.empty:
        st.info("No data available for stacked bar chart.")
        return
    
    st.markdown("### Employer Branding Ads by Company and Theme")
    st.markdown("Compare total ad volume and theme mix for each company.")
    
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
        st.markdown("#### Total Ads per Company")
        fig_total = px.bar(
            x=totals.index,
            y=totals.values,
            labels={'x': 'Company', 'y': 'Ads'},
            color_discrete_sequence=['#2FB375']
        )
        fig_total.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=50, b=100),
            xaxis=dict(tickangle=-45)
        )
        fig_total.update_traces(hovertemplate='<b>%{x}</b><br>Ads: %{y}<extra></extra>')
        st.plotly_chart(fig_total, use_container_width=True)
    
    with tabs[1]:
        st.markdown("#### Theme Mix (Share of Ads)")
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
                hovertemplate='<b>%{x}</b><br>Theme: %{fullData.name}<br>Share: %{y:.1f}%<br>Ads: %{customdata}<extra></extra>'
            ))
        fig_mix.update_layout(
            barmode='stack',
            title='Theme Mix per Company (100% stacked)',
            xaxis_title='Company',
            yaxis_title='Share of Ads (%)',
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
    themes_data, theme_distribution, counts_df = _load_all_employer_branding_data()
    
    # First render the stacked bar chart
    _render_company_theme_stacked_bar(counts_df)
    
    st.markdown("---")
    
    # Then render the pie charts
    _render_theme_distribution_charts(theme_distribution)
    
    st.markdown("---")
    
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
            
            # Display each theme with metrics and examples stacked vertically
            for theme_item in themes:
                theme_name = html.escape(str(theme_item['theme']))
                percentage = theme_item['percentage']
                count = theme_item['count']
                examples = theme_item.get('examples', [])
                
                example_cards = []
                if examples:
                    for example in examples:
                        # Handle both dict format (new) and string format (old, for backward compatibility)
                        if isinstance(example, dict):
                            example_text = example.get('text', '')
                            ad_id = example.get('ad_id')
                        else:
                            example_text = str(example)
                            ad_id = None
                        
                        escaped_text = html.escape(example_text)
                        
                        if ad_id:
                            escaped_ad_id = html.escape(str(ad_id))
                            ad_url = f"https://www.facebook.com/ads/library/?id={escaped_ad_id}"
                            example_html = f'<a href="{ad_url}" target="_blank" style="color:#2FB375; text-decoration:none;">"{escaped_text}"</a>'
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
                    f'            <h5 style="margin:0; color:#333; font-weight:600;">{theme_name}</h5>\n'
                    f'        </div>\n'
                    f'        <div style="text-align:right;">\n'
                    f'            <p style="margin:0; font-size:1.8em; color:#2FB375; font-weight:700;">{percentage:.1f}%</p>\n'
                    f'            <p style="margin:4px 0 0; color:#666; font-size:0.9em;">{count} ads</p>\n'
                    f'        </div>\n'
                    f'    </div>\n'
                    f'    <div style="margin-top:12px;">\n'
                    f'{example_cards_html}\n'
                    f'    </div>\n'
                    f'</div>'
                )
                
                st.markdown(card_html, unsafe_allow_html=True)

