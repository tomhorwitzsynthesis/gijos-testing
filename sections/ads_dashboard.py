# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils.date_utils import get_selected_date_range
from utils.file_io import load_ads_data, load_creativity_rankings, load_brand_summaries
import ast
import os
import glob
import re
import unicodedata
from utils.file_io import load_agility_data
from utils.config import BRANDS, DATA_ROOT, BRAND_COLORS

BRAND_ORDER = list(BRAND_COLORS.keys())
DEFAULT_COLOR = "#BDBDBD"  # used for any brand not in BRAND_COLORS

ADS_COMPOS_DIR = os.path.join(DATA_ROOT, "ads", "compos")
COMPOS_SUMMARY_PATH = os.path.join(DATA_ROOT, "ads", "compos", "compos_summary.xlsx")

ARCHETYPE_DISPLAY_ORDER = [
    "The Technologist",
    "The Optimiser",
    "The Globe-trotter",
    "The Accelerator",
    "The Value Seeker",
    "The Expert",
    "The Guardian",
    "The Futurist",
    "The Simplifier",
    "The Personaliser",
    "The Principled",
    "The Collaborator",
    "The Mentor",
    "The Nurturer",
    "The People's Champion",
    "The Eco Warrior",
]

_ARCHETYPE_CANONICAL_MAP = {
    "technologist": "The Technologist",
    "optimizer": "The Optimiser",
    "optimiser": "The Optimiser",
    "jet setter": "The Globe-trotter",
    "jetsetter": "The Globe-trotter",
    "globe trotter": "The Globe-trotter",
    "accelerator": "The Accelerator",
    "value seeker": "The Value Seeker",
    "valueseeker": "The Value Seeker",
    "value s": "The Value Seeker",
    "expert": "The Expert",
    "guardian": "The Guardian",
    "futurist": "The Futurist",
    "simplifier": "The Simplifier",
    "personalizer": "The Personaliser",
    "personaliser": "The Personaliser",
    "principled": "The Principled",
    "collaborator": "The Collaborator",
    "mentor": "The Mentor",
    "nurturer": "The Nurturer",
    "peoples champion": "The People's Champion",
    "people champion": "The People's Champion",
    "people s champion": "The People's Champion",
    "eco warrior": "The Eco Warrior",
    "ecowarrior": "The Eco Warrior",
}


def _render_summary_tabs(summary_records):
    if not summary_records:
        return
    brand_to_summary = {}
    brand_to_media = {}
    for record in summary_records:
        brand = str(record.get("brand", "")).strip()
        summary = record.get("summary", "")
        if not brand or not summary:
            continue
        brand_to_summary[brand] = summary
        brand_to_media[brand] = record.get("media_type")
    if not brand_to_summary:
        return
    ordered_brands = [b for b in BRAND_ORDER if b in brand_to_summary]
    extra_brands = sorted([b for b in brand_to_summary.keys() if b not in BRAND_ORDER])
    tab_labels = ordered_brands + extra_brands
    st.markdown("### Executive Summary")
    tabs = st.tabs(tab_labels)
    card_style = (
        "border:1px solid #2FB375; border-left:6px solid #2FB375;"
        "border-radius:10px; padding:15px; margin-top:10px; margin-bottom:10px;"
        "background-color:#F5FFF9; box-shadow:0 2px 4px rgba(0,0,0,0.08);"
    )
    text_style = "margin:0; color:#1F2933; line-height:1.6;"
    meta_style = "margin:0 0 8px 0; color:#2FB375; font-size:0.85em; font-weight:600; text-transform:uppercase;"
    for idx, brand in enumerate(tab_labels):
        with tabs[idx]:
            summary_text = str(brand_to_summary.get(brand, "")).strip()
            if not summary_text:
                st.info("No summary available.")
                continue
            media_type = brand_to_media.get(brand)
            summary_html = summary_text.replace("\n", "<br>")
            media_html = (
                f'<p style="{meta_style}">{media_type}</p>' if media_type else ""
            )
            st.markdown(
                f"""
                <div style="{card_style}">
                    {media_html}
                    <p style="{text_style}">{summary_html}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _parse_platforms(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except Exception:
        return []
    return []


# Added: brand normalization helper to align names across sources
def _normalize_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = name.split("|")[0].strip()
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in base)
    return " ".join(cleaned.split())


def _summary(df: pd.DataFrame, value_col: str, split_mid: datetime):
    curr = df[df['startDateFormatted'] >= split_mid]
    prev = df[df['startDateFormatted'] < split_mid]

    curr_agg = curr.groupby('brand')[value_col].sum()
    prev_agg = prev.groupby('brand')[value_col].sum()

    rank_curr = curr_agg.sort_values(ascending=False).rank(ascending=False, method="min")
    rank_prev = prev_agg.sort_values(ascending=False).rank(ascending=False, method="min")

    result = pd.DataFrame({
        'current': curr_agg,
        'previous': prev_agg,
        'change_pct': ((curr_agg - prev_agg) / prev_agg.replace(0, 1)) * 100,
        'rank_now': rank_curr,
        'rank_prev': rank_prev,
        'rank_change': rank_curr - rank_prev
    }).fillna(0)

    return result


def _format_metric_card(label, val, pct, rank_now, rank_change, debug=False):
    if rank_change > 0:
        arrow = "↓"
        rank_color = "red"
    elif rank_change < 0:
        arrow = "↑"
        rank_color = "green"
    else:
        arrow = "→"
        rank_color = "gray"

    pct_color = "green" if pct > 0 else "red" if pct < 0 else "gray"

    st.markdown(f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
        <h5 style="margin:0;">{label}</h5>
        <h3 style="margin:5px 0;">{val}</h3>
        <p style="margin:0; color:{pct_color};">Δ {pct:.1f}%</p>
        <p style="margin:0; color:{rank_color};">{arrow} Rank {int(rank_now)}</p>
    """, unsafe_allow_html=True)

    if debug:
        st.markdown(f"""
        <small style="color:#666;">
        Debug: pct={pct:.2f}, rank_now={rank_now}, rank_change={rank_change}
        </small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


# --- New helpers pulled from LP dashboard (adapted to current data layout) ---

# New: Load brand strength from data/ads/compos (fallback to Agility if empty)
def _load_brand_strength_from_ads_compos():
    strength = {}
    if os.path.isdir(ADS_COMPOS_DIR):
        for path in glob.glob(os.path.join(ADS_COMPOS_DIR, "*.xlsx")):
            fname = os.path.basename(path)
            if fname.startswith("~$"):
                continue
            brand_display = fname.replace("_compos_analysis.xlsx", "").replace(".xlsx", "").strip()
            try:
                try:
                    df_comp = pd.read_excel(path, sheet_name="Raw Data")
                except Exception:
                    df_comp = pd.read_excel(path)
                if 'Top Archetype' in df_comp.columns and len(df_comp.dropna(subset=['Top Archetype'])) > 0:
                    vc = df_comp['Top Archetype'].dropna().value_counts()
                    pct = float((vc.max() / vc.sum()) * 100) if vc.sum() > 0 else 0.0
                    strength[brand_display] = pct
            except Exception:
                pass
    return strength


# Kept for fallback

def _compute_brand_strength_from_agility():
    """Return dict brand -> % of dominant archetype from Agility data."""
    strength = {}
    for brand in BRANDS:
        df_ag = load_agility_data(brand)
        if df_ag is None or df_ag.empty:
            continue
        if 'Top Archetype' in df_ag.columns and len(df_ag.dropna(subset=['Top Archetype'])) > 0:
            vc = df_ag['Top Archetype'].dropna().value_counts()
            pct = float((vc.max() / vc.sum()) * 100) if vc.sum() > 0 else 0.0
            strength[brand] = pct
    return strength


# New: Load top archetypes from data/ads/compos (fallback to Agility if empty)
def _load_top_archetypes_from_ads_compos():
    results = {}
    if os.path.isdir(ADS_COMPOS_DIR):
        for path in glob.glob(os.path.join(ADS_COMPOS_DIR, "*.xlsx")):
            fname = os.path.basename(path)
            if fname.startswith("~$"):
                continue
            brand_display = fname.replace("_compos_analysis.xlsx", "").replace(".xlsx", "").strip()
            try:
                try:
                    df_comp = pd.read_excel(path, sheet_name="Raw Data")
                except Exception:
                    df_comp = pd.read_excel(path)
                if 'Top Archetype' in df_comp.columns:
                    vc = df_comp['Top Archetype'].dropna().value_counts()
                    total = int(vc.sum()) if vc.sum() else 0
                    top3 = vc.head(3)
                    items = []
                    for archetype, count in top3.items():
                        pct = (count / total) * 100 if total > 0 else 0
                        items.append({'archetype': archetype, 'percentage': pct, 'count': int(count)})
                    if items:
                        results[brand_display] = items
            except Exception:
                pass
    return results


def _load_top_archetypes_from_agility():
    """Return dict brand -> list of top 3 archetypes with percentage and count."""
    results = {}
    for brand in BRANDS:
        df_ag = load_agility_data(brand)
        if df_ag is None or df_ag.empty or 'Top Archetype' not in df_ag.columns:
            continue
        vc = df_ag['Top Archetype'].dropna().value_counts()
        total = int(vc.sum()) if vc.sum() else 0
        top3 = vc.head(3)
        items = []
        for archetype, count in top3.items():
            pct = (count / total) * 100 if total > 0 else 0
            items.append({
                'archetype': archetype,
                'percentage': pct,
                'count': int(count)
            })
        if items:
            results[brand] = items
    return results


@st.cache_data(ttl=0)
def _load_key_advantages():
    path = os.path.join(DATA_ROOT, "key_advantages", "key_advantages.xlsx")
    if not os.path.exists(path):
        return {}
    advantages = {}
    try:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            if sheet_name.strip().lower() == "summary":
                continue
            brand_name = sheet_name.replace("_", " ")
            df_adv = pd.read_excel(path, sheet_name=sheet_name)
            df_adv.columns = [str(col).lower().strip() for col in df_adv.columns]
            required_cols = ['advantage_id', 'title', 'evidence_list', 'example_ad_index', 'example_quote']
            for col in required_cols:
                if col not in df_adv.columns:
                    df_adv[col] = ""
            advantages[brand_name] = df_adv[required_cols]
    except Exception as e:
        st.error(f"Error loading key advantages: {e}")
    return advantages


def _load_key_advantages_summary():
    path = os.path.join(DATA_ROOT, "key_advantages", "key_advantages_summary.xlsx")
    if not os.path.exists(path):
        return ""
    try:
        df_summary = pd.read_excel(path)
        summary_col = None
        for col in df_summary.columns:
            if 'summary' in str(col).lower():
                summary_col = col
                break
        if summary_col and len(df_summary) > 0:
            return str(df_summary[summary_col].iloc[0])
        else:
            return ""
    except Exception as e:
        st.error(f"Error loading key advantages summary: {e}")
        return ""


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

def _present_color_map(present_brands):
    m = dict(BRAND_COLORS)
    for b in present_brands:
        if b not in m:
            m[b] = DEFAULT_COLOR
    return m


def _canonicalize_archetype(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    text = unicodedata.normalize("NFKD", name)
    text = text.replace("�", "'").replace("’", "'")
    text = text.lower().strip()
    if text.startswith("the "):
        text = text[4:]
    text = text.replace("-", " ")
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z ]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return None
    mapped = _ARCHETYPE_CANONICAL_MAP.get(text)
    if mapped:
        return mapped
    display = " ".join(word.capitalize() for word in text.split())
    return f"The {display}" if display else None


def _green_gradient(pct: float) -> tuple[str, str]:
    """Return background and text colors based on percentage intensity."""
    try:
        value = float(pct)
    except (TypeError, ValueError):
        value = 0.0
    value = max(0.0, min(100.0, value))
    base_rgb = (245, 255, 249)  # light mint
    peak_rgb = (31, 179, 117)   # strong green
    ratio = value / 100.0
    r = round(base_rgb[0] + (peak_rgb[0] - base_rgb[0]) * ratio)
    g = round(base_rgb[1] + (peak_rgb[1] - base_rgb[1]) * ratio)
    b = round(base_rgb[2] + (peak_rgb[2] - base_rgb[2]) * ratio)
    bg_hex = f"#{r:02X}{g:02X}{b:02X}"
    text_color = "#1F2933" if value < 60 else "#FFFFFF"
    return bg_hex, text_color


def render():

    df = load_ads_data()
    _render_summary_tabs(load_brand_summaries("ads"))
    if df is None or df.empty:
        st.info("No ads data available.")
        return

    if "startDateFormatted" in df.columns:
        df["startDateFormatted"] = pd.to_datetime(df["startDateFormatted"], errors="coerce")
        df = df.dropna(subset=["startDateFormatted"])

    # Use global selected date range for pie chart and cards
    start_date, end_date = get_selected_date_range()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Clip to selected range for pie chart and cards
    if "startDateFormatted" in df.columns:
        df_filtered = df[(df['startDateFormatted'] >= start_ts) & (df['startDateFormatted'] < end_ts)].copy()
    else:
        df_filtered = df.copy()

    # Build a rolling six-month window aligned to the most recent data we have within the selection (fallback to full dataset)
    if "startDateFormatted" in df.columns and not df.empty:
        if not df_filtered.empty:
            window_end = df_filtered["startDateFormatted"].max()
        else:
            window_end = df["startDateFormatted"].max()

        if pd.isna(window_end):
            df_fixed = df.copy()
        else:
            data_min = df["startDateFormatted"].min()
            window_start = window_end - pd.DateOffset(months=6)
            if pd.notna(data_min):
                window_start = max(window_start, data_min)
            if window_start > window_end:
                window_start = data_min
            df_fixed = df[(df['startDateFormatted'] >= window_start) & (df['startDateFormatted'] <= window_end)].copy()
    else:
        df_fixed = df.copy()

    # Pie charts and cards for selected months only
    st.markdown("### Ad Volume Share (Selected Months)")
    sub_tabs = st.tabs(["Number of Ads", "Reach"])
    with sub_tabs[0]:
        ad_counts = df_filtered["brand"].value_counts().reset_index()
        if not ad_counts.empty:
            ad_counts.columns = ["brand", "count"]

            # Build a color map that covers all present brands (unknowns -> gray)
            color_map_ads = {**BRAND_COLORS}
            for b in ad_counts["brand"].unique():
                color_map_ads.setdefault(b, DEFAULT_COLOR)

            fig = px.pie(
                ad_counts,
                values="count",
                names="brand",
                title=f'Ad Count Share – {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")}',
                color="brand",
                color_discrete_map=color_map_ads,
                category_orders={"brand": BRAND_ORDER},
            )
            st.plotly_chart(fig, use_container_width=True, key="pie_ads_selected")
        else:
            st.info("No ads in selected months.")

    with sub_tabs[1]:
        reach_totals = df_filtered.groupby("brand", as_index=False)["reach"].sum()
        if not reach_totals.empty:

            color_map_reach = {**BRAND_COLORS}
            for b in reach_totals["brand"].unique():
                color_map_reach.setdefault(b, DEFAULT_COLOR)

            fig = px.pie(
                reach_totals,
                values="reach",
                names="brand",
                title=f'Reach Share – {start_date.strftime("%b %Y")} to {end_date.strftime("%b %Y")}',
                color="brand",
                color_discrete_map=color_map_reach,
                category_orders={"brand": BRAND_ORDER},
            )
            st.plotly_chart(fig, use_container_width=True, key="pie_reach_selected")
        else:
            st.info("No reach data in selected months.")

    # Summary stats (midpoint split) for potential future use; old 4 cards removed
    mid_date = start_date + (end_date - start_date) / 2
    reach_stats = _summary(df_filtered, 'reach', mid_date)
    df_filtered['ad_count'] = 1
    ads_stats = _summary(df_filtered, 'ad_count', mid_date)
    duration_stats = _summary(df_filtered, 'duration_days', mid_date) if 'duration_days' in df_filtered.columns else None

    # --- Brand Summary (cards + creativity analysis) ---
    st.markdown("### Brand Summary")

    # Compute 6-month reach totals and ranks using the fixed window
    reach_6m = df_fixed.groupby('brand')['reach'].sum() if not df_fixed.empty else pd.Series(dtype=float)
    reach_mean = reach_6m.mean() if len(reach_6m) else 0
    reach_ranks = reach_6m.rank(ascending=False, method="min") if len(reach_6m) else pd.Series(dtype=float)
    reach_6m_map = reach_6m.to_dict()
    reach_norm_map = {
        _normalize_brand(brand): value
        for brand, value in reach_6m_map.items()
        if _normalize_brand(brand)
    }
    reach_rank_map = reach_ranks.to_dict()
    reach_rank_norm_map = {
        _normalize_brand(brand): rank
        for brand, rank in reach_rank_map.items()
        if _normalize_brand(brand)
    }

    # Brand strength from compos files in data/ads/compos, fallback to Agility
    strength_map = _load_brand_strength_from_summary()
    if not strength_map:
        strength_map = _compute_brand_strength_from_agility()
    if strength_map:
        bs_df = pd.DataFrame({'brand': list(strength_map.keys()), 'strength': list(strength_map.values())})
        bs_df['brand_norm'] = bs_df['brand'].apply(_normalize_brand)
        bs_df['rank'] = bs_df['strength'].rank(ascending=False, method='min')
        bs_mean = bs_df['strength'].mean() if len(bs_df) else 0
        bs_df['delta_vs_mean_pct'] = ((bs_df['strength'] - bs_mean) / (bs_mean if bs_mean != 0 else 1)) * 100
    else:
        bs_df = pd.DataFrame(columns=['brand', 'brand_norm', 'strength', 'rank', 'delta_vs_mean_pct'])
        bs_mean = 0

    creativity_df = load_creativity_rankings("ads")
    if not creativity_df.empty:
        creativity_df['rank'] = pd.to_numeric(creativity_df['rank'], errors='coerce')
        creativity_df['originality_score'] = pd.to_numeric(creativity_df['originality_score'], errors='coerce')
        cre_mean = creativity_df['originality_score'].mean()
        denom = cre_mean if cre_mean != 0 else 1
        creativity_df['delta_vs_mean_pct'] = ((creativity_df['originality_score'] - denom) / denom) * 100

    # Build brand tabs from union of brands across sources (ads, compos, creativity)
    ads_brands = set(df_fixed['brand'].dropna().unique())
    compos_brands = set(bs_df['brand'].unique()) if len(bs_df) else set()
    creativity_brands = set(creativity_df['brand'].dropna().unique()) if not creativity_df.empty else set()

    # Normalize for matching, but display original preferred names: prefer ads brand if available, else compos/creativity name
    norm_to_display = {}
    ads_norm_map = { _normalize_brand(b): b for b in ads_brands }
    for b in compos_brands.union(ads_brands).union(creativity_brands):
        norm = _normalize_brand(b)
        if not norm:
            continue
        if norm in norm_to_display:
            continue
        preferred = ads_norm_map.get(norm, b)
        norm_to_display[norm] = preferred

    available_brands = sorted(list(norm_to_display.values()))

    if not available_brands:
        st.info("No brands available to display.")
    else:
        brand_tabs = st.tabs(available_brands)
        for i, brand_name in enumerate(available_brands):
            with brand_tabs[i]:
                col1, col2, col3 = st.columns(3)

                # Reach 6 months
                with col1:
                    norm_brand = _normalize_brand(brand_name)
                    total_reach = 0
                    if len(reach_6m):
                        total_reach = int(
                            reach_6m_map.get(brand_name, reach_norm_map.get(norm_brand, 0))
                        )
                    delta_mean_pct = ((total_reach - (reach_mean if reach_mean != 0 else 1)) / (reach_mean if reach_mean != 0 else 1)) * 100 if reach_mean != 0 else 0
                    rank_now = None
                    if len(reach_ranks):
                        rank_now = reach_rank_map.get(brand_name, reach_rank_norm_map.get(norm_brand))
                    _format_simple_metric_card(
                        label="Reach",
                        val=f"{total_reach:,}",
                        pct=delta_mean_pct,
                        rank_now=rank_now,
                        total_ranks=len(reach_ranks) if len(reach_ranks) else None
                    )

                # Brand Strength
                with col2:
                    row = bs_df[bs_df['brand_norm'] == _normalize_brand(brand_name)]
                    if not row.empty:
                        strength = float(row['strength'].iloc[0])
                        rank_bs = int(row['rank'].iloc[0])
                        delta_bs = float(row['delta_vs_mean_pct'].iloc[0])
                        _format_simple_metric_card(
                            label="Brand Strength",
                            val=f"{strength:.1f}%",
                            pct=delta_bs,
                            rank_now=rank_bs,
                            total_ranks=len(bs_df)
                        )
                    else:
                        _format_simple_metric_card("Brand Strength", "N/A")

                # Creativity
                with col3:
                    cre_row = creativity_df[creativity_df['brand'].astype(str).str.lower() == brand_name.lower()] if not creativity_df.empty else pd.DataFrame()
                    if not cre_row.empty:
                        score = cre_row['originality_score'].iloc[0]
                        rank_cre = int(cre_row['rank'].iloc[0]) if pd.notna(cre_row['rank'].iloc[0]) else None
                        delta_cre = float(cre_row['delta_vs_mean_pct'].iloc[0]) if pd.notna(cre_row['delta_vs_mean_pct'].iloc[0]) else None
                        _format_simple_metric_card(
                            label="Creativity",
                            val=f"{score:.2f}",
                            pct=delta_cre,
                            rank_now=rank_cre,
                            total_ranks=creativity_df['brand'].nunique() if not creativity_df.empty else None
                        )
                    else:
                        _format_simple_metric_card("Creativity", "N/A")

                # Creativity Analysis section
                if not creativity_df.empty:
                    cre_row = creativity_df[creativity_df['brand'].astype(str).str.lower() == brand_name.lower()]
                    if not cre_row.empty:
                        score = cre_row['originality_score'].iloc[0]
                        rank_cre = int(cre_row['rank'].iloc[0]) if pd.notna(cre_row['rank'].iloc[0]) else None
                        just_text = str(cre_row['justification'].iloc[0]) if pd.notna(cre_row['justification'].iloc[0]) else ""
                        examples_text = str(cre_row['examples'].iloc[0]) if pd.notna(cre_row['examples'].iloc[0]) else ""
                        if just_text or examples_text:
                            st.markdown("#### Creativity Analysis")
                            st.markdown(f"""
                            <div style=\"border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;\">
                                <h5 style=\"margin:0;\">{brand_name} — {f'Rank {rank_cre} — ' if rank_cre is not None else ''}Score {score:.2f}</h5>
                                {f'<p style=\"margin:8px 0 0; color:#444;\">{just_text}</p>' if just_text else ''}
                                {f'<p style=\"margin:8px 0 0; color:#444;\">Examples: {examples_text}</p>' if examples_text else ''}
                            </div>
                            """, unsafe_allow_html=True)

    # --- Top Archetypes by Company ---
    st.markdown("### Top Archetypes by Company")
    archetype_stats = _load_archetype_stats_from_summary()
    if not archetype_stats:
        archetype_stats = _load_archetype_stats_from_agility()

    overall_counts = {}
    overall_total = 0
    if archetype_stats:
        for info in archetype_stats.values():
            counts = info.get("counts", {})
            for archetype, count in counts.items():
                count_int = int(count)
                overall_counts[archetype] = overall_counts.get(archetype, 0) + count_int
                overall_total += count_int

    archetypes_data = {}
    if archetype_stats:
        for company, info in archetype_stats.items():
            counts = info.get("counts", {})
            total = info.get("total", 0)
            if not counts or total <= 0:
                continue
            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            top_items = []
            for archetype, count in sorted_items[:3]:
                pct = (count / total) * 100 if total else 0
                top_items.append({
                    "archetype": archetype,
                    "percentage": pct,
                    "count": int(count),
                })
            if top_items:
                archetypes_data[company] = top_items

    if not archetype_stats:
        st.info("No archetype data available. Ensure compos files are in data/ads/compos or Agility files contain 'Top Archetype'.")
    else:
        if archetypes_data:
            overall_items = []
            if overall_counts:
                overall_top3 = sorted(overall_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for archetype, count in overall_top3:
                    pct = (count / overall_total) * 100 if overall_total else 0
                    overall_items.append({
                        "archetype": archetype,
                        "percentage": pct,
                        "count": int(count),
                    })

            tab_labels = ["🌍 Overall"] + list(archetypes_data.keys())
            company_tabs = st.tabs(tab_labels)
            # Overall tab
            with company_tabs[0]:
                st.subheader("Overall - Top 3 Archetypes")
                col1, col2, col3 = st.columns(3)
                for j, archetype_info in enumerate(overall_items):
                    col = col1 if j == 0 else col2 if j == 1 else col3
                    with col:
                        st.markdown(f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:10px; text-align:center;">
                            <h4 style="margin:0; color:#333;">{archetype_info['archetype']}</h4>
                            <h2 style="margin:5px 0; color:#333; font-size:2.0em;">{archetype_info['percentage']:.1f}%</h2>
                            <p style="margin:0; color:#666; font-size:0.9em;">{archetype_info['count']} items</p>
                        </div>
                        """, unsafe_allow_html=True)
            # Company tabs
            for i, (company, archetypes) in enumerate(archetypes_data.items()):
                with company_tabs[i + 1]:
                    st.subheader(f"{company} - Top 3 Archetypes")
                    col1, col2, col3 = st.columns(3)
                    for j, archetype_info in enumerate(archetypes):
                        col = col1 if j == 0 else col2 if j == 1 else col3
                        with col:
                            st.markdown(f"""
                            <div style="border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:10px; text-align:center;">
                                <h4 style="margin:0; color:#333;">{archetype_info['archetype']}</h4>
                                <h2 style="margin:5px 0; color:#333; font-size:2.0em;">{archetype_info['percentage']:.1f}%</h2>
                                <p style="margin:0; color:#666; font-size:0.9em;">{archetype_info['count']} items</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No archetype data available. Ensure compos files are in data/ads/compos or Agility files contain 'Top Archetype'.")

        # --- Archetype Matrix View ---
        st.markdown("### Archetype Coverage Matrix")
        def _render_archetype_matrix(counts_dict, total_count):
            counts_dict = counts_dict or {}
            total = total_count if total_count and total_count > 0 else 0
            card_base_style = (
                "border:1px solid #CDE7D8; border-radius:10px; padding:12px; "
                "margin-bottom:12px; text-align:center;"
            )
            for row_start in range(0, len(ARCHETYPE_DISPLAY_ORDER), 4):
                cols = st.columns(4)
                for idx, archetype in enumerate(ARCHETYPE_DISPLAY_ORDER[row_start:row_start + 4]):
                    pct_value = 0.0
                    count_value = int(counts_dict.get(archetype, 0))
                    if total > 0:
                        pct_value = (count_value / total) * 100
                    bg_hex, text_color = _green_gradient(pct_value)
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div style="{card_base_style} background-color:{bg_hex}; color:{text_color};">
                                <h5 style="margin:0;">{archetype}</h5>
                                <p style="margin:6px 0 0; font-size:1.1em; font-weight:600;">{pct_value:.1f}%</p>
                                <p style="margin:0; font-size:0.85em; color:{text_color}; opacity:0.85;">{count_value} ads</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        matrix_labels = ["🌍 Overall"] + list(archetype_stats.keys())
        matrix_tabs = st.tabs(matrix_labels)

        with matrix_tabs[0]:
            st.subheader("Overall Archetype Distribution")
            _render_archetype_matrix(overall_counts, overall_total)

        for idx, (company, info) in enumerate(archetype_stats.items()):
            with matrix_tabs[idx + 1]:
                st.subheader(f"{company} Archetype Distribution")
                _render_archetype_matrix(info.get("counts", {}), info.get("total", 0))


    st.markdown(
    'Read more about brand archetypes here: [Brandtypes](https://www.comp-os.com/brandtypes)')
        

    # --- Key Advantages ---
    st.markdown("### Key Advantages")
    key_advantages_data = _load_key_advantages()
    if not key_advantages_data:
        st.info("No key advantages data loaded. Place key_advantages.xlsx in data/key_advantages.")
    else:
        brand_tabs = st.tabs(list(key_advantages_data.keys()))
        for i, brand_disp in enumerate(key_advantages_data.keys()):
            with brand_tabs[i]:
                st.subheader(f"{brand_disp} Key Advantages")
                brand_data = key_advantages_data.get(brand_disp)
                if brand_data is None or brand_data.empty:
                    st.info(f"No specific key advantages found for {brand_disp}.")
                else:
                    grouped_data = brand_data.groupby(['title', 'evidence_list']).agg({
                        'example_quote': lambda x: list(x)
                    }).reset_index()
                    col1, col2 = st.columns(2)
                    for idx, (_, row) in enumerate(grouped_data.iterrows()):
                        with (col1 if idx % 2 == 0 else col2):
                            examples_html = ""
                            for ii, example in enumerate(row['example_quote']):
                                if example and str(example).strip():
                                    examples_html += f"<li style='margin:4px 0; color:#444;'>{example}</li>"
                            examples_html = f"<ul style='margin:4px 0 0; padding-left:20px;'>{examples_html}</ul>" if examples_html else "<p style='margin:4px 0 0; color:#444;'>No examples available</p>"
                            st.markdown(f"""
                            <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px; max-height:800px; overflow-y:auto;">
                                <h5 style="margin:0; word-wrap:break-word;">{row['title']}</h5>
                                <p style="margin:8px 0 0; font-weight:bold; color:#333;">Evidence:</p>
                                <p style="margin:4px 0 8px; color:#444; word-wrap:break-word; white-space:pre-wrap;">{row['evidence_list']}</p>
                                <p style="margin:8px 0 0; font-weight:bold; color:#333;">Examples:</p>
                                {examples_html}
                            </div>
                            """, unsafe_allow_html=True)

    # In-Depth View (rolling six-month window)
    st.markdown("### In-Depth View")

    st.markdown("Ad Start Date Distribution")
    hist = px.histogram(df_fixed, x='startDateFormatted', color='brand', nbins=60, barmode='overlay')
    _present = df_fixed["brand"].unique()
    hist = px.histogram(
        df_fixed,
        x="startDateFormatted",
        color="brand",
        nbins=60,
        barmode="overlay",
        color_discrete_map=_present_color_map(_present),
        category_orders={"brand": BRAND_ORDER},
        labels={"startDateFormatted": "Month"}
    )
    st.plotly_chart(hist, use_container_width=True)

    # Text analysis and new campaigns sections can be added later as optional blocks

def _load_archetype_stats_from_summary():
    if not os.path.exists(COMPOS_SUMMARY_PATH):
        return {}
    df = pd.read_excel(COMPOS_SUMMARY_PATH)
    stats = {}
    for col in df.columns:
        values = df[col].dropna()
        counts = values.value_counts()
        canonical_counts = {}
        for archetype, count in counts.items():
            canonical = _canonicalize_archetype(archetype)
            if not canonical:
                continue
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + int(count)
        total = int(sum(canonical_counts.values()))
        if total == 0:
            continue
        stats[col] = {
            "total": total,
            "counts": canonical_counts,
        }
    return stats


def _load_archetype_stats_from_agility():
    stats = {}
    for brand in BRANDS:
        df_ag = load_agility_data(brand)
        if df_ag is None or df_ag.empty or 'Top Archetype' not in df_ag.columns:
            continue
        values = df_ag['Top Archetype'].dropna()
        counts = values.value_counts()
        canonical_counts = {}
        for archetype, count in counts.items():
            canonical = _canonicalize_archetype(archetype)
            if not canonical:
                continue
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + int(count)
        total = int(sum(canonical_counts.values()))
        if total == 0:
            continue
        stats[brand] = {
            "total": total,
            "counts": canonical_counts,
        }
    return stats

def _load_brand_strength_from_summary():
    if not os.path.exists(COMPOS_SUMMARY_PATH):
        return {}
    df = pd.read_excel(COMPOS_SUMMARY_PATH)
    strength = {}
    for col in df.columns:
        vc = df[col].dropna().value_counts()
        total = int(vc.sum()) if vc.sum() else 0
        if total > 0:
            dominant = vc.iloc[0]
            strength[col] = (dominant / total) * 100
    return strength



