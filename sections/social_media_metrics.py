from typing import Dict, Tuple

import os
import glob
import streamlit as st
import pandas as pd

from utils.config import BRAND_NAME_MAPPING, DATA_ROOT, BRANDS
from utils.date_utils import get_selected_date_range
from utils.file_io import load_social_data, load_creativity_rankings, load_brand_summaries


SOCIAL_COMPOS_DIR = os.path.join(DATA_ROOT, "social_media", "compos")


def _normalize_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = name.split("|")[0].strip()
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in base)
    return " ".join(cleaned.split())


def _format_simple_metric_card(label, val, pct=None, rank_now=None, total_ranks=None, metric_explanation=None):
    rank_color = "gray"
    if rank_now is not None and total_ranks:
        if int(rank_now) == 1:
            rank_color = "green"
        elif int(rank_now) == int(total_ranks):
            rank_color = "red"
    pct_color = None
    if pct is not None:
        pct_color = "green" if pct > 0 else "red" if pct < 0 else "gray"
    # Add tooltip to percentage change
    pct_tooltip = '<span class="pct-tooltip-icon" data-tooltip="Difference from the average metric">?</span>' if pct is not None else ''
    pct_html = f'<p style="margin:0; color:{pct_color}; display:inline-flex; align-items:center; gap:4px;">Δ {pct:.1f}%{pct_tooltip}</p>' if pct is not None else ''
    rank_html = f'<p style="margin:0; color:{rank_color};">Rank {int(rank_now)}</p>' if rank_now is not None else ''
    # Add tooltip icon next to metric label if explanation provided
    label_with_tooltip = label
    if metric_explanation:
        # Escape HTML in explanation for data attribute
        import html
        escaped_explanation = html.escape(metric_explanation)
        label_with_tooltip = f'{label}<span class="metric-tooltip-icon" data-explanation="{escaped_explanation}">?</span>'
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h5 style="margin:0; display:inline-flex; align-items:center; gap:4px;">{label_with_tooltip}</h5>
            <h3 style="margin:5px 0;">{val}</h3>
            {pct_html}
            {rank_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _rolling_window(df: pd.DataFrame, date_col: str, months: int = 6) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    window_end = df[date_col].max()
    if pd.isna(window_end):
        return df
    window_start = window_end - pd.DateOffset(months=months)
    return df[(df[date_col] >= window_start) & (df[date_col] <= window_end)].copy()


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype=float)


@st.cache_data(ttl=0)
def _load_social_frames(platform: str = "linkedin"):
    frames = {}
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
        if df is None or df.empty:
            continue
        if "Published Date" not in df.columns:
            continue
        df = df.copy()
        df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
        df = df.dropna(subset=["Published Date"])
        if df.empty:
            continue
        likes = _numeric_series(df, "num_likes")
        comments = _numeric_series(df, "num_comments")
        shares = _numeric_series(df, "num_shares")
        df["Engagement"] = likes + comments * 3 + shares
        frames[brand_display] = df
    return frames


@st.cache_data(ttl=0)
def _load_social_brand_strength():
    strength = {}
    if not os.path.isdir(SOCIAL_COMPOS_DIR):
        return strength
    for path in glob.glob(os.path.join(SOCIAL_COMPOS_DIR, "*.xlsx")):
        fname = os.path.basename(path)
        if fname.startswith("~$"):
            continue
        brand_candidate = fname.replace("_compos_analysis.xlsx", "").replace(".xlsx", "").strip()
        # Normalize brand name from file name to canonical name
        normalized_brand = BRAND_NAME_MAPPING.get(brand_candidate, brand_candidate)
        # Only process if it's one of our current brands
        if normalized_brand not in BRANDS:
            continue
        try:
            try:
                df_comp = pd.read_excel(path, sheet_name="Raw Data")
            except Exception:
                df_comp = pd.read_excel(path)
        except Exception:
            continue
        if "Top Archetype" not in df_comp.columns:
            continue
        counts = df_comp["Top Archetype"].dropna().value_counts()
        total = counts.sum()
        if total > 0:
            strength[normalized_brand] = float(counts.iloc[0] / total * 100)
    return strength


def _compute_engagement_stats(frames: dict) -> Tuple[Dict[str, dict], int]:
    if not frames:
        return {}, 0
    engagement_map = {}
    for brand, df in frames.items():
        windowed = _rolling_window(df, "Published Date")
        if windowed.empty:
            continue
        total_engagement = windowed["Engagement"].sum()
        engagement_map[brand] = float(total_engagement)
    if not engagement_map:
        return {}, 0
    series = pd.Series(engagement_map)
    ranks = series.rank(ascending=False, method="min")
    mean_val = series.mean()
    denom = mean_val if mean_val != 0 else 1
    stats = {}
    for brand, value in series.items():
        norm = _normalize_brand(brand)
        stats[norm] = {
            "value": float(value),
            "delta": ((value - mean_val) / denom) * 100 if mean_val else 0,
            "rank": int(ranks[brand]),
        }
    return stats, len(series)


def _compute_brand_strength_stats() -> Tuple[Dict[str, dict], int]:
    strength_map = _load_social_brand_strength()
    if not strength_map:
        return {}, 0
    series = pd.Series(strength_map)
    ranks = series.rank(ascending=False, method="min")
    mean_val = series.mean()
    denom = mean_val if mean_val != 0 else 1
    stats = {}
    for brand, value in series.items():
        norm = _normalize_brand(brand)
        stats[norm] = {
            "value": float(value),
            "delta": ((value - mean_val) / denom) * 100 if mean_val else 0,
            "rank": int(ranks[brand]),
        }
    return stats, len(series)


def _compute_creativity_stats() -> Tuple[Dict[str, dict], int, Dict[str, dict]]:
    df = load_creativity_rankings("social_media")
    if df is None or df.empty:
        return {}, 0, {}
    df = df.copy()
    # Normalize brand names from creativity file using BRAND_NAME_MAPPING
    df["brand_normalized"] = df["brand"].apply(lambda b: BRAND_NAME_MAPPING.get(str(b).strip(), str(b).strip()))
    # Only keep brands that are in our BRANDS list
    df = df[df["brand_normalized"].isin(BRANDS)].copy()
    if df.empty:
        return {}, 0, {}
    df["brand_norm"] = df["brand_normalized"].apply(_normalize_brand)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["originality_score"] = pd.to_numeric(df["originality_score"], errors="coerce")
    df = df.dropna(subset=["originality_score"])
    if df.empty:
        return {}, 0, {}
    mean_val = df["originality_score"].mean()
    denom = mean_val if mean_val != 0 else 1
    df["delta_vs_mean_pct"] = ((df["originality_score"] - mean_val) / denom) * 100
    stats = {}
    details = {}
    for _, row in df.iterrows():
        details[row["brand_norm"]] = {
            "brand": row["brand_normalized"],  # Use normalized brand name
            "rank": int(row["rank"]) if not pd.isna(row["rank"]) else None,
            "score": float(row["originality_score"]),
            "justification": row.get("justification", ""),
            "examples": row.get("examples", ""),
        }
        stats[row["brand_norm"]] = {
            "score": float(row["originality_score"]),
            "delta": float(row["delta_vs_mean_pct"]) if not pd.isna(row["delta_vs_mean_pct"]) else None,
            "rank": int(row["rank"]) if not pd.isna(row["rank"]) else None,
        }
    return stats, df["brand_normalized"].nunique(), details


def _render_creativity_block(display_name: str, detail: Dict[str, object]) -> None:
    if not detail:
        return
    score = detail.get("score")
    rank = detail.get("rank")
    justification = str(detail.get("justification", "") or "").strip()
    examples = str(detail.get("examples", "") or "").strip()
    if not any([justification, examples]):
        return
    headline_parts = []
    if rank is not None:
        headline_parts.append(f"Rank {rank}")
    if score is not None:
        headline_parts.append(f"Score {score:.2f}")
    headline = f"{display_name} — {' — '.join(headline_parts)}" if headline_parts else display_name
    examples_html = ""
    if examples:
        examples_html = f'<p style="margin:8px 0 0; color:#444;">Examples: {examples}</p>'
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-top:15px; margin-bottom:10px;">
            <h5 style="margin:0;">{headline}</h5>
            {'<p style="margin:8px 0 0; color:#444;">' + justification + '</p>' if justification else ''}
            {examples_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _has_creativity_content(detail: Dict[str, object]) -> bool:
    if not detail:
        return False
    justification = str(detail.get("justification", "") or "").strip()
    examples = str(detail.get("examples", "") or "").strip()
    return bool(justification or examples)


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
    # Normalize brand names from summaries using BRAND_NAME_MAPPING
    normalized_brand_to_summary = {}
    normalized_brand_to_media = {}
    for brand, summary in brand_to_summary.items():
        normalized_brand = BRAND_NAME_MAPPING.get(brand, brand)
        # Only include brands that are in our BRANDS list (current brands)
        if normalized_brand in BRANDS:
            # If already normalized, use it; otherwise try to find the canonical name
            if normalized_brand not in normalized_brand_to_summary:
                normalized_brand_to_summary[normalized_brand] = summary
                normalized_brand_to_media[normalized_brand] = brand_to_media.get(brand)
            # If we have duplicates (e.g., "Ignitis" and "Ignitis grupe"), merge them
            elif normalized_brand in normalized_brand_to_summary:
                # Keep the first one or merge summaries if needed
                pass
    
    brand_to_summary = normalized_brand_to_summary
    brand_to_media = normalized_brand_to_media
    
    # If no brands remain after filtering, don't show summary tabs
    if not brand_to_summary:
        return
    
    preferred_order = BRANDS  # Use BRANDS directly instead of BRAND_NAME_MAPPING.values()
    ordered = [b for b in preferred_order if b in brand_to_summary]
    extras = sorted([b for b in brand_to_summary.keys() if b not in ordered])
    tab_labels = ordered + extras
    
    # Ensure we have at least one tab label
    if not tab_labels:
        return
    
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


def render(selected_platforms=None):
    # Add tooltip CSS
    st.markdown("""
    <style>
        .metric-tooltip-icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #2FB375;
            color: white;
            text-align: center;
            line-height: 16px;
            font-size: 12px;
            font-weight: bold;
            cursor: help;
            margin-left: 6px;
            vertical-align: middle;
            position: relative;
        }
        .metric-tooltip-icon:hover::after {
            content: attr(data-explanation);
            white-space: pre-line;
            word-wrap: break-word;
            overflow-wrap: break-word;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 12px;
            margin-bottom: 8px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            max-width: 300px;
            min-width: 200px;
            text-align: left;
            line-height: 1.5;
        }
        .metric-tooltip-icon:hover::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #333;
            margin-bottom: 2px;
            z-index: 1001;
        }
        .pct-tooltip-icon {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background-color: currentColor;
            color: inherit;
            text-align: center;
            line-height: 14px;
            font-size: 10px;
            font-weight: bold;
            cursor: help;
            margin-left: 4px;
            vertical-align: middle;
            position: relative;
            opacity: 0.7;
        }
        .pct-tooltip-icon:hover {
            opacity: 1;
        }
        .pct-tooltip-icon:hover::after {
            content: "Difference from the average metric";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            white-space: nowrap;
            font-size: 11px;
            margin-bottom: 8px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .pct-tooltip-icon:hover::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #333;
            margin-bottom: 2px;
            z-index: 1001;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Social Media Brand Metrics")

    if selected_platforms is None:
        selected_platforms = ["linkedin"]
    selected_platforms = [p.lower() for p in selected_platforms]
    platform = selected_platforms[0] if selected_platforms else "linkedin"

    _render_summary_tabs(load_brand_summaries("social_media"))
    frames = _load_social_frames(platform)
    if not frames:
        st.info("No social media data available.")
        return

    engagement_stats, engagement_total = _compute_engagement_stats(frames)
    strength_stats, strength_total = _compute_brand_strength_stats()
    creativity_stats, creativity_total, creativity_details = _compute_creativity_stats()

    available_brands = []
    seen = set()
    # Use BRANDS to avoid duplicates
    for display in BRANDS:
        norm = _normalize_brand(display)
        if (norm in engagement_stats) or (norm in strength_stats) or (norm in creativity_stats):
            available_brands.append(display)
            seen.add(display)
    extra_keys = set(engagement_stats.keys()) | set(strength_stats.keys()) | set(creativity_stats.keys())
    for norm in sorted(extra_keys):
        # Try to find the brand name from frames or use normalized name
        display = next((name for name in frames.keys() if _normalize_brand(name) == norm), None)
        if display is None:
            # Try to find in BRANDS
            display = next((b for b in BRANDS if _normalize_brand(b) == norm), norm.title())
        if display not in seen:
            available_brands.append(display)
            seen.add(display)

    if not available_brands:
        st.info("No brand metrics available to display.")
        return

    tabs = st.tabs(available_brands)
    for idx, display_name in enumerate(available_brands):
        norm = _normalize_brand(display_name)
        with tabs[idx]:
            col1, col2, col3 = st.columns(3)

            with col1:
                engagement_info = engagement_stats.get(norm)
                if engagement_info:
                    _format_simple_metric_card(
                        label="Engagement",
                        val=f"{engagement_info['value']:.0f}",
                        pct=engagement_info["delta"],
                        rank_now=engagement_info["rank"],
                        total_ranks=engagement_total if engagement_total else None,
                        metric_explanation="The total number of interactions (likes, comments, shares, reactions) on your social media posts. This metric reflects how actively your audience interacts with your content."
                    )
                else:
                    _format_simple_metric_card("Engagement", "N/A", metric_explanation="The total number of interactions (likes, comments, shares, reactions) on your social media posts. This metric reflects how actively your audience interacts with your content.")

            with col2:
                strength_info = strength_stats.get(norm)
                if strength_info:
                    _format_simple_metric_card(
                        label="Brand Strength",
                        val=f"{strength_info['value']:.1f}%",
                        pct=strength_info["delta"],
                        rank_now=strength_info["rank"],
                        total_ranks=strength_total if strength_total else None,
                        metric_explanation="A percentage representing how consistently your brand communicates through a dominant brand archetype. Higher percentages indicate stronger, more consistent brand messaging and positioning."
                    )
                else:
                    _format_simple_metric_card("Brand Strength", "N/A", metric_explanation="A percentage representing how consistently your brand communicates through a dominant brand archetype. Higher percentages indicate stronger, more consistent brand messaging and positioning.")

            with col3:
                creativity_info = creativity_stats.get(norm)
                if creativity_info:
                    _format_simple_metric_card(
                        label="Creativity",
                        val=f"{creativity_info['score']:.2f}",
                        pct=creativity_info["delta"],
                        rank_now=creativity_info["rank"],
                        total_ranks=creativity_total if creativity_total else None,
                        metric_explanation="A score measuring the originality and uniqueness of your social media content. Higher scores indicate more creative and distinctive messaging that stands out from competitors."
                    )
                else:
                    _format_simple_metric_card("Creativity", "N/A", metric_explanation="A score measuring the originality and uniqueness of your social media content. Higher scores indicate more creative and distinctive messaging that stands out from competitors.")

            detail = creativity_details.get(norm, {})
            if _has_creativity_content(detail):
                st.markdown("#### Creativity Analysis")
                _render_creativity_block(display_name, detail)

    start_date, end_date = get_selected_date_range()
    st.caption(
        f"Metrics evaluate performance using data up to the most recent posts, "
        f"with a rolling window anchored to activity between {start_date:%b %Y} and {end_date:%b %Y}."
    )

