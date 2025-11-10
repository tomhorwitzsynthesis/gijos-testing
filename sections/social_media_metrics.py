from typing import Dict, Tuple

import os
import glob
import streamlit as st
import pandas as pd

from utils.config import BRAND_NAME_MAPPING, DATA_ROOT
from utils.date_utils import get_selected_date_range
from utils.file_io import load_social_data, load_creativity_rankings, load_brand_summaries


SOCIAL_COMPOS_DIR = os.path.join(DATA_ROOT, "social_media", "compos")


def _normalize_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    base = name.split("|")[0].strip()
    cleaned = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in base)
    return " ".join(cleaned.split())


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
    for brand_key, brand_display in BRAND_NAME_MAPPING.items():
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
            strength[brand_candidate] = float(counts.iloc[0] / total * 100)
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
    df["brand_norm"] = df["brand"].apply(_normalize_brand)
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
            "brand": row["brand"],
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
    return stats, df["brand"].nunique(), details


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
    preferred_order = list(BRAND_NAME_MAPPING.values())
    ordered = [b for b in preferred_order if b in brand_to_summary]
    extras = sorted([b for b in brand_to_summary.keys() if b not in ordered])
    tab_labels = ordered + extras
    st.markdown("### Executive Summary")
    tabs = st.tabs(tab_labels)
    card_style = (
        "border:1px solid #ddd; border-radius:10px; padding:15px; "
        "margin-top:10px; margin-bottom:10px; background-color:#fafafa;"
    )
    text_style = "margin:0; color:#333; line-height:1.5;"
    meta_style = "margin:0 0 10px 0; color:#666; font-size:0.9em;"
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
    st.subheader("📱 Social Media Brand Metrics")

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
    for display in BRAND_NAME_MAPPING.values():
        norm = _normalize_brand(display)
        if (norm in engagement_stats) or (norm in strength_stats) or (norm in creativity_stats):
            available_brands.append(display)
            seen.add(display)
    extra_keys = set(engagement_stats.keys()) | set(strength_stats.keys()) | set(creativity_stats.keys())
    for norm in sorted(extra_keys):
        display = next((name for name in frames.keys() if _normalize_brand(name) == norm), norm.title())
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
                    )
                else:
                    _format_simple_metric_card("Engagement", "N/A")

            with col2:
                strength_info = strength_stats.get(norm)
                if strength_info:
                    _format_simple_metric_card(
                        label="Brand Strength",
                        val=f"{strength_info['value']:.1f}%",
                        pct=strength_info["delta"],
                        rank_now=strength_info["rank"],
                        total_ranks=strength_total if strength_total else None,
                    )
                else:
                    _format_simple_metric_card("Brand Strength", "N/A")

            with col3:
                creativity_info = creativity_stats.get(norm)
                if creativity_info:
                    _format_simple_metric_card(
                        label="Creativity",
                        val=f"{creativity_info['score']:.2f}",
                        pct=creativity_info["delta"],
                        rank_now=creativity_info["rank"],
                        total_ranks=creativity_total if creativity_total else None,
                    )
                else:
                    _format_simple_metric_card("Creativity", "N/A")

            detail = creativity_details.get(norm, {})
            if _has_creativity_content(detail):
                st.markdown("#### Creativity Analysis")
                _render_creativity_block(display_name, detail)

    start_date, end_date = get_selected_date_range()
    st.caption(
        f"Metrics evaluate performance using data up to the most recent posts, "
        f"with a rolling window anchored to activity between {start_date:%b %Y} and {end_date:%b %Y}."
    )

