from typing import Dict, Tuple

import streamlit as st
import pandas as pd

from utils.config import BRANDS, BRAND_NAME_MAPPING
from utils.date_utils import get_selected_date_range
from utils.file_io import load_agility_data, load_creativity_rankings, load_brand_summaries


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


@st.cache_data(ttl=0)
def _load_press_release_frames():
    frames = {}
    for brand in BRANDS:
        df = load_agility_data(brand)
        if df is None or df.empty:
            continue
        if "Published Date" not in df.columns:
            continue
        df = df.copy()
        df["Published Date"] = pd.to_datetime(df["Published Date"], errors="coerce")
        df = df.dropna(subset=["Published Date"])
        if df.empty:
            continue
        if "Impressions" in df.columns:
            df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
        else:
            df["Impressions"] = 0
        display_name = BRAND_NAME_MAPPING.get(brand, brand)
        df["Company"] = display_name
        frames[display_name] = df
    return frames


def _compute_impressions_stats(frames: dict) -> Tuple[Dict[str, dict], int]:
    if not frames:
        return {}, 0
    impression_map = {}
    for brand, df in frames.items():
        windowed = _rolling_window(df, "Published Date")
        if windowed.empty or "Impressions" not in windowed.columns:
            continue
        total_impressions = windowed["Impressions"].sum()
        impression_map[brand] = float(total_impressions)
    if not impression_map:
        return {}, 0
    series = pd.Series(impression_map)
    impressions_mean = series.mean()
    impressions_ranks = series.rank(ascending=False, method="min")
    stats = {}
    for brand_name, value in series.items():
        norm = _normalize_brand(brand_name)
        denom = impressions_mean if impressions_mean != 0 else 1
        delta = ((value - impressions_mean) / denom) * 100 if impressions_mean else 0
        rank_now = impressions_ranks.get(brand_name, None)
        stats[norm] = {
            "value": float(value),
            "delta": float(delta),
            "rank": int(rank_now) if pd.notna(rank_now) else None,
        }
    return stats, len(series)


def _compute_brand_strength_stats(frames: dict) -> Tuple[Dict[str, dict], int]:
    if not frames:
        return {}, 0
    strength_map = {}
    for brand, df in frames.items():
        if "Top Archetype" not in df.columns:
            continue
        counts = df["Top Archetype"].dropna().value_counts()
        total = counts.sum()
        if total > 0:
            strength_map[brand] = float(counts.iloc[0] / total * 100)
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
    df = load_creativity_rankings("pr")
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
    preferred_order = [BRAND_NAME_MAPPING.get(b, b) for b in BRANDS]
    ordered = [b for b in preferred_order if b in brand_to_summary]
    extras = sorted([b for b in brand_to_summary.keys() if b not in ordered])
    tab_labels = ordered + extras
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


def render():
    st.subheader("📈 Press Release Brand Metrics")

    start_date, end_date = get_selected_date_range()
    _render_summary_tabs(load_brand_summaries("pr"))
    frames = _load_press_release_frames()
    if not frames:
        st.info("No press release data available.")
        return

    impressions_stats, impressions_total = _compute_impressions_stats(frames)
    strength_stats, strength_total = _compute_brand_strength_stats(frames)
    creativity_stats, creativity_total, creativity_details = _compute_creativity_stats()

    available_brands = []
    seen = set()
    for display in [BRAND_NAME_MAPPING.get(b, b) for b in BRANDS]:
        norm = _normalize_brand(display)
        if (norm in impressions_stats) or (norm in strength_stats) or (norm in creativity_stats):
            available_brands.append(display)
            seen.add(display)
    extra_norms = set(impressions_stats.keys()) | set(strength_stats.keys()) | set(creativity_stats.keys())
    for norm in sorted(extra_norms):
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
                impressions_info = impressions_stats.get(norm)
                if impressions_info:
                    _format_simple_metric_card(
                        label="Impressions",
                        val=f"{int(impressions_info['value']):,}",
                        pct=impressions_info["delta"],
                        rank_now=impressions_info["rank"],
                        total_ranks=impressions_total if impressions_total else None,
                    )
                else:
                    _format_simple_metric_card("Impressions", "N/A")

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

