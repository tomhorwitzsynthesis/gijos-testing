import streamlit as st
from typing import Union, List, Dict, Any, Optional
from utils.file_io import load_website_brand_jsons, load_website_category_json
import os
import pandas as pd
from utils.config import DATA_ROOT, WEBSITE_BRAND_MAPPING


def _val_or_obj(v, key="value", default="not stated"):
    if isinstance(v, dict):
        return v.get(key, default)
    if isinstance(v, str):
        return v
    return default


def _pill(text: str, tone: str = "neutral") -> str:
    tones = {
        "neutral": ("#F3F4F6", "#0F172A"),
        "green": ("#10B981", "#FFFFFF"),
        "amber": ("#F59E0B", "#0F172A"),
        "blue": ("#3B82F6", "#FFFFFF"),
        "rose": ("#EF4444", "#FFFFFF"),
        "slate": ("#E5E7EB", "#0F172A"),
    }
    bg, fg = tones.get(tone, tones["neutral"])  # background, foreground
    return f"<span class='ws-pill' style=\"background:{bg};color:{fg};\">{text}</span>"


def _quote(text: str) -> str:
    if not text:
        return ""
    return (
        "<div class=\"ws-quote\">"
        f"{text}</div>"
    )


def _render_comment_item(item: Union[dict, str]):
    if isinstance(item, str):
        html = (
            "<div class='ws-row'>"
            "<div class='ws-labels'></div>"
            f"<div class='ws-text'><div>{item}</div></div>"
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
        return
    comment = item.get("comment") or item.get("value") or "not stated"
    ref = (item.get("reference") or "").strip()
    summary = item.get("summary_points") or []
    labels_html = "".join([_pill(s, "slate")
                          for s in summary]) if summary else ""
    right_html = [f"<div>{comment}</div>"]
    if ref:
        right_html.append(f"<div class='ws-ref'>Reference: {ref}</div>")

    if labels_html:
        html = (
            "<div class='ws-row'>"
            f"<div class='ws-labels'>{labels_html}</div>"
            f"<div class='ws-text'>{''.join(right_html)}</div>"
            "</div>"
        )
    else:
        # No labels/tags → use single-column layout to avoid left gap
        html = (
            "<div class='ws-row ws-row--no-labels'>"
            f"<div class='ws-text'>{''.join(right_html)}</div>"
            "</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _split_items_by_type(items: List[Union[dict, str]]):
    stated: List[Union[dict, str]] = []
    inferred: List[Union[dict, str]] = []
    for it in items or []:
        t = ""
        if isinstance(it, dict):
            t = (it.get("type") or "").lower()
        if t == "stated":
            stated.append(it)
        else:
            inferred.append(it)
    return stated, inferred


def _kpi_chip(label: str, value: int) -> str:
    return (
        "<div class='ws-kpi'>"
        f"<div class='ws-kpi__value'>{value}</div>"
        f"<div class='ws-kpi__label'>{label}</div>"
        "</div>"
    )


def _card(title: str, body_html: str) -> None:
    st.markdown(
        """
        <div class="ws-card">
            <div class="ws-card__title">{title}</div>
            <div class="ws-card__body">{body}</div>
        </div>
        """.format(title=title, body=body_html),
        unsafe_allow_html=True,
    )


def _card_html(title: str, body_html: str) -> str:
    # Avoid leading spaces which Markdown could treat as a code block
    return (
        "<div class=\"ws-card\">"
        f"<div class=\"ws-card__title\">{title}</div>"
        f"<div class=\"ws-card__body\">{body_html}</div>"
        "</div>"
    )


def _filter_placeholder_items(items: List[Union[dict, str]]):
    """Remove placeholder items like 'not stated'/'not mentioned' when other informative items exist.

    If all items are placeholders, return the original list so the UI can show the absence.
    """
    def _text(it: Union[dict, str]) -> str:
        if isinstance(it, dict):
            return (it.get("comment") or it.get("value") or "").strip()
        return str(it).strip()

    items = items or []
    informative = [it for it in items if _text(it).lower()
                   not in {"not stated", "not mentioned"}]
    return informative if informative else items


def _is_placeholder_text(txt: str) -> bool:
    low = (txt or "").strip().lower()
    return low in {"not stated", "not mentioned"}


def _item_text(it: Union[dict, str]) -> str:
    if isinstance(it, dict):
        return (it.get("comment") or it.get("value") or "").strip()
    return str(it).strip()


def _render_brand_card(brand: str, data: dict, top_archetypes: Optional[List[Dict[str, Any]]] = None):
    display_brand = WEBSITE_BRAND_MAPPING.get((brand or "").lower(), brand)
    st.markdown(
        f"<h3 class='ws-brand'>{display_brand}</h3>", unsafe_allow_html=True)
    aud = data.get("audience", {}) or {}
    prod = data.get("product", {}) or {}
    model = data.get("model", {}) or {}

    # Archetypes below website name
    if top_archetypes:
        st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
        st.subheader("Top 3 Archetypes")
        col1, col2, col3 = st.columns(3)
        for j, archetype_info in enumerate(top_archetypes):
            col = col1 if j == 0 else col2 if j == 1 else col3
            with col:
                st.markdown(f"""
                <div style=\"border:1px solid #ddd; border-radius:12px; padding:22px; margin-bottom:14px; text-align:center;\">\n                    <div class=\"ws-rank\">RANK {j+1}</div>\n                    <h2 style=\"margin:0; color:#333; font-size:1.4em; line-height:1.2;\">{archetype_info['archetype']}</h2>\n                </div>
                """, unsafe_allow_html=True)

    # Section divider and heading for Product Analysis
    st.markdown("---")
    st.subheader("Product Analysis")

    # PRODUCT OVERVIEW — at the top, four equal cards
    framing = _val_or_obj(prod.get('product_framing'))
    primary_aud = _val_or_obj(aud.get('primary_audience'))
    type_of_product = _val_or_obj(prod.get('type_of_product'))
    benefits_focus = _val_or_obj(prod.get('benefits_focus'))
    overview_grid = (
        "<div class='ws-overview-grid'>"
        + _card_html("Overview", f"<div class='ws-framing'>{framing}</div>")
        + _card_html("Primary audience",
                     f"<div class='ws-framing'>{primary_aud}</div>")
        + _card_html("Type of product",
                     f"<div class='ws-framing'>{type_of_product}</div>")
        + _card_html("Benefit focus",
                     f"<div class='ws-framing'>{benefits_focus}</div>")
        + "</div>"
    )
    st.markdown(overview_grid, unsafe_allow_html=True)

    # CONTENT — sections vertically with conditional columns
    uc_items = _filter_placeholder_items(aud.get('use_cases_addressed') or [])
    uc_stated, uc_inferred = _split_items_by_type(uc_items)
    if uc_stated or uc_inferred:
        st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
        st.subheader("Use cases")
        if uc_stated and uc_inferred:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("STATED")
                for u in uc_stated:
                    _render_comment_item(u)
            with c2:
                st.caption("INFERRED")
                for u in uc_inferred:
                    _render_comment_item(u)
        elif uc_stated:
            st.caption("STATED")
            for u in uc_stated:
                _render_comment_item(u)
        else:
            st.caption("INFERRED")
            for u in uc_inferred:
                _render_comment_item(u)
        st.markdown("<div class='ws-section'></div>", unsafe_allow_html=True)

    vp_items = _filter_placeholder_items(
        prod.get('key_value_propositions') or [])
    vp_stated, vp_inferred = _split_items_by_type(vp_items)
    if vp_stated or vp_inferred:
        st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
        st.subheader("Value propositions")
        if vp_stated and vp_inferred:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("STATED")
                for v in vp_stated:
                    _render_comment_item(v)
            with c2:
                st.caption("INFERRED")
                for v in vp_inferred:
                    _render_comment_item(v)
        elif vp_stated:
            st.caption("STATED")
            for v in vp_stated:
                _render_comment_item(v)
        else:
            st.caption("INFERRED")
            for v in vp_inferred:
                _render_comment_item(v)
        st.markdown("<div class='ws-section'></div>", unsafe_allow_html=True)

    fh_items = _filter_placeholder_items(
        prod.get('feature_highlights') or [])
    fh_stated, fh_inferred = _split_items_by_type(fh_items)
    if fh_stated or fh_inferred:
        st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
        st.subheader("Feature highlights")
        if fh_stated and fh_inferred:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("STATED")
                for f in fh_stated:
                    _render_comment_item(f)
            with c2:
                st.caption("INFERRED")
                for f in fh_inferred:
                    _render_comment_item(f)
        elif fh_stated:
            st.caption("STATED")
            for f in fh_stated:
                _render_comment_item(f)
        else:
            st.caption("INFERRED")
            for f in fh_inferred:
                _render_comment_item(f)
        st.markdown("<div class='ws-section'></div>", unsafe_allow_html=True)

    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
    st.subheader("Pricing & monetization")
    pricing_text = _val_or_obj(model.get('pricing_or_monetization_hints'))
    if _is_placeholder_text(pricing_text):
        st.markdown(_quote("Pricing details are not specified."),
                    unsafe_allow_html=True)
    else:
        st.markdown(_quote(pricing_text), unsafe_allow_html=True)

    eco_items = model.get('ecosystem / integrations') or []
    eco_raw = eco_items if isinstance(eco_items, list) else [eco_items]
    eco_list = _filter_placeholder_items(eco_raw)
    eco_stated, eco_inferred = _split_items_by_type(eco_list)
    # Determine if everything is placeholder or empty
    eco_has_real = any(not _is_placeholder_text(_item_text(x))
                       for x in eco_raw)
    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
    st.subheader("Integrations")
    if eco_has_real and (eco_stated or eco_inferred):
        if eco_stated and eco_inferred:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("STATED")
                for e in eco_stated:
                    _render_comment_item(e)
            with c2:
                st.caption("INFERRED")
                for e in eco_inferred:
                    _render_comment_item(e)
        elif eco_stated:
            st.caption("STATED")
            for e in eco_stated:
                _render_comment_item(e)
        else:
            st.caption("INFERRED")
            for e in eco_inferred:
                _render_comment_item(e)
    else:
        st.markdown(_quote("Integration details are not specified."),
                    unsafe_allow_html=True)
    st.markdown("<div class='ws-section'></div>", unsafe_allow_html=True)

    cred_items = model.get('credibility_and_social_proof') or []
    cred_raw = cred_items if isinstance(cred_items, list) else [cred_items]
    cred_list = _filter_placeholder_items(cred_raw)
    cr_stated, cr_inferred = _split_items_by_type(cred_list)
    cr_has_real = any(not _is_placeholder_text(_item_text(x))
                      for x in cred_raw)
    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
    st.subheader("Social proof")
    if cr_has_real and (cr_stated or cr_inferred):
        if cr_stated and cr_inferred:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("STATED")
                for c in cr_stated:
                    _render_comment_item(c)
            with c2:
                st.caption("INFERRED")
                for c in cr_inferred:
                    _render_comment_item(c)
        elif cr_stated:
            st.caption("STATED")
            for c in cr_stated:
                _render_comment_item(c)
        else:
            st.caption("INFERRED")
            for c in cr_inferred:
                _render_comment_item(c)
    else:
        st.markdown(_quote("Social proof details are not specified."),
                    unsafe_allow_html=True)
    st.markdown("<div class='ws-section'></div>", unsafe_allow_html=True)

    # (product overview now shown at the top)


def _render_category(meta: dict):
    def _humanize_title(s: str) -> str:
        try:
            return (s or "").replace("_", " ").capitalize()
        except Exception:
            return s
    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
    if not meta:
        st.info("No category overview available.")
        return
    over = meta.get("overlapping_practices", {}) or {}
    diff = meta.get("differentiated_practices", {}) or {}
    cat = meta.get("category_synthesis", {}) or {}
    ws_items = (cat.get("white_space_opportunities") or [])

    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)

    # Overlapping practices — per-field cards in a responsive grid
    st.subheader("Overlapping practices")
    if isinstance(over, dict) and over:
        items_over = list(over.items())
        ncols = 3 if len(items_over) >= 3 else (
            2 if len(items_over) == 2 else 1)
        for i in range(0, len(items_over), ncols):
            cols = st.columns(ncols)
            for j, (field, bullets) in enumerate(items_over[i:i + ncols]):
                with cols[j]:
                    # Normalize bullets to list
                    items = bullets if isinstance(bullets, list) else (
                        [bullets] if bullets else [])
                    if len(items) == 1:
                        body = f"<div>{items[0]}</div>"
                    else:
                        body = (
                            "<ul class='ws-list'>"
                            + "".join([f"<li>{b}</li>" for b in items])
                            + "</ul>"
                        )
                    _card(_humanize_title(field), body)
    else:
        st.caption("No overlapping practices.")

    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)

    # Differentiated practices — per-field cards in a responsive grid
    st.subheader("Differentiated practices")
    if isinstance(diff, dict) and diff:
        items_diff = list(diff.items())
        ncols = 3 if len(items_diff) >= 3 else (
            2 if len(items_diff) == 2 else 1)
        for i in range(0, len(items_diff), ncols):
            cols = st.columns(ncols)
            for j, (field, bullets) in enumerate(items_diff[i:i + ncols]):
                with cols[j]:
                    items = bullets if isinstance(bullets, list) else (
                        [bullets] if bullets else [])
                    if len(items) == 1:
                        body = f"<div>{items[0]}</div>"
                    else:
                        body = (
                            "<ul class='ws-list'>"
                            + "".join([f"<li>{b}</li>" for b in items])
                            + "</ul>"
                        )
                    _card(_humanize_title(field), body)
    else:
        st.caption("No differentiated practices.")

    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)

    # White space opportunities — ordered list for scan-ability
    st.subheader("White space opportunities")
    if ws_items:
        items = ws_items if isinstance(ws_items, list) else (
            [ws_items] if ws_items else [])
        if len(items) == 1:
            body_ws = f"<div>{items[0]}</div>"
        else:
            body_ws = (
                "<ol class='ws-list'>"
                + "".join([f"<li>{w}</li>" for w in items])
                + "</ol>"
            )
        _card("Ideas", body_ws)
    else:
        st.caption("No white space opportunities.")

    st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)

    # Category narrative — split into two clear cards
    st.subheader("Category narrative")
    col_left, col_right = st.columns(2)
    with col_left:
        state_text = _val_or_obj(cat.get("state_of_category"))
        _card("State of category", _quote(state_text)
              if state_text else "<div class='ws-empty'>(none)</div>")
    with col_right:
        dom_text = _val_or_obj(cat.get("dominant_narrative"))
        _card("Dominant narrative", _quote(dom_text)
              if dom_text else "<div class='ws-empty'>(none)</div>")


def render():
    st.title("Website Intelligence Dashboard")
    # Loader for website archetypes summary

    def _load_top_archetypes_from_website_compos() -> Dict[str, List[Dict[str, Any]]]:
        try:
            compos_path = os.path.join(
                DATA_ROOT, "website", "analysis", "compos", "compos_analysis_website.xlsx")
            if not os.path.exists(compos_path):
                return {}
            try:
                df = pd.read_excel(compos_path, sheet_name="Summary")
            except Exception:
                df = pd.read_excel(compos_path)
            if df is None or df.empty:
                return {}

            # Confidence columns end with "Avg Confidence"
            conf_cols = [c for c in df.columns if c.endswith("Avg Confidence")]
            site_col = "Site" if "Site" in df.columns else df.columns[0]
            if not conf_cols:
                return {}

            archetypes_map: Dict[str, List[Dict[str, Any]]] = {}
            for _, row in df.iterrows():
                site = str(row.get(site_col)) if site_col in df.columns else str(
                    row.iloc[0])
                scores = []
                total_conf = 0.0
                for col in conf_cols:
                    name = col.replace(" Avg Confidence", "").strip()
                    try:
                        conf = float(row.get(col, 0) or 0)
                    except Exception:
                        conf = 0.0
                    total_conf += max(0.0, conf)
                    scores.append((name, max(0.0, conf)))
                if not scores:
                    continue
                # Normalize to percentages and take top 3
                items = []
                denom = total_conf if total_conf > 0 else sum(
                    conf for _, conf in scores) or 1.0
                for name, conf in sorted(scores, key=lambda x: x[1], reverse=True)[:3]:
                    pct = (conf / denom) * 100.0
                    items.append({
                        "archetype": name,
                        "percentage": pct,
                        # use confidence as a count proxy for UI symmetry
                        "count": int(round(conf))
                    })
                if items:
                    archetypes_map[site] = items
            return archetypes_map
        except Exception:
            return {}

    website_archetypes = _load_top_archetypes_from_website_compos()
    # CSS once
    st.markdown(
        """
        <style>
        :root {
            --ws-bg: #FFFFFF;
            --ws-text: #0F172A;
            --ws-muted: #6B7280;
            --ws-border: #E5E7EB;
            --ws-subtle: #F3F4F6;
            --ws-soft: #F9FAFB;
            --ws-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
        }
        .ws-brand{margin:0 0 4px;color:var(--ws-text)}
        .ws-card{border:1px solid var(--ws-border);border-radius:14px;padding:16px 16px;margin:10px 0;background:var(--ws-bg);box-shadow:none}
        .ws-card__title{font-weight:700;color:var(--ws-text);margin-bottom:10px;letter-spacing:.02em;font-size:13px;text-transform:uppercase}
        .ws-card__body{color:var(--ws-text)}
        .ws-framing{font-size:18px;line-height:1.6;color:var(--ws-text)}
        .ws-chip-row{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px}
        .ws-chip{background:var(--ws-subtle);border-radius:999px;padding:6px 10px;display:inline-flex;gap:6px}
        .ws-chip-k{color:var(--ws-muted)}
        .ws-chip-v{color:var(--ws-text)}
        .ws-pill{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;line-height:1}
        .ws-quote{margin-top:6px;border-left:3px solid var(--ws-border);padding:6px 10px;color:#374151;background:var(--ws-soft);border-radius:6px}
        .ws-rank{display:inline-block;background:var(--ws-subtle);border:1px solid var(--ws-border);color:var(--ws-muted);font-size:11px;letter-spacing:.06em;border-radius:999px;padding:4px 8px;margin-bottom:8px;text-transform:uppercase}
        .ws-stats{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 8px}
        .ws-kpi{background:var(--ws-subtle);border:1px solid var(--ws-border);border-radius:12px;padding:10px 14px}
        .ws-kpi__value{font-size:20px;font-weight:700;color:var(--ws-text);line-height:1}
        .ws-kpi__label{font-size:12px;color:var(--ws-muted);margin-top:4px}
        .ws-field{font-weight:600;margin:8px 0 4px;color:var(--ws-text)}
        .ws-list{margin:0 0 6px 18px}
        .ws-empty{color:#9CA3AF}
        .ws-row{display:grid;grid-template-columns:240px 1fr;gap:12px;align-items:flex-start;padding:12px 0;border-bottom:1px solid var(--ws-border)}
        .ws-row--no-labels{grid-template-columns:1fr}
        .ws-labels{display:flex;flex-wrap:wrap;gap:6px}
        .ws-text{display:flex;flex-direction:column;gap:6px}
        .ws-ref{color:#6B7280;font-size:12px}
        .ws-section{margin:40px 0}
        .ws-gap{height:24px}
        .ws-overview-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;align-items:stretch}
        .ws-overview-grid .ws-card{height:100%}
        </style>
        """,
        unsafe_allow_html=True,
    )
    brands = load_website_brand_jsons()
    meta = load_website_category_json()

    tabs = st.tabs(["By brand", "Category"])
    with tabs[0]:
        if not brands:
            st.info(
                "No website analyses found. Run the Website analysis to populate data.")
        else:
            # Brand selector at top (consistent with other pages)
            brand_keys = sorted(brands.keys())
            selected = st.selectbox(
                "Brand", brand_keys, format_func=lambda k: WEBSITE_BRAND_MAPPING.get((k or "").lower(), k))
            tops = website_archetypes.get(
                selected) if 'website_archetypes' in locals() else None
            _render_brand_card(selected, brands[selected], tops)
    with tabs[1]:
        # Overall archetypes at category tab top
        if website_archetypes:
            overall_counts = {}
            for archetypes in website_archetypes.values():
                for item in archetypes:
                    a = item['archetype']
                    c = item['count']
                    overall_counts[a] = overall_counts.get(a, 0) + c
            top3 = sorted(overall_counts.items(),
                          key=lambda x: x[1], reverse=True)[:3]
            st.markdown("<div class='ws-gap'></div>", unsafe_allow_html=True)
            st.subheader("Overall - Top 3 Archetypes")
            col1, col2, col3 = st.columns(3)
            for j, (name, _) in enumerate(top3):
                col = col1 if j == 0 else col2 if j == 1 else col3
                with col:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:18px; margin-bottom:10px; text-align:center;">
                        <div class="ws-rank">RANK {j+1}</div>
                        <h4 style="margin:0; color:#333; font-size:1.1rem;">{name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(
                "No website archetype data available. Ensure compos file is in data/website/analysis/compos.")

        _render_category(meta)
