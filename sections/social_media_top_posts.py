import streamlit as st
import pandas as pd
import html
from utils.config import BRAND_NAME_MAPPING, BRANDS
from utils.date_utils import get_selected_date_range
from utils.file_io import load_social_data

POST_TEXT_COLUMNS = ["Post", "post_text", "content"]


def _render_post_cards(posts_df: pd.DataFrame):
    """Render stacked cards for top posts."""
    if posts_df.empty:
        st.info("No posts to display.")
        return

    for _, row in posts_df.iterrows():
        engagement = int(row.get("Engagement", 0))
        company = html.escape(str(row.get("Company", "")))
        date = html.escape(str(row.get("Date", "")))
        post_preview = html.escape(str(row.get("PostPreview", "")))
        url = row.get("URL")
        if url:
            post_html = f'<a href="{html.escape(str(url))}" target="_blank" style="color:#00C35F; text-decoration:none;">"{post_preview}"</a>'
        else:
            post_html = f'<span style="color:#00C35F;">"{post_preview}"</span>'

        card_html = (
            f'<div style="border:1px solid #ddd; border-radius:12px; padding:18px; margin-bottom:16px; '
            f'background-color:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.06);">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">'
            f'<div><p style="margin:0; font-size:0.9em; color:#6B7280;">Company</p>'
            f'<h5 style="margin:4px 0; color:#111827;">{company}</h5>'
            f'<p style="margin:0; font-size:0.9em; color:#6B7280;">Date</p>'
            f'<p style="margin:4px 0 0; color:#111827;">{date}</p></div>'
            f'<div style="text-align:right;">'
            f'<p style="margin:0; font-size:0.9em; color:#6B7280;">Engagement</p>'
            f'<p style="margin:4px 0; font-size:1.8em; color:#0E34A0; font-weight:700;">{engagement:,}</p>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:12px;">'
            f'<p style="margin:0; font-size:0.95em; color:#00C35F;">{post_html}</p>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

def render(selected_platforms=None):
    if selected_platforms is None:
        selected_platforms = ["linkedin"]

    selected_platforms = [platform.lower() for platform in selected_platforms if platform.lower() == "linkedin"]
    if not selected_platforms:
        st.info("No supported social platforms selected.")
        return

    st.subheader("Top Social Media Posts")

    start_date, end_date = get_selected_date_range()

    for platform in selected_platforms:
        st.markdown(f"### {platform.capitalize()}")
        all_posts = []

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
            if df is None or df.empty or "Published Date" not in df.columns:
                continue

            df = df.dropna(subset=["Published Date"])
            df = df[(df["Published Date"] >= start_date) & (df["Published Date"] < end_date)]
            if df.empty:
                continue

            post_col = next((col for col in POST_TEXT_COLUMNS if col in df.columns), None)
            if not post_col:
                continue
            
            url_col = "url" if "url" in df.columns else None
            if not url_col:
                continue

            df["Engagement"] = (
                df.get("num_likes", 0).fillna(0) +
                df.get("num_comments", 0).fillna(0) * 3
            )

            df = df[df["Engagement"] > 0]
            if df.empty:
                continue

            for _, row in df.iterrows():
                full_text = str(row[post_col]).strip()
                preview = full_text.replace("\n", " ")
                preview = preview[:180] + "..." if len(preview) > 180 else preview
                url = row[url_col]
                all_posts.append({
                    "Company": brand_display,
                    "Date": row["Published Date"].strftime('%Y-%m-%d'),
                    "PostPreview": preview,
                    "URL": url,
                    "Engagement": int(row["Engagement"])
                })

        if not all_posts:
            st.info(f"No {platform.capitalize()} posts found in the selected date range.")
            continue

        df_all = pd.DataFrame(all_posts).sort_values(by="Engagement", ascending=False)

        # Use BRANDS to avoid duplicates
        brand_display_names = BRANDS
        tab_labels = ["Overall"] + [f"{brand}" for brand in brand_display_names]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.markdown("**Top 5 posts overall**")
            _render_post_cards(df_all.head(5))

        for i, brand_display in enumerate(brand_display_names, start=1):
            with tabs[i]:
                brand_df = df_all[df_all["Company"] == brand_display]
                if brand_df.empty:
                    st.info(f"No posts for {brand_display}.")
                else:
                    st.markdown(f"**Top posts for {brand_display}**")
                    _render_post_cards(brand_df.head(5))

