import streamlit as st

# Set page configuration - must be called before any other Streamlit commands
st.set_page_config(
    layout="wide",  # Always use wide mode
    page_title="Dashboard",  # Page title in browser tab
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)

from utils.date_utils import init_month_selector

# --- Section Imports ---
from sections.compos_matrix import render as render_matrix
from sections.sentiment_analysis import render as render_sentiment
from sections.topical_analysis import render as render_topics
from sections.volume_trends import render as render_volume
from sections.media_coverage import render as render_media_shares
from sections.pr_metrics import render as render_pr_metrics
from sections.pr_archetype_matrix import render as render_pr_archetype_matrix

from sections.volume_engagement_trends import render as render_social_trends
from sections.social_media_top_posts import render as render_top_posts
from sections.social_media_metrics import render as render_social_metrics
from sections.social_media_archetype_matrix import render as render_social_archetype_matrix
from sections.social_media_chatkit import render as render_chatkit

from sections.content_pillars import render as render_content_pillars

from sections.audience_affinity import render as render_audience_affinity
from sections.ads_dashboard import render as render_ads_dashboard

#from sections.content_pillar_analysis import render as render_pillars  # If implemented
# from sections.audience_affinity import render as render_affinity     # Optional

# --- Sidebar ---
st.sidebar.title("📁 Navigation")
section = st.sidebar.radio("Go to", [
    "Press Releases",
    "Social Media",
    "Content Pillars",
    "Audience Affinity",
    "Ad Intelligence"
])

# --- Month Filter ---
init_month_selector()  # Sets start_date / end_date globally

# --- Section Routing ---
if section == "Press Releases":
    st.title("📰 Press Release Dashboard")
    render_pr_metrics()
    render_pr_archetype_matrix()
    render_matrix()
    render_sentiment(mode="by_company")
    render_topics()
    render_volume(mode="by_company")
    render_media_shares(mode="by_brand")

elif section == "Social Media":
    st.title("📱 Social Media Dashboard")
    render_social_metrics(selected_platforms=["linkedin"])
    render_social_archetype_matrix()
    render_social_trends(selected_platforms=["linkedin"])
    render_top_posts(selected_platforms=["linkedin"])
    # render_chatkit()

elif section == "Content Pillars":
    st.title("🧱 Content Pillar Dashboard")
    render_content_pillars()
    #st.info("This section is under construction")

elif section == "Audience Affinity":
    st.title("🎯 Audience Affinity Dashboard")
    render_audience_affinity()
    #st.info("This section is under construction")

elif section == "Ad Intelligence":
    st.title("📣 Ad Intelligence Dashboard")
    render_ads_dashboard()
