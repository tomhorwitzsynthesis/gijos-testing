from pathlib import Path
import streamlit as st

# Set page configuration - must be called before any other Streamlit commands
st.set_page_config(
    layout="wide",  # Always use wide mode
    page_title="Dashboard",  # Page title in browser tab
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)

# Initialize font preference in session state
# if 'use_inter_font' not in st.session_state:
#     st.session_state.use_inter_font = False

st.session_state.use_inter_font = True


BASE_DIR = Path(__file__).resolve().parent
BANNER_IMAGE = BASE_DIR / "visuals" / "synthesis_cover.jpg"

# Apply font styling based on user preference
if st.session_state.use_inter_font:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebarContent"], 
            [data-testid="stHeader"], [data-testid="stToolbar"], 
            .stApp, .stAppViewContainer, .stSidebar,
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] h1,
            div[data-testid="stMarkdownContainer"] h2,
            div[data-testid="stMarkdownContainer"] h3,
            .stText, .stSelectbox, .stRadio, .stButton, .stDataFrame,
            .stMetric, .stMarkdown, .element-container {
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
# When False, no CSS override - Streamlit uses its default font

from utils.date_utils import init_month_selector

# --- Section Imports ---
from sections.compos_matrix import render as render_matrix
from sections.sentiment_analysis import render as render_sentiment
from sections.topical_analysis_backlog import render as render_topics
from sections.volume_trends import render as render_volume
from sections.media_coverage import render as render_media_shares
from sections.pr_metrics import render as render_pr_metrics
from sections.pr_archetype_matrix import render as render_pr_archetype_matrix

from sections.volume_engagement_trends import render as render_social_trends
from sections.social_media_top_posts import render as render_top_posts
from sections.social_media_metrics import render as render_social_metrics
from sections.social_media_archetype_matrix import render as render_social_archetype_matrix
from sections.social_media_chatkit import render as render_chatkit
from sections.social_media_employer_branding_themes import render as render_employer_branding_themes

from sections.content_pillars import render as render_content_pillars

from sections.audience_affinity import render as render_audience_affinity
from sections.ads_dashboard import render as render_ads_dashboard
from sections.ads_employer_branding_themes import render as render_ads_employer_branding_themes
from sections.employer_branding import render as render_employer_branding
from sections.employer_branding_archetype_matrix import render as render_employer_branding_archetype_matrix

#from sections.content_pillar_analysis import render as render_pillars  # If implemented
# from sections.audience_affinity import render as render_affinity     # Optional

def render_banner():
    if BANNER_IMAGE.exists():
        st.image(str(BANNER_IMAGE), use_container_width=True)
    else:
        st.warning("Banner image not found.")


# --- Sidebar ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Press Releases",
    "Social Media",
    "Content Pillars",
    "Audience Affinity",
    "Ad Intelligence",
    "Employer Branding",
])

# # Font toggle in sidebar
# st.sidebar.divider()
# st.sidebar.subheader("Settings")
# st.sidebar.toggle(
#     "Use Inter Font",
#     value=st.session_state.use_inter_font,
#     key="use_inter_font",
#     help="Toggle between Inter font (Google Fonts) and Streamlit's default font"
# )

# --- Month Filter ---
init_month_selector()  # Sets start_date / end_date globally
# render_banner()

# --- Section Routing ---
if section == "Press Releases":
    st.title("Press Release Dashboard")
    render_pr_metrics()
    render_pr_archetype_matrix()
    render_matrix()
    render_sentiment(mode="by_company")
    render_topics()
    render_volume(mode="by_company")
    render_media_shares(mode="by_brand")

elif section == "Social Media":
    st.title("Social Media Dashboard")
    render_social_metrics(selected_platforms=["linkedin"])
    render_social_archetype_matrix()
    render_social_trends(selected_platforms=["linkedin"])
    render_top_posts(selected_platforms=["linkedin"])
    render_employer_branding_themes()
    # render_chatkit()

elif section == "Content Pillars":
    st.title("Content Pillar Dashboard")
    render_content_pillars()
    #st.info("This section is under construction")

elif section == "Audience Affinity":
    st.title("Audience Affinity Dashboard")
    render_audience_affinity()
    #st.info("This section is under construction")

elif section == "Ad Intelligence":
    st.title("Ad Intelligence Dashboard")
    render_ads_dashboard()
    render_ads_employer_branding_themes()

elif section == "Employer Branding":
    st.title("Employer Branding Dashboard")
    render_employer_branding_archetype_matrix()
    render_employer_branding()
