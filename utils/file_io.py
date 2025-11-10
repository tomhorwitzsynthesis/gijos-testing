import os
import pandas as pd
import streamlit as st
from utils.config import DATA_ROOT, BRAND_NAME_MAPPING  # <-- import here

# ------------------------
# 📝 Load Media Summaries
# ------------------------

_SUMMARY_FILE_MAP = {
    "ads": "ads_summaries.json",
    "pr": "pr_summaries.json",
    "social_media": "social_media.json",
}


@st.cache_data(ttl=0)
def load_brand_summaries(media_type: str):
    filename = _SUMMARY_FILE_MAP.get(media_type, f"{media_type}.json")
    path = os.path.join(DATA_ROOT, "summaries", filename)
    if not os.path.exists(path):
        return []
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as e:
        st.error(f"[Summaries] Error loading summaries for '{media_type}': {e}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []

# ------------------------
# 🎨 Load Creativity Rankings by Media Type
# ------------------------

@st.cache_data(ttl=0)
def load_creativity_rankings(media_type: str) -> pd.DataFrame:
	path = os.path.join(DATA_ROOT, "creativity", media_type, "creativity_ranking.xlsx")
	if not os.path.exists(path):
		return pd.DataFrame(columns=["brand", "rank", "originality_score", "justification", "examples"])
	try:
		df_cre = pd.read_excel(path, sheet_name="Overall Ranking")
	except ValueError:
		df_cre = pd.read_excel(path)
	except Exception as e:
		st.error(f"[Creativity] Error loading rankings for '{media_type}': {e}")
		return pd.DataFrame(columns=["brand", "rank", "originality_score", "justification", "examples"])

	df_cre = df_cre.rename(columns={c: str(c).lower() for c in df_cre.columns})
	required_cols = ["brand", "rank", "originality_score", "justification", "examples"]
	for col in required_cols:
		if col not in df_cre.columns:
			df_cre[col] = None if col in {"rank", "originality_score"} else ""
	return df_cre[required_cols]

# ------------------------
# 📄 Load Agility (News)
# ------------------------
@st.cache_data
def load_agility_data(company_name: str):
    path = os.path.join(DATA_ROOT, "agility", f"{company_name.lower()}_agility.xlsx")
    if not os.path.exists(path):
        return None
    try:
        xls = pd.ExcelFile(path)
        target_sheet = "Raw Data" if "Raw Data" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=target_sheet)
    except Exception as e:
        st.error(f"[Agility] Error loading {company_name}: {e}")
        return None

    # Normalize Published Date column if needed
    if "Published Date" not in df.columns:
        for candidate in ["PublishedDate", "published_date", "Date", "date"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Published Date"})
                break

    return df

# ------------------------
# 📱 Load Social Media Data
# ------------------------

@st.cache_data
def load_social_data(company_name: str, platform: str = "linkedin"):
	"""Load LinkedIn data for a company, normalize date format."""
	platform = platform.lower()
	if platform != "linkedin":
		raise ValueError("Only the 'linkedin' platform is supported.")

	path = os.path.join(DATA_ROOT, "social_media", "linkedin_posts.xlsx")
	if not os.path.exists(path):
		return None

	try:
		df = pd.read_excel(path, sheet_name=0)
	except Exception as e:
		st.error(f"[Social] Error loading LinkedIn data for {company_name}: {e}")
		return None

	company_col = "user_id"
	if company_col not in df.columns:
		st.error("[Social] Company column 'user_id' not found in linkedin_posts.xlsx")
		return None

	df = df[df[company_col].astype(str).str.strip().str.lower() == company_name.lower()].copy()
	if df.empty:
		return None

	if "Published Date" in df.columns:
		df["Published Date"] = pd.to_datetime(df["Published Date"], utc=True, errors="coerce")
	elif "date_posted" in df.columns:
		df["Published Date"] = pd.to_datetime(df["date_posted"], utc=True, errors="coerce")
	elif "timestamp" in df.columns:
		df["Published Date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
	else:
		st.warning(
			"[Social] No recognizable date column in linkedin_posts.xlsx. "
			f"Available columns: {list(df.columns)}"
		)
		return None

	df["Published Date"] = df["Published Date"].dt.tz_localize(None)

	return df

# ------------------------
# 🗃️ Load the Actual Volume 
# ------------------------

AGILITY_METADATA_PATH = os.path.join(DATA_ROOT, "agility", "agility_metadata.xlsx")

def load_agility_volume_map():
	if os.path.exists(AGILITY_METADATA_PATH):
		return pd.read_excel(AGILITY_METADATA_PATH, index_col="Company").to_dict()["Volume"]
	else:
		return {}

# ------------------------
# 🗃️ Load All Brands' Social Media Data for a Platform
# ------------------------

def load_all_social_data(brands, platform: str = "linkedin"):
	"""Return dict[brand] = DataFrame for the selected platform."""
	results = {}
	for brand in brands:
		df = load_social_data(brand, platform)
		if df is not None and not df.empty:
			results[brand] = df
	return results

# ------------------------
# 📢 Load Ads Intelligence data
# ------------------------

@st.cache_data
def load_ads_data():
	"""
	Load ads scraping Excel and normalize key fields.
	Returns a pandas DataFrame or None if not found.
	"""
	# Build potential roots: configured root, module-relative root, and CWD/data
	module_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.abspath(os.path.join(module_dir, os.pardir))
	module_data_root = os.path.join(project_root, "data")
	cwd_data_root = os.path.join(os.getcwd(), "data")
	roots = [DATA_ROOT, module_data_root, cwd_data_root]

	candidate_filenames = [
		"ads.xlsx",
		"ads_scraping.xlsx",
		"ads_scraping (2).xlsx",
		"ads_scraping_LP.xlsx",
	]

	candidate_paths = []
	for root in roots:
		candidate_paths.extend([os.path.join(root, "ads", fname) for fname in candidate_filenames])

	# Deduplicate while preserving order
	seen = set()
	candidate_paths = [p for p in candidate_paths if not (p in seen or seen.add(p))]

	path = next((p for p in candidate_paths if os.path.exists(p)), None)
	if path is None:
		st.warning("Ads data file not found in expected locations.")
		return None

	try:
		df = pd.read_excel(path, sheet_name=0)
	except Exception as e:
		st.error(f"[Ads] Error loading ads data: {e}")
		return None

	# Normalize dates
	for col in ["startDateFormatted", "endDateFormatted"]:
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)

	# Normalize numeric reach
	reach_col = "ad_details/aaa_info/eu_total_reach"
	if reach_col in df.columns:
		df["reach"] = pd.to_numeric(df[reach_col], errors="coerce")
	else:
		# Fallbacks commonly seen in exports
		for alt in ["reach", "estimated_audience_size", "eu_total_reach"]:
			if alt in df.columns:
				df["reach"] = pd.to_numeric(df[alt], errors="coerce")
				break
		else:
			df["reach"] = 0

	# Brand and flags
	if "pageName" in df.columns:
		df["brand"] = df["pageName"]
	elif "page_name" in df.columns:
		df["brand"] = df["page_name"]
	# Normalize brand casing to match configured names
	if "brand" in df.columns:
		brand_lookup = {k.lower(): v for k, v in BRAND_NAME_MAPPING.items()}
		df["brand"] = df["brand"].astype(str).str.strip()
		df["brand"] = df["brand"].apply(lambda b: brand_lookup.get(b.lower(), b))
	if "isActive" in df.columns:
		df["isActive"] = df["isActive"].astype(bool)

	# Duration
	if "startDateFormatted" in df.columns and "endDateFormatted" in df.columns:
		df["duration_days"] = (df["endDateFormatted"] - df["startDateFormatted"]).dt.days

	return df

# ------------------------
# 🎯 Load Audience Affinity outputs (pickled)
# ------------------------

@st.cache_data
def load_audience_affinity_outputs():
	path = os.path.join(DATA_ROOT, "audience_affinity", "audience_affinity_outputs.pkl")
	if not os.path.exists(path):
		st.warning("Audience affinity outputs not found.")
		return None
	try:
		import pickle
		with open(path, 'rb') as f:
			return pickle.load(f)
	except Exception as e:
		st.error(f"[Audience Affinity] Error loading outputs: {e}")
		return None

# ------------------------
# 🧱 Load Content Pillars outputs (pickled)
# ------------------------

@st.cache_data
def load_content_pillar_outputs():
	path = os.path.join(DATA_ROOT, "content_pillars", "content_pillar_outputs.pkl")
	if not os.path.exists(path):
		st.warning("Content pillar outputs not found.")
		return None
	try:
		import pickle
		with open(path, 'rb') as f:
			return pickle.load(f)
	except Exception as e:
		st.error(f"[Content Pillars] Error loading outputs: {e}")
		return None
