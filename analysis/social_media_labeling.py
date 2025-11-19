import os
import re
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------
# CONFIG (defaults; can be overridden by .env)
# ----------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_WORKERS = 100
MAX_CHARS_PER_AD = 1400

# Feature flags - set to True/False to enable/disable each step
ENABLE_EMPLOYER_BRANDING_LABEL = True
ENABLE_SUMMARY = True
ENABLE_THEMES = True

# Input/output defaults
# Paths relative to script location
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback if __file__ is not available (e.g., in some REPL contexts)
    SCRIPT_DIR = Path.cwd() / "analysis"

INPUT_PATH = SCRIPT_DIR.parent / "data" / "social_media" / "linkedin_posts.xlsx"
OUTPUT_PREFIX = SCRIPT_DIR.parent / "data" / "social_media" / "linkedin_posts_labeled"

# Column names
COL_TEXT = "post_text"
COL_EMPLOYER_BRANDING = "employer_branding"
COL_SUMMARY = "post_summary"
COL_THEME_1 = "theme_1"
COL_THEME_2 = "theme_2"
COL_THEME_3 = "theme_3"

# ----------------------------------------------------
# PROMPTS
# ----------------------------------------------------
EMPLOYER_BRANDING_SYSTEM_PROMPT = (
    "You are a classifier for LinkedIn posts.\n"
    "Determine if a LinkedIn post is about employer branding or related to showing what it's like to work at the company.\n"
    "Label as 1 (employer branding) if the post:\n"
    "- Contains job ads or job openings\n"
    "- Shows company culture, work environment, or employee experiences\n"
    "- Promotes the company as a great place to work\n"
    "- Features employee testimonials, team highlights, or workplace benefits\n"
    "- Recruits talent or encourages applications\n"
    "- Shows behind-the-scenes of working at the company\n"
    "Label as 0 (not employer branding) if the post:\n"
    "- Is about products, services, or promotions\n"
    "- Is general brand marketing without employment focus\n"
    "- Is unrelated to employment or workplace culture\n"
    "Return STRICT JSON ONLY as: {\"employer_branding\": 0 or 1}"
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a precise annotator of ad copy.\n"
    "Given an ad's text, return a ONE-SENTENCE description of the clear product/service/promotion being advertised.\n"
    "Rules:\n"
    "- If a clear single product/service/promotion/venue/event is identifiable, describe it succinctly in one sentence.\n"
    "- If the ad is only brand building, employer branding, atmosphere, or ambiguous with no concrete offer, still summarize the ad in one sentence.\n"
    "- Keep it factual (no hype), <= 140 characters where feasible, no emojis, no hashtags, no URLs.\n"
    "- Treat promotions/discount weekends/contests as valid 'products'.\n"
    "- ALWAYS return everything in English.\n"
    'Return STRICT JSON ONLY as: {"summary":"<ONE_SENTENCE_OR_NONE>"}'
)

THEME_SYSTEM_PROMPT = (
    "You are labeling LinkedIn posts against a FIXED taxonomy of employer branding themes.\n"
    "Rules:\n"
    "- Choose 1 to 3 themes from ALLOWED THEMES (listed below) that best describe the post. Don't force yourself to choose more than 1 theme if it does not fit any of the other themes.\n"
    "- The FIRST theme must be the single MOST APPROPRIATE theme.\n"
    "- If no theme fits, output OTHER.\n"
    "- Output ENGLISH only in EXACTLY this format:\n"
    "Themes: <Theme A>; <Theme B>; <Theme C>\n"
    "- Use 1–3 themes; separate with semicolons; do not number them.\n"
    "- Return the exact theme name as listed below.\n\n"
    "Available themes:\n"
    "1. Employee stories & career journeys\n"
    "2. Learning, development & leadership growth\n"
    "3. Innovation, technology & digitalisation\n"
    "4. Purpose, impact & sustainability\n"
    "5. Health, safety & wellbeing\n"
    "6. Diversity, inclusion & equal opportunity\n"
    "7. Students, graduates & early careers\n"
    "8. Recruitment & career opportunities\n"
    "9. Company culture, community & team spirit\n"
    "10. Partnerships, academia & social initiatives\n"
    "11. Recognition, awards & employer reputation\n"
    "12. Work environment & ways of working\n"
    "13. Governance, transparency & employer facts\n"
    "14. OTHER — Post does not fit any of the above themes\n"
)



# ----------------------------------------------------
# Small helpers
# ----------------------------------------------------
def load_env():
    """Load environment vars from .env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Put it in a .env file.")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return api_key, model

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def compact_text(s: str, limit: int = MAX_CHARS_PER_AD) -> str:
    s = normalize_text(s)
    return s if len(s) <= limit else (s[:limit] + "…")

def build_summary_prompt(ad_text: str) -> str:
    return f"Ad text:\n{ad_text}"

def build_employer_branding_prompt(post_text: str) -> str:
    return f"LinkedIn post:\n{post_text}"

def build_theme_prompt(post_text: str) -> str:
    return f"LinkedIn post:\n{post_text}\n\nChoose 1–3 themes from ALLOWED THEMES."

def parse_theme_line(text: str) -> List[Optional[str]]:
    """Parse theme labels from response text."""
    if not isinstance(text, str):
        return [None, None, None]
    m = re.search(r"Themes\s*:\s*(.+)$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return [None, None, None]
    parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
    parts = parts[:3]
    while len(parts) < 3:
        parts.append(None)
    return parts

# ----------------------------------------------------
# OpenAI calls
# ----------------------------------------------------
def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def generate_employer_branding_label(client: OpenAI, model: str, post_text: str) -> int:
    """Generate employer branding label: 1 if about employer branding, 0 otherwise."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EMPLOYER_BRANDING_SYSTEM_PROMPT},
                {"role": "user", "content": build_employer_branding_prompt(post_text)},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=50,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw.startswith("{") else json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        label = data.get("employer_branding")
        # Ensure it's 0 or 1
        if label in [0, 1]:
            return int(label)
        elif isinstance(label, bool):
            return 1 if label else 0
        else:
            # Try to parse string representations
            label_str = str(label).strip().lower()
            if label_str in ["1", "true", "yes"]:
                return 1
            else:
                return 0
    except Exception as e:
        print(f"[ERROR] Employer branding label failed: {e}")
        return 0

def generate_summary(client: OpenAI, model: str, ad_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": build_summary_prompt(ad_text)},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw.startswith("{") else json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        summary = (data.get("summary") or "").strip()
        if not summary:
            return "NONE"
        summary = re.sub(r'https?://\S+', '', summary)
        summary = normalize_text(summary)
        if len(summary) > 160:
            summary = summary[:160].rstrip(" ,.;:") + "."
        return summary
    except Exception as e:
        print(f"[ERROR] Summary failed: {e}")
        return "NONE"

def generate_themes(client: OpenAI, model: str, post_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate theme labels: up to 3 themes per post."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": THEME_SYSTEM_PROMPT},
                {"role": "user", "content": build_theme_prompt(post_text)},
            ],
            temperature=TEMPERATURE,
            max_tokens=150,
        )
        raw = (resp.choices[0].message.content or "").strip()
        t1, t2, t3 = parse_theme_line(raw)
        return t1, t2, t3
    except Exception as e:
        print(f"[ERROR] Themes failed: {e}")
        return None, None, None

def process_single_ad(
    client: OpenAI, 
    model: str, 
    ad_text: str,
    enable_employer_branding: bool = ENABLE_EMPLOYER_BRANDING_LABEL,
    enable_summary: bool = ENABLE_SUMMARY,
    enable_themes: bool = ENABLE_THEMES
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Process a single post and generate labels based on enabled features.
    Themes are only generated if employer_branding == 1.
    Returns: (employer_branding, summary, theme_1, theme_2, theme_3)
    """
    employer_branding = None
    summary = None
    theme_1, theme_2, theme_3 = None, None, None
    
    # If themes are enabled, we need to check employer branding first
    # (even if enable_employer_branding is False, we need it internally for theme filtering)
    if enable_employer_branding or enable_themes:
        employer_branding = generate_employer_branding_label(client, model, ad_text)
    
    if enable_summary:
        summary = generate_summary(client, model, ad_text)
    
    # Only generate themes if employer_branding == 1
    if enable_themes:
        if employer_branding == 1:
            theme_1, theme_2, theme_3 = generate_themes(client, model, ad_text)
        else:
            # If employer_branding is 0 or None, set themes to None
            theme_1, theme_2, theme_3 = None, None, None
    
    return employer_branding, summary, theme_1, theme_2, theme_3

# ----------------------------------------------------
# Main labeling logic
# ----------------------------------------------------
def label_dataframe(
    df: pd.DataFrame, 
    api_key: str, 
    model: str, 
    max_workers: int = MAX_WORKERS,
    enable_employer_branding: bool = ENABLE_EMPLOYER_BRANDING_LABEL,
    enable_summary: bool = ENABLE_SUMMARY,
    enable_themes: bool = ENABLE_THEMES
) -> pd.DataFrame:
    if COL_TEXT not in df.columns:
        raise ValueError(f"Input file is missing required column: {COL_TEXT}")

    # Keep only non-empty texts
    df = df.copy()
    df[COL_TEXT] = df[COL_TEXT].map(lambda x: compact_text(x if isinstance(x, str) else ""))

    df = df[df[COL_TEXT].str.len() > 0].reset_index(drop=True)
    if df.empty:
        print("[INFO] No rows with non-empty ad text.")
        return df

    texts = df[COL_TEXT].tolist()

    # Initialize result lists based on enabled features
    # Note: If themes are enabled, we need employer branding internally (even if not output)
    needs_employer_branding = enable_employer_branding or enable_themes
    employer_branding_labels = [None] * len(texts) if needs_employer_branding else None
    summaries = [None] * len(texts) if enable_summary else None
    theme_1_list = [None] * len(texts) if enable_themes else None
    theme_2_list = [None] * len(texts) if enable_themes else None
    theme_3_list = [None] * len(texts) if enable_themes else None

    def task(text):
        # create a lightweight client per call to avoid cross-thread issues
        local_client = make_client(api_key)
        return process_single_ad(
            local_client, 
            model, 
            text,
            enable_employer_branding=enable_employer_branding,
            enable_summary=enable_summary,
            enable_themes=enable_themes
        )

    desc = "Labeling posts"
    if enable_employer_branding:
        desc += " [EB]"
    elif enable_themes:
        desc += " [EB check for themes]"
    if enable_summary:
        desc += " [Summary]"
    if enable_themes:
        desc += " [Themes (EB=1 only)]"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, t): i for i, t in enumerate(texts)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[fut]
            try:
                eb, s, t1, t2, t3 = fut.result()
                # Store employer branding if enabled OR if themes are enabled (needed internally)
                if needs_employer_branding:
                    employer_branding_labels[i] = eb
                if enable_summary:
                    summaries[i] = s
                if enable_themes:
                    # Themes are only set if employer_branding == 1 (handled in process_single_ad)
                    theme_1_list[i] = t1
                    theme_2_list[i] = t2
                    theme_3_list[i] = t3
            except Exception as e:
                print(f"[ERROR] Future failed on row {i}: {e}")
                if needs_employer_branding:
                    employer_branding_labels[i] = 0
                if enable_summary:
                    summaries[i] = "NONE"
                if enable_themes:
                    theme_1_list[i] = None
                    theme_2_list[i] = None
                    theme_3_list[i] = None

    # Add columns based on enabled features
    if enable_employer_branding:
        df[COL_EMPLOYER_BRANDING] = employer_branding_labels
    if enable_summary:
        df[COL_SUMMARY] = summaries
    if enable_themes:
        df[COL_THEME_1] = theme_1_list
        df[COL_THEME_2] = theme_2_list
        df[COL_THEME_3] = theme_3_list

    return df

# ----------------------------------------------------
# I/O
# ----------------------------------------------------
def read_input(path) -> pd.DataFrame:
    """Read input file. Accepts str or Path."""
    p = Path(path) if isinstance(path, str) else path
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p.absolute()}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Input must be .xlsx, .xls or .csv")

def write_output(df: pd.DataFrame, prefix):
    """Write output files. Prefix can be str or Path (will add timestamp and extensions)."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix_path = Path(prefix) if isinstance(prefix, str) else prefix
    # Remove any existing extension and add timestamp
    xlsx_path = prefix_path.parent / f"{prefix_path.name}_{ts}.xlsx"
    csv_path = prefix_path.parent / f"{prefix_path.name}_{ts}.csv"
    
    # Ensure output directory exists
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="LabeledAds")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[DONE] Wrote: {xlsx_path}")
    print(f"[DONE] Wrote: {csv_path}")

# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
def main():
    api_key, model = load_env()
    df = read_input(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_PATH.absolute()}")
    print(f"[INFO] Features enabled:")
    print(f"  - Employer branding label: {ENABLE_EMPLOYER_BRANDING_LABEL}")
    print(f"  - Summary: {ENABLE_SUMMARY}")
    print(f"  - Themes: {ENABLE_THEMES}")
    df_labeled = label_dataframe(
        df, 
        api_key, 
        model, 
        MAX_WORKERS,
        enable_employer_branding=ENABLE_EMPLOYER_BRANDING_LABEL,
        enable_summary=ENABLE_SUMMARY,
        enable_themes=ENABLE_THEMES
    )
    write_output(df_labeled, OUTPUT_PREFIX)

if __name__ == "__main__":
    main()
