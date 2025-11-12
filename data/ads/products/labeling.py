import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------
# CONFIG (defaults; can be overridden by .env)
# ----------------------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_WORKERS = 50
MAX_CHARS_PER_AD = 1400

# Input/output defaults
INPUT_PATH = "ads.xlsx"  # change or pass via CLI if you prefer
OUTPUT_PREFIX = "ads_labeled"

# Column names in the FB ads dump
COL_TEXT = "snapshot/body/text"
COL_SUMMARY = "ad_summary"
COL_CLUSTER_1 = "cluster_1"
COL_CLUSTER_2 = "cluster_2"
COL_CLUSTER_3 = "cluster_3"

# ----------------------------------------------------
# PROMPTS
# ----------------------------------------------------
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

CLUSTER_SYSTEM_PROMPT = (
    "You are labeling a product/promotion one-liner against a FIXED taxonomy.\n"
    "Rules:\n"
    "- Choose 1 to 3 labels from ALLOWED THEMES (listed below).\n"
    "- The FIRST label must be the single MOST APPROPRIATE cluster.\n"
    "- If no cluster fits, output OTHER.\n"
    "- Output ENGLISH only in EXACTLY this format:\n"
    "Labels: <Theme A>; <Theme B>; <Theme C>\n"
    "- Use 1–3 labels; separate with semicolons; do not number them.\n"
    "- Return cluster name only (text before the dash).\n\n"
    "Available clusters (return only the name before the dash):\n"
    "1. Black Friday—Members/Signup Gate — Black Friday campaigns requiring membership or registration for access.\n"
    "2. Black Friday—Giveaway Tie-in — Black Friday campaigns linked to prize draws or gift-card giveaways.\n"
    "3. Black Friday—Early Access — Pre-sale or exclusive early Black Friday access messaging.\n"
    "4. Black Friday—Outlet — Black Friday promotions tied to physical outlets.\n"
    "5. Black Friday—Sitewide % Off — Broad Black Friday percentage-off discounts.\n"
    "6. Drops—Lifestyle — New lifestyle/fashion product drops (“latest drop”, “new collection”).\n"
    "7. Drops—Performance/Sport — New performance gear drops (running, training, football boots, etc.).\n"
    "8. Drops—Available Now — ‘Available now’ / ‘out now’ availability announcements.\n"
    "9. Brand/Ambassador—Lifestyle Slogans — Creator/ambassador posts with lifestyle taglines, mood/vibe copy, #ad.\n"
    "10. Brand/Ambassador—Performance Motivation — Creator/athlete narratives about progress, grit, mindset, #ad.\n"
    "11. Hashtag Campaign—Nike Shox/Team Nike — Posts centered on #NikeShox / #TeamNike hashtag campaigns.\n"
    "12. Style—Elevate Your Look — Generic style-forward claims.\n"
    "13. Fragrance—Eau de Parfum Collection — Fragrance collection promotion.\n"
    "14. Fragrance—Body & Hair Mist — Body/Hair mist freshness/boost messaging.\n"
    "15. Engage—Signup/Early Access (Non-seasonal) — Membership/registration pushes not tied to Black Friday.\n"
    "16. Giveaway—Generic — Prize draws/gift-card giveaways not tied to a specific season.\n"
    "17. Outlet—Evergreen/Local — Year-round or location-specific outlet/store promotions.\n"
    "18. Value—Free Shipping/Returns — Free delivery/returns policy value props.\n"
    "19. OTHER — Copy without a discernible promotional or product message.\n"
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

def parse_label_line(text: str) -> List[str]:
    if not isinstance(text, str):
        return [None, None, None]
    m = re.search(r"Labels\s*:\s*(.+)$", text.strip(), flags=re.IGNORECASE)
    if not m:
        return [None, None, None]
    parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
    parts = parts[:3]
    while len(parts) < 3:
        parts.append(None)
    return parts

def build_summary_prompt(ad_text: str) -> str:
    return f"Ad text:\n{ad_text}"

def build_cluster_prompt(ad_text: str) -> str:
    return f"Item:\n{ad_text}\n\nChoose 1–3 from ALLOWED THEMES."

# ----------------------------------------------------
# OpenAI calls
# ----------------------------------------------------
def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

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

def generate_clusters(client: OpenAI, model: str, ad_text: str) -> Tuple[str, str, str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {"role": "user", "content": build_cluster_prompt(ad_text)},
            ],
            temperature=TEMPERATURE,
            max_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
        c1, c2, c3 = parse_label_line(raw)
        return c1 or "NONE", c2, c3
    except Exception as e:
        print(f"[ERROR] Clusters failed: {e}")
        return "NONE", None, None

def process_single_ad(client: OpenAI, model: str, ad_text: str) -> Tuple[str, str, str, str]:
    summary = generate_summary(client, model, ad_text)
    c1, c2, c3 = generate_clusters(client, model, ad_text)
    return summary, c1, c2, c3

# ----------------------------------------------------
# Main labeling logic
# ----------------------------------------------------
def label_dataframe(df: pd.DataFrame, api_key: str, model: str, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
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

    # We create ONE client per thread task inside, so we'll just pass key/model
    summaries = [None] * len(texts)
    cl1 = [None] * len(texts)
    cl2 = [None] * len(texts)
    cl3 = [None] * len(texts)

    def task(text):
        # create a lightweight client per call to avoid cross-thread issues
        local_client = make_client(api_key)
        return process_single_ad(local_client, model, text)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, t): i for i, t in enumerate(texts)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Labeling ads"):
            i = futures[fut]
            try:
                s, a, b, c = fut.result()
                summaries[i] = s
                cl1[i] = a
                cl2[i] = b
                cl3[i] = c
            except Exception as e:
                print(f"[ERROR] Future failed on row {i}: {e}")
                summaries[i] = "NONE"
                cl1[i] = "NONE"
                cl2[i] = None
                cl3[i] = None

    df[COL_SUMMARY] = summaries
    df[COL_CLUSTER_1] = cl1
    df[COL_CLUSTER_2] = cl2
    df[COL_CLUSTER_3] = cl3

    return df

# ----------------------------------------------------
# I/O
# ----------------------------------------------------
def read_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Input must be .xlsx, .xls or .csv")

def write_output(df: pd.DataFrame, prefix: str):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    xlsx_path = f"{prefix}_{ts}.xlsx"
    csv_path = f"{prefix}_{ts}.csv"
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
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_PATH}")
    df_labeled = label_dataframe(df, api_key, model, MAX_WORKERS)
    write_output(df_labeled, OUTPUT_PREFIX)

if __name__ == "__main__":
    main()
