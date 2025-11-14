import os
import re
import math
import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIGURATION (EDIT THESE)
# ============================================================

# Path relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "agility"))  # folder where files are stored
FILES = [
    "epso-g_agility.xlsx",
    "exergi_agility.xlsx",
    "helen_agility.xlsx",
    "hofor_agility.xlsx",
    "ignitis_agility.xlsx",
    "kauno-energija_agility.xlsx",
    "ltg_agility.xlsx",
    "ltou_agility.xlsx",
    "teltonika_agility.xlsx",
    "via-lietuva_agility.xlsx"
]  # filenames inside DATA_FOLDER
BRANDS = [
    "epso",
    "exergi",
    "helen",
    "hofor",
    "ignitis",
    "energi",
    "ltg",
    "ltou",
    "teltonika",
    "lietuva"
]  # brand per file, same order
SHEET = "Raw Data"                    # sheet name for Excel files

# ============================================================
# SUPPORT FUNCTIONS (SELF-CONTAINED)
# ============================================================

def load_file(path, sheet):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            print(f"    ⚠ Error reading Excel file: {e}")
            print(f"    Available sheets (if any):")
            try:
                xl_file = pd.ExcelFile(path)
                print(f"      {xl_file.sheet_names}")
            except:
                pass
            raise
    raise ValueError(f"Unsupported file format: {path}")


def count_query_occurrences(df, query):
    occurrences = []
    q = query.lower()
    for _, row in df.iterrows():
        c = 0
        if pd.notna(row.get("Headline")):
            c += row["Headline"].lower().count(q)
        if pd.notna(row.get("Maintext")):
            c += row["Maintext"].lower().count(q)
        occurrences.append(c)
    return pd.Series(occurrences)


def check_text_presence_100(df, brand):
    df["term_in_truncated_maintext_100"] = False
    for idx, row in df.iterrows():
        text = row.get("Maintext")
        if pd.notna(text):
            truncated = " ".join(text.split()[:100])
            if brand.lower() in truncated.lower():
                df.at[idx, "term_in_truncated_maintext_100"] = True
    return df


def check_text_presence_200(df, brand):
    df["term_in_truncated_maintext_200"] = False
    for idx, row in df.iterrows():
        text = row.get("Maintext")
        if pd.notna(text):
            truncated = " ".join(text.split()[:200])
            if brand.lower() in truncated.lower():
                df.at[idx, "term_in_truncated_maintext_200"] = True
    return df


def check_title(df, brand):
    df["term_in_title"] = df["Headline"].str.contains(brand, case=False, na=False)
    return df


def truncate_link(link):
    try:
        m = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", link)
        if m:
            return m.group(1)
        return link
    except:
        return link


def get_page_rank(domain, api_key):
    try:
        r = requests.get(
            "https://openpagerank.com/api/v1.0/getPageRank",
            params={"domains[]": domain},
            headers={"API-OPR": api_key},
            timeout=10
        )
        return r.json()["response"][0]
    except:
        return None


def calculate_log_score(rank, max_rank=100):
    if rank <= 0:
        return 0
    return round(100 * (1 - (math.log(rank + 1) / math.log(max_rank + 1))), 2)


def log_scale_clipped_lower_better(v, a, b):
    x = np.clip(v, a, b)
    return 1 - (np.log(x) - np.log(a)) / (np.log(b) - np.log(a))


def log_scale_clipped_higher_better(v, a, b):
    x = np.clip(v, a, b)
    return (np.log(x) - np.log(a)) / (np.log(b) - np.log(a))


def linear_scale_clipped(v, a, b):
    x = np.clip(v, a, b)
    return (x - a) / (b - a)


def calculate_article_score(P, R, n, Q):
    scaled_P = log_scale_clipped_lower_better(P, 1, 10**8)
    scaled_R = log_scale_clipped_lower_better(R, 1, 100)
    scaled_n = log_scale_clipped_higher_better(n, 1, 30)
    scaled_Q = linear_scale_clipped(Q, 1, 3)

    return 0.25 * scaled_P + 0.25 * scaled_R + 0.25 * scaled_n + 0.25 * scaled_Q


def assign_quality_score(df):
    df["Quality_Score"] = 0
    df.loc[df["term_in_title"] == True, "Quality_Score"] = 3
    df.loc[(df["Quality_Score"] == 0) & (df["term_in_truncated_maintext_100"] == True), "Quality_Score"] = 2
    df.loc[(df["Quality_Score"] == 0) & (df["term_in_truncated_maintext_200"] == True), "Quality_Score"] = 1
    return df

# ============================================================
# QUALITY + RANKING PROCESS
# ============================================================

def process_file(df, brand):
    df = df.copy()
    print(f"    Starting process_file for {len(df)} rows...")

    # --- Query occurrences ---
    print(f"    Counting query occurrences for '{brand}'...")
    if "," in brand:
        p, s = [x.strip() for x in brand.split(",")]
        df["query_occurrences"] = count_query_occurrences(df, p) + count_query_occurrences(df, s)
        print(f"    Using multi-term search: '{p}' + '{s}'")
    else:
        df["query_occurrences"] = count_query_occurrences(df, brand)
    print(f"    Query occurrences range: {df['query_occurrences'].min()} - {df['query_occurrences'].max()}")

    # --- Term checks ---
    print(f"    Checking term presence in text (100, 200 words) and title...")
    df = check_text_presence_100(df, brand)
    df = check_text_presence_200(df, brand)
    df = check_title(df, brand)
    print(f"    Term in title: {df['term_in_title'].sum()} rows")
    print(f"    Term in first 100 words: {df['term_in_truncated_maintext_100'].sum()} rows")
    print(f"    Term in first 200 words: {df['term_in_truncated_maintext_200'].sum()} rows")

    df["Quality"] = "A"
    df = df.reset_index(drop=True)

    # --- PageRank ---
    print(f"    Fetching PageRank for {len(df)} URLs (this may take a while)...")
    api_key = "wsc0sskwcow88448sswc88kg0okwc40so4k48cgk"
    ranks = []
    for i, link in enumerate(df["URL"], 1):
        if i % 10 == 0 or i == 1:
            print(f"      Progress: {i}/{len(df)} URLs processed...", end='\r')
        domain = truncate_link(link)
        pr = get_page_rank(domain, api_key)
        r = pr.get("rank") if pr else None
        ranks.append(int(r) if r is not None else 10_000_000_000)
    print(f"    ✓ PageRank complete: {len(ranks)} ranks fetched")
    df["Rank"] = ranks

    # --- Log Rank ---
    print(f"    Calculating Log_Rank...")
    df["Log_Rank"] = [calculate_log_score(i) / 100 for i in df.index]
    if not df.empty:
        df.at[0, "Log_Rank"] += 1

    df = assign_quality_score(df)
    print(f"    Quality scores assigned: {df['Quality_Score'].value_counts().to_dict()}")

    # --- BMQ Calculation for A Only ---
    df_A = df[df["Quality"] == "A"].copy()
    print(f"    Filtering Quality='A': {len(df_A)} rows remaining")
    print(f"    Calculating BMQ scores...")
    BMQ = []
    for idx in df_A.index:
        P = df_A.at[idx, "Rank"]
        R = df_A.at[idx, "Log_Rank"]
        n = df_A.at[idx, "query_occurrences"]
        Q = df_A.at[idx, "Quality_Score"]
        BMQ.append(calculate_article_score(P, R, n, Q))
    df_A["BMQ"] = BMQ
    print(f"    BMQ range: {min(BMQ):.4f} - {max(BMQ):.4f}")

    return df_A


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BMQ Calculations - Starting Processing")
    print("=" * 60)
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Number of files to process: {len(FILES)}")
    print(f"Sheet name: {SHEET}")
    print()
    
    if len(FILES) != len(BRANDS):
        raise ValueError("FILES and BRANDS must have the same length.")

    results = []  # Store tuples of (filename, processed_df)

    for idx, (filename, brand) in enumerate(zip(FILES, BRANDS), 1):
        print(f"\n[{idx}/{len(FILES)}] Processing: {filename} (Brand: {brand})")
        path = os.path.join(DATA_FOLDER, filename)
        print(f"  Full path: {path}")
        
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            continue
        
        print(f"  ✓ File found, loading...")
        try:
            df = load_file(path, SHEET)
            print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            continue
        
        print(f"  Processing data for brand '{brand}'...")
        try:
            processed = process_file(df, brand)
            print(f"  ✓ Processed: {len(processed)} rows after filtering (Quality='A')")
            results.append((filename, processed))
        except Exception as e:
            print(f"  ❌ Error processing file: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    if not results:
        print("❌ No data processed.")
        return

    # Save each processed file back to its original location (overwriting completely)
    print(f"Saving processed data back to original files (overwriting)...")
    processed_files = []
    
    for filename, processed_df in results:
        path = os.path.join(DATA_FOLDER, filename)
        
        print(f"\nSaving: {filename}")
        try:
            # Overwrite the entire file with processed data
            processed_df.to_excel(path, index=False)
            print(f"  ✓ Overwritten: {len(processed_df)} rows saved")
            processed_files.append(filename)
        except Exception as e:
            print(f"  ❌ Error saving {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully saved {len(processed_files)} files:")
    for f in processed_files:
        print(f"  - {f}")
    print("=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
