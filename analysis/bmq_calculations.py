import os
import re
import math
import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIGURATION (EDIT THESE)
# ============================================================

DATA_FOLDER = "../data/agility"                # folder where files are stored
FILES = ["file1.csv", "file2.xlsx"] # filenames inside DATA_FOLDER
BRANDS = ["nike", "adidas"]         # brand per file, same order
SHEET = "Raw Data"                    # sheet name for Excel files

# ============================================================
# SUPPORT FUNCTIONS (SELF-CONTAINED)
# ============================================================

def load_file(path, sheet):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported file format: {path}")


def count_query_occurrences(df, query):
    occurrences = []
    q = query.lower()
    for _, row in df.iterrows():
        c = 0
        if pd.notna(row.get("Title")):
            c += row["Title"].lower().count(q)
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
    df["term_in_title"] = df["Title"].str.contains(brand, case=False, na=False)
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

    # --- Query occurrences ---
    if "," in brand:
        p, s = [x.strip() for x in brand.split(",")]
        df["query_occurrences"] = count_query_occurrences(df, p) + count_query_occurrences(df, s)
    else:
        df["query_occurrences"] = count_query_occurrences(df, brand)

    # --- Term checks ---
    df = check_text_presence_100(df, brand)
    df = check_text_presence_200(df, brand)
    df = check_title(df, brand)

    df["Quality"] = "A"
    df = df.reset_index(drop=True)

    # --- PageRank ---
    api_key = "wsc0sskwcow88448sswc88kg0okwc40so4k48cgk"
    ranks = []
    for link in df["Link"]:
        pr = get_page_rank(truncate_link(link), api_key)
        r = pr.get("rank") if pr else None
        ranks.append(int(r) if r is not None else 10_000_000_000)
    df["Rank"] = ranks

    # --- Log Rank ---
    df["Log_Rank"] = [calculate_log_score(i) / 100 for i in df.index]
    if not df.empty:
        df.at[0, "Log_Rank"] += 1

    df = assign_quality_score(df)

    # --- BMQ Calculation for A Only ---
    df_A = df[df["Quality"] == "A"].copy()
    BMQ = []
    for idx in df_A.index:
        P = df_A.at[idx, "Rank"]
        R = df_A.at[idx, "Log_Rank"]
        n = df_A.at[idx, "query_occurrences"]
        Q = df_A.at[idx, "Quality_Score"]
        BMQ.append(calculate_article_score(P, R, n, Q))
    df_A["BMQ"] = BMQ

    return df_A


# ============================================================
# MAIN
# ============================================================

def main():
    if len(FILES) != len(BRANDS):
        raise ValueError("FILES and BRANDS must have the same length.")

    results = []

    for filename, brand in zip(FILES, BRANDS):
        path = os.path.join(DATA_FOLDER, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        df = load_file(path, SHEET)
        processed = process_file(df, brand)
        results.append(processed)

    if not results:
        print("No data processed.")
        return

    final_df = pd.concat(results, ignore_index=True)
    print("Processing complete.")
    print(final_df.head())

    final_df.to_excel("processed_output.xlsx", index=False)


if __name__ == "__main__":
    main()
