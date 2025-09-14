# preprocess_to_scores.py
# Converts Qualtrics-style cleaned export (CLEANED_FINAL.csv) into a tidy scores file
# expected by the analysis script (Appendix_D_analysis.py).
#
# Output: CLEANED_FINAL_SCORES.csv with columns:
#   Firm (text), FirmA (anonymized A–E), Role, Dept,
#   Tech, Org, Env, Trust, Adopt_mean, (optionally Q9/Q10/Q11 open-ended)
#
# Usage:
#   py preprocess_to_scores.py

import pandas as pd
import numpy as np

IN_PATH  = "CLEANED_FINAL.csv"           # your frozen file (after exclusions; n≈28)
OUT_PATH = "CLEANED_FINAL_SCORES.csv"    # output with computed construct scores

# ---------------------------
# 1) Load and strip Qualtrics header rows
# ---------------------------
# Read as strings to avoid mixed-type surprises
df_raw = pd.read_csv(IN_PATH, dtype=str)

def looks_like_qualtrics_header(r0: pd.Series, r1: pd.Series) -> bool:
    """Heuristic: first row has column labels (e.g., 'StartDate'), second row often JSON or long text."""
    # Common cues: first row has "Start Date"/"StartDate"; second row contains JSON-like content
    r0_keys = {k.lower().replace(" ", "") for k in r0.index}
    startdate_in_header = "startdate" in r0_keys or "start date" in r0.index
    r1_values = " ".join([str(v) for v in r1.values])[:200]
    looks_jsonish = "{" in r1_values and "}" in r1_values
    return startdate_in_header or looks_jsonish

if df_raw.shape[0] >= 2 and looks_like_qualtrics_header(df_raw.iloc[0], df_raw.iloc[1]):
    df = df_raw.iloc[2:].copy()
else:
    df = df_raw.copy()

df = df.reset_index(drop=True)

# ---------------------------
# 2) Helper: convert Likert strings to numeric
# ---------------------------
def to_num(series):
    """Convert a pandas Series of possible strings ('1'..'5', text) to numeric; invalid -> NaN."""
    return pd.to_numeric(series, errors="coerce")

# ---------------------------
# 3) Rename demographics (from your survey structure)
# Q1 = Firm, Q2 = Role, Q3 = Dept
# ---------------------------
rename_demo = {"Q1": "Firm", "Q2": "Role", "Q3": "Dept"}
for old, new in rename_demo.items():
    if old in df.columns:
        df[new] = df[old].astype(str).str.strip()

# ---------------------------
# 4) Decode numeric codes -> text labels
#    (Adjust mappings if your Qualtrics option order differs)
# ---------------------------
# Firm (Q1): PwC, Deloitte, EY, KPMG, Other  (as per your questionnaire)
firm_code_map = {
    "1": "PwC",
    "2": "Deloitte",
    "3": "EY",
    "4": "KPMG",
    "5": "Other"
}
if "Firm" in df.columns:
    df["Firm"] = df["Firm"].astype(str).str.strip().map(firm_code_map).fillna(df["Firm"])

# Role (Q2): Partner/Director, Senior Manager/Manager, Senior Associate/Consultant, Associate/Junior, Working Student
role_code_map = {
    "1": "Partner/Director",
    "2": "Senior Manager/Manager",
    "3": "Senior Associate/Consultant",
    "4": "Associate/Junior",
    "5": "Working Student"
}
if "Role" in df.columns:
    df["Role"] = df["Role"].astype(str).str.strip().map(role_code_map).fillna(df["Role"])

# Dept (Q3): CIT, VAT, TCR, TaxTech, Other
dept_code_map = {
    "1": "CIT",
    "2": "VAT",
    "3": "TCR",
    "4": "TaxTech",
    "5": "Other"
}
if "Dept" in df.columns:
    df["Dept"] = df["Dept"].astype(str).str.strip().map(dept_code_map).fillna(df["Dept"])

# Optional: Normalize common free-text variants (defensive)
norm_dept = {
    "corporate income tax": "CIT",
    "indirect tax": "VAT",
    "transfer pricing": "TCR",
    "tax technology": "TaxTech",
    "tax tech": "TaxTech",
    "taxtech": "TaxTech",
}
if "Dept" in df.columns:
    df["Dept"] = (df["Dept"].astype(str)
                  .apply(lambda s: norm_dept.get(s.strip().lower(), s.strip())))

# Optional: Normalize role variants (defensive)
norm_role = {
    "partner": "Partner/Director",
    "director": "Partner/Director",
    "senior manager": "Senior Manager/Manager",
    "manager": "Senior Manager/Manager",
    "senior associate": "Senior Associate/Consultant",
    "consultant": "Senior Associate/Consultant",
    "associate": "Associate/Junior",
    "junior": "Associate/Junior",
    "working student": "Working Student",
}
if "Role" in df.columns:
    df["Role"] = (df["Role"].astype(str)
                  .apply(lambda s: norm_role.get(s.strip().lower(), s.strip())))

# ---------------------------
# 5) Define construct item sets (based on your export)
# ---------------------------
TECH_ITEMS = [c for c in ["Q5_1","Q5_2","Q5_3","Q5_4"] if c in df.columns]
ORG_ITEMS  = [c for c in ["Q6_1","Q6_2","Q6_3","Q6_4","Q6_5"] if c in df.columns]
ENV_ITEMS  = [c for c in ["Q7_1","Q7_2","Q7_3","Q7_4"] if c in df.columns]
ADOPT_ITEMS= [c for c in ["Q8_1","Q8_2","Q8_3","Q8_4"] if c in df.columns]
TRUST_ITEM = "Q4_1" if "Q4_1" in df.columns else None  # single-item trust in your file

# Convert items to numeric
for cols in [TECH_ITEMS, ORG_ITEMS, ENV_ITEMS, ADOPT_ITEMS]:
    for c in cols:
        df[c] = to_num(df[c])

if TRUST_ITEM:
    df[TRUST_ITEM] = to_num(df[TRUST_ITEM])

# ---------------------------
# 6) Compute construct means
# ---------------------------
if TECH_ITEMS:
    df["Tech"] = df[TECH_ITEMS].mean(axis=1, skipna=True)
if ORG_ITEMS:
    df["Org"] = df[ORG_ITEMS].mean(axis=1, skipna=True)
if ENV_ITEMS:
    df["Env"] = df[ENV_ITEMS].mean(axis=1, skipna=True)
if ADOPT_ITEMS:
    df["Adopt_mean"] = df[ADOPT_ITEMS].mean(axis=1, skipna=True)
if TRUST_ITEM:
    df["Trust"] = df[TRUST_ITEM]

# ---------------------------
# 7) Firm anonymization (Firm -> FirmA) using YOUR mapping
#     Deloitte → Firm C
#     EY       → Firm A
#     KPMG     → Firm D
#     PwC      → Firm B
#     Other    → Firm E
# ---------------------------
if "Firm" in df.columns:
    firm_map = {
        "Deloitte": "Firm C",
        "EY":       "Firm A",
        "KPMG":     "Firm D",
        "PwC":      "Firm B",
        "Other":    "Firm E"
    }
    df["FirmA"] = df["Firm"].map(firm_map).fillna("Firm E")

# ---------------------------
# 8) Keep only what’s needed for analysis (+ open text if you want)
# ---------------------------
keep_cols = []
for c in ["Firm","FirmA","Role","Dept","Tech","Org","Env","Trust","Adopt_mean",
          "Q9","Q10","Q11"]:  # keep open-ended for later qualitative summary if present
    if c in df.columns:
        keep_cols.append(c)

df_out = df[keep_cols].copy()

# ---------------------------
# 9) Save
# ---------------------------
df_out.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH} with shape {df_out.shape}")
print("Columns:", list(df_out.columns))
