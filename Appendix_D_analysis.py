# Appendix D - Python Analysis Script
# Thesis: Adoption of Generative AI in Corporate Tax Functions (Big Four focus)
# Author: Natan Verhoeckx
# Environment: Python 3.10+
#
# Dependencies (install once):
#   pip install pandas numpy scipy statsmodels matplotlib
#
# Notes:
# - This script reproduces the analyses reported in the thesis:
#   data loading, cleaning filters, descriptives, correlations with p-values,
#   normality checks, ANOVA + Kruskal-Wallis with post-hoc tests,
#   and significance-annotated boxplots + correlation heatmap.
# - Figures are saved to ./figures/ by default.
# - All charts default to a white background.
#
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import shapiro, f_oneway, kruskal, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Ensure white background for all plots
plt.style.use("default")

# -----------------------------
# 0) CONFIGURATION
# -----------------------------
DATA_PATH = "CLEANED_FINAL_SCORES.csv"   # <-- set to your frozen dataset file
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Column names expected in the dataset (adjust to your file if needed)
CONSTRUCT_COLS = ["Tech", "Org", "Env", "Trust", "Adopt_mean"]
GROUP_COLS = {
    "Dept": "Dept",      # e.g., CIT, VAT, TCR, TaxTech
    "Role": "Role",      # e.g., Partner, Manager, Sr Associate, Associate, Working Student
    "Firm": "FirmA"      # anonymized firm labels (e.g., Firm A, Firm B, ...)
}

# -----------------------------
# 1) LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Optional: anonymize firm names (only if you have raw firm names and want to map them)
# firm_map = {"Deloitte":"Firm A","EY":"Firm B","KPMG":"Firm C","PwC":"Firm D"}
# if "Firm" in df.columns:
#     df["FirmA"] = df["Firm"].map(firm_map).fillna("Firm E")

# -----------------------------
# 2) CLEANING FILTERS (documented in Appendix C)
#    - This section demonstrates how cases would be classified for analysis.
#    - Actual exclusions used in the thesis were set prior to analysis.
# -----------------------------
def is_demographics_only(row, construct_cols=CONSTRUCT_COLS):
    # true if all construct columns are NaN for the row
    return pd.isna(row[construct_cols]).all()

def has_adoption(row, adopt_col="Adopt_mean"):
    val = row.get(adopt_col, np.nan)
    return pd.notna(val)

# Create flags (informational; the analytic dataset should already be frozen)
df["flag_demo_only"] = df.apply(is_demographics_only, axis=1)
df["flag_has_adoption"] = df.apply(has_adoption, axis=1)

# Analytic dataset used for hypothesis testing (example logic)
analytic_mask = (~df["flag_demo_only"]) & (df["flag_has_adoption"])
df_analytic = df.loc[analytic_mask].copy()

# --- ROLE NORMALIZATION & POOLING ---
# Map possible short forms to full labels and pool Working Student into Associate/Junior
role_map = {
    "Assoc": "Associate/Junior",
    "Associate": "Associate/Junior",
    "Junior": "Associate/Junior",
    "SrAssoc": "Senior Associate/Consultant",
    "Senior Associate": "Senior Associate/Consultant",
    "Consultant": "Senior Associate/Consultant",
    "SrMgr": "Senior Manager/Manager",
    "Senior Manager": "Senior Manager/Manager",
    "Manager": "Senior Manager/Manager",
    "Partner": "Partner/Director",
    "Director": "Partner/Director",
    "Working Student": "Associate/Junior",   # <- pool here
}

if "Role" in df_analytic.columns:
    df_analytic["Role"] = (
        df_analytic["Role"]
        .astype(str).str.strip()
        .replace(role_map)
    )

# Fixed colors for groups
COLOR_MAPS = {
    "Dept": {
        "CIT": "skyblue",
        "VAT": "orange",
        "TaxTech": "lightgreen",
        "TCR": "violet",
        "Other": "gray",
    },
    "Role": {
        "Associate/Junior": "violet",
        "Senior Associate/Consultant": "lightgreen",
        "Senior Manager/Manager": "orange",
        "Partner/Director": "skyblue",
    },
    # NOTE: key must be FirmA because GROUP_COLS["Firm"] == "FirmA"
    "FirmA": {
        "Firm A": "skyblue",
        "Firm B": "orange",
        "Firm C": "lightgreen",
        "Firm D": "violet",
        "Firm E": "gray",
    },
}

# Just after creating df_analytic (or before):
norm_map = {
    "tax technology": "TaxTech",
    "tax tech": "TaxTech",
    "taxtech": "TaxTech",
    "cit": "CIT",
    "corporate income tax": "CIT",
    "vat": "VAT",
    "indirect tax": "VAT",
    "tcr": "TCR",
    "trc": "TCR",
    "transfer pricing": "TCR",
}
for col in ["Dept"]:
    if col in df_analytic.columns:
        df_analytic[col] = (df_analytic[col]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .map(norm_map)
                            .fillna(df_analytic[col].astype(str).str.strip()))

# -----------------------------
# 3) DESCRIPTIVES
# -----------------------------
def descriptives_table(data, cols):
    out = pd.DataFrame({
        "n": data[cols].count(),
        "mean": data[cols].mean(),
        "sd": data[cols].std()
    })
    return out

desc_all = descriptives_table(df, CONSTRUCT_COLS)
desc_analytic = descriptives_table(df_analytic, CONSTRUCT_COLS)

# Save descriptives
desc_all.to_csv(os.path.join(FIG_DIR, "descriptives_all_cases.csv"))
desc_analytic.to_csv(os.path.join(FIG_DIR, "descriptives_analytic_only.csv"))

# -----------------------------
# 4) CORRELATIONS (Pearson r) WITH p-VALUES
# -----------------------------
def corr_with_p(data, cols):
    r = pd.DataFrame(np.nan, index=cols, columns=cols)
    p = pd.DataFrame(np.nan, index=cols, columns=cols)
    for a in cols:
        for b in cols:
            x = pd.to_numeric(data[a], errors="coerce")
            y = pd.to_numeric(data[b], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 3:
                rr, pp = pearsonr(x[mask], y[mask])
                r.loc[a, b] = rr
                p.loc[a, b] = pp
    stars = p.map(lambda v: "***" if (pd.notna(v) and v <= .001)
                       else ("**" if (pd.notna(v) and v <= .01)
                       else ("*" if (pd.notna(v) and v <= .05) else "")))
    return r, p, stars

rmat, pmat, stars = corr_with_p(df_analytic, CONSTRUCT_COLS)
rmat.round(3).to_csv(os.path.join(FIG_DIR, "correlations_r.csv"))
pmat.round(4).to_csv(os.path.join(FIG_DIR, "correlations_p.csv"))

# Heatmap with embedded r + stars
def heatmap_text(r, stars, title="Correlations (Pearson r with significance)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = r.values.astype(float)
    im = ax.imshow(vals, vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(r.columns))); ax.set_yticks(range(len(r.index)))
    ax.set_xticklabels(r.columns, rotation=45, ha="right"); ax.set_yticklabels(r.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            lab = "" if np.isnan(vals[i, j]) else f"{vals[i, j]:.2f}{stars.iloc[i, j]}"
            ax.text(j, i, lab, ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

fig, _ = heatmap_text(rmat, stars)
fig.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# 5) NORMALITY CHECKS (Shapiro–Wilk) BY GROUP
# -----------------------------
def normality_by_group(data, group_col, value_col, min_n=3):
    rows = []
    for g, sub in data.groupby(group_col):
        x = pd.to_numeric(sub[value_col], errors="coerce").dropna().values
        if len(x) >= min_n:
            W, p = shapiro(x)
            rows.append((g, len(x), W, p))
        else:
            rows.append((g, len(x), np.nan, np.nan))
    return pd.DataFrame(rows, columns=[group_col, "n", "Shapiro_W", "p_value"]).sort_values(group_col)

# Example: test normality of Adoption by Department
if GROUP_COLS.get("Dept") in df_analytic.columns:
    shapiro_dept = normality_by_group(df_analytic, GROUP_COLS["Dept"], "Adopt_mean")
    shapiro_dept.to_csv(os.path.join(FIG_DIR, "normality_shapiro_adopt_by_dept.csv"), index=False)

# -----------------------------
# 6) OMNIBUS TESTS: ANOVA + KRUSKAL-WALLIS
# -----------------------------
def omnibus_tests(data, group_col, value_col, min_groups=2):
    groups = [pd.to_numeric(v[value_col], errors="coerce").dropna().values
              for _, v in data.groupby(group_col)]
    labels = [g for g, _ in data.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < min_groups:
        return {"anova": None, "kruskal": None, "labels": labels}

    # ANOVA
    try:
        F, pA = f_oneway(*groups)
        anova = {"F": F, "p": pA, "k": len(groups), "N": sum(len(g) for g in groups)}
    except Exception:
        anova = None

    # Kruskal-Wallis
    try:
        H, pK = kruskal(*groups)
        kr = {"H": H, "p": pK, "k": len(groups), "N": sum(len(g) for g in groups)}
    except Exception:
        kr = None

    return {"anova": anova, "kruskal": kr, "labels": labels}

# Example omnibus tests for Adoption by Department / Role / Firm
for label, gcol in GROUP_COLS.items():
    if gcol in df_analytic.columns:
        res = omnibus_tests(df_analytic, gcol, "Adopt_mean")
        pd.DataFrame([res["anova"]]).to_csv(os.path.join(FIG_DIR, f"anova_adopt_by_{label.lower()}.csv"), index=False)
        pd.DataFrame([res["kruskal"]]).to_csv(os.path.join(FIG_DIR, f"kruskal_adopt_by_{label.lower()}.csv"), index=False)

# -----------------------------
# 7) POST-HOC TESTS
# -----------------------------
def tukey_posthoc(data, group_col, value_col):
    x = pd.to_numeric(data[value_col], errors="coerce")
    g = data[group_col].astype(str)
    mask = x.notna() & g.notna()
    if mask.sum() < 3:
        return None
    # Keep only groups with at least 2 observations
    tmp = pd.DataFrame({value_col: x[mask], group_col: g[mask]})
    counts = tmp[group_col].value_counts()
    valid_groups = counts[counts >= 2].index.tolist()
    tmp = tmp[tmp[group_col].isin(valid_groups)]
    if tmp[group_col].nunique() < 3:
        return None  # need >=3 groups for Tukey to be meaningful
    res = pairwise_tukeyhsd(endog=tmp[value_col].values,
                            groups=tmp[group_col].values,
                            alpha=0.05)
    ph = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
    return ph

def pairwise_mannwhitney(data, group_col, value_col, correction="holm"):
    from scipy.stats import mannwhitneyu
    # Build groups with at least 2 obs
    groups = {}
    for g, v in data.groupby(group_col):
        arr = pd.to_numeric(v[value_col], errors="coerce").dropna().values
        if len(arr) >= 2:
            groups[str(g)] = arr
    keys = list(groups.keys())
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i+1, len(keys))]
    rows, pvals = [], []
    for a, b in pairs:
        xa, xb = groups[a], groups[b]
        if len(xa) >= 2 and len(xb) >= 2:
            _, p = mannwhitneyu(xa, xb, alternative="two-sided")
        else:
            p = np.nan
        rows.append([a, b, p])
        pvals.append(p)
    if not rows:
        return pd.DataFrame(columns=["group1","group2","p_raw","p_adj","sig"])
    ok = ~pd.isna(pvals)
    adj = np.array([np.nan] * len(pvals), dtype=float)
    if ok.sum():
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(np.array(pvals)[ok], method=correction)
        adj[ok] = p_adj
    out = pd.DataFrame(rows, columns=["group1", "group2", "p_raw"])
    out["p_adj"] = adj
    out["sig"] = out["p_adj"].apply(lambda x: "***" if pd.notna(x) and x <= .001
                                    else ("**" if pd.notna(x) and x <= .01
                                    else ("*" if pd.notna(x) and x <= .05 else "")))
    return out

# Example: if ANOVA is significant, run Tukey; if Kruskal is significant, run MW
# (Here we run both for illustration and save tables)
for label, gcol in GROUP_COLS.items():
    if gcol in df_analytic.columns:
        tuk = tukey_posthoc(df_analytic, gcol, "Adopt_mean")
        if tuk is not None:
            tuk.to_csv(os.path.join(FIG_DIR, f"tukey_adopt_by_{label.lower()}.csv"), index=False)
        mw = pairwise_mannwhitney(df_analytic, gcol, "Adopt_mean")
        mw.to_csv(os.path.join(FIG_DIR, f"mw_adopt_by_{label.lower()}.csv"), index=False)

# -----------------------------
# 10) REGRESSION ANALYSIS (OLS with HC3 robust SE)
# -----------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

REG_DIR = FIG_DIR  # save outputs with the other tables/figures

# Columns we'll use
reg_cols = ["Adopt_mean", "Tech", "Org", "Env", "Trust"]
df_reg = df_analytic[reg_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()

# Mean-center predictors (interpretability; required before interactions)
for c in ["Tech", "Org", "Env", "Trust"]:
    df_reg[c + "_c"] = df_reg[c] - df_reg[c].mean()

# Interaction terms
df_reg["Tech_x_Trust"] = df_reg["Tech_c"] * df_reg["Trust_c"]
df_reg["Org_x_Trust"]  = df_reg["Org_c"]  * df_reg["Trust_c"]
df_reg["Env_x_Trust"]  = df_reg["Env_c"]  * df_reg["Trust_c"]

def tidy_ols(result, label):
    """Return a tidy DataFrame with coef, robust SE (HC3), t, p, stars, R2, adjR2, n."""
    # HC3 robust
    robust = result.get_robustcov_results(cov_type="HC3")
    summ = robust.summary2().tables[1].copy()  # coef table

    tbl = summ.rename(columns={
        "Coef.": "coef",
        "Std.Err.": "se",
        "t": "t",
        "P>|t|": "p"
    })[["coef","se","t","p"]]

    # stars
    def star(p):
        return "***" if p <= .001 else ("**" if p <= .01 else ("*" if p <= .05 else ""))
    tbl["sig"] = tbl["p"].apply(star)

    # model stats
    n = int(result.nobs)
    r2 = float(result.rsquared)
    r2adj = float(result.rsquared_adj)

    # Nice index name
    tbl.index.name = "term"
    tbl.reset_index(inplace=True)

    # append model-level info as last row for convenience
    info = pd.DataFrame({
        "term": ["__model__"],
        "coef": [np.nan],
        "se":   [np.nan],
        "t":    [np.nan],
        "p":    [np.nan],
        "sig":  [f"R2={r2:.3f}; adj.R2={r2adj:.3f}; n={n}"]
    })
    out = pd.concat([tbl, info], ignore_index=True)
    out.insert(0, "model", label)
    return out

# Baseline model: Adoption ~ Tech + Org + Env (centered)
formula_base = "Adopt_mean ~ Tech_c + Org_c + Env_c"
mod_base = smf.ols(formula=formula_base, data=df_reg).fit()
base_tbl = tidy_ols(mod_base, "Baseline")

# Moderation model: add Trust + interactions
formula_mod = "Adopt_mean ~ Tech_c + Org_c + Env_c + Trust_c + Tech_x_Trust + Org_x_Trust + Env_x_Trust"
mod_moderated = smf.ols(formula=formula_mod, data=df_reg).fit()
mod_tbl = tidy_ols(mod_moderated, "Moderated")

# Save CSVs
base_path = os.path.join(REG_DIR, "regression_baseline_HC3.csv")
mod_path  = os.path.join(REG_DIR, "regression_moderated_HC3.csv")
base_tbl.to_csv(base_path, index=False)
mod_tbl.to_csv(mod_path, index=False)

# Also save plain-text summaries (handy for Appendix)
with open(os.path.join(REG_DIR, "regression_baseline_summary.txt"), "w") as f:
    f.write(mod_base.summary().as_text())
with open(os.path.join(REG_DIR, "regression_moderated_summary.txt"), "w") as f:
    f.write(mod_moderated.summary().as_text())

print(f"Saved regression tables:\n  - {base_path}\n  - {mod_path}")

# (Optional) VIF diagnostics on centered predictors (without interaction terms)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
    X = df_reg[["Tech_c","Org_c","Env_c","Trust_c"]].copy()
    X = sm.add_constant(X)
    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_rows.append({"variable": col, "VIF": float(VIF(X.values, i))})
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(os.path.join(REG_DIR, "vif_centered_predictors.csv"), index=False)
    print("Saved VIF table: vif_centered_predictors.csv")
except Exception as e:
    print("VIF calculation skipped:", e)

# -----------------------------
# 8) SIGNIFICANCE-ANNOTATED BOXPLOTS
# -----------------------------
def add_sig_bar(ax, x1, x2, y, p, h=0.03):
    """Draw a significance bar only when called (we call it only if p<.05)."""
    # star labels only
    if p < 0.001:
        label = "***"
    elif p < 0.01:
        label = "**"
    else:  # 0.01 <= p < 0.05
        label = "*"

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c="black")
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha="center", va="bottom")


def boxplot_with_sigs(data, group_col, value_col, comparisons=None, title=None, ylabel=None, fname=None):
    # 1) Categories present in data (order as observed)
    order = [str(g) for g in data[group_col].dropna().unique()]
    series = [
        pd.to_numeric(
            data.loc[data[group_col].astype(str) == g, value_col],
            errors="coerce"
        ).dropna().values
        for g in order
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bp = ax.boxplot(series, tick_labels=order, patch_artist=True, showmeans=False)

    # 2) Fixed colors per group using your COLOR_MAPS
    color_dict = COLOR_MAPS.get(group_col, {})
    for patch, lbl in zip(bp['boxes'], order):
        color = color_dict.get(lbl, "lightgray")  # fallback to gray if not found
        patch.set_facecolor(color)
        patch.set_edgecolor("black")

    for key in ['whiskers', 'caps', 'medians']:
        for line in bp[key]:
            line.set_color("black")

    # 3) Titles and axis labels (rename FirmA -> Firm)
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.set_ylabel(ylabel or value_col)
    xlabel_map = {"FirmA": "Firm", "Role": "Role", "Dept": "Department"}
    ax.set_xlabel(xlabel_map.get(group_col, group_col))

    # 4) x-tick labels with sample sizes; make Role labels multi-line to avoid overlap
    ns = [len(x) for x in series]

    # Canonical order for Role so it’s consistent across plots
    if group_col == "Role":
        canonical_order = [
            "Associate/Junior",
            "Senior Associate/Consultant",
            "Senior Manager/Manager",
            "Partner/Director",
        ]
        # keep only those present
        order = [lbl for lbl in canonical_order if lbl in order]

        # FULL names with line breaks (no truncation)
        role_labels_wrapped = {
            "Associate/Junior": "Associate/\nJunior",
            "Senior Associate/Consultant": "Senior Associate/\nConsultant",
            "Senior Manager/Manager": "Senior Manager/\nManager",
            "Partner/Director": "Partner/\nDirector",
        }
        display_labels = [role_labels_wrapped.get(lbl, lbl) for lbl in order]

        # Recompute series & ns because we changed 'order'
        series = [
            pd.to_numeric(
                data.loc[data[group_col].astype(str) == g, value_col],
                errors="coerce"
            ).dropna().values
            for g in order
        ]
        ns = [len(x) for x in series]
    else:
        display_labels = order

    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(display_labels, ns)])

    # 5) Optional significance bars (only for pairs that exist)
    valid_pairs = []
    if comparisons:
        for (a, b) in comparisons:
            sa, sb = str(a), str(b)
            if sa in order and sb in order:
                valid_pairs.append((sa, sb))

    if valid_pairs:
        ymax = np.nanmax([np.max(x) if len(x) else np.nan for x in series])
        ymin = np.nanmin([np.min(x) if len(x) else np.nan for x in series])
        step = (ymax - ymin) * 0.15 if np.isfinite(ymax) and np.isfinite(ymin) else 0.3
        y = ymax + step
        from scipy.stats import mannwhitneyu
        for (a, b) in valid_pairs:
            A = pd.to_numeric(data.loc[data[group_col].astype(str) == a, value_col], errors="coerce").dropna().values
            B = pd.to_numeric(data.loc[data[group_col].astype(str) == b, value_col], errors="coerce").dropna().values
            if len(A) >= 2 and len(B) >= 2:
                _, p = mannwhitneyu(A, B, alternative="two-sided")
                if p < 0.05:  # draw only if significant
                    x1, x2 = order.index(a) + 1, order.index(b) + 1
                    add_sig_bar(ax, x1, x2, y, p)
                    y += step

    fig.tight_layout()
    if fname:
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

# Examples: Adoption/Trust by Department/Role/Firm with a few illustrative comparisons
if GROUP_COLS.get("Dept") in df_analytic.columns:
    boxplot_with_sigs(df_analytic, GROUP_COLS["Dept"], "Adopt_mean",
                      comparisons=[("TaxTech", "VAT"), ("CIT", "VAT")],
                      title="Adoption by Department",
                      ylabel="Adoption (1–5)",
                      fname="box_adopt_by_dept.png")

if GROUP_COLS.get("Role") in df_analytic.columns:
    boxplot_with_sigs(df_analytic, GROUP_COLS["Role"], "Adopt_mean",
                      comparisons=[("Partner/Director", "Senior Associate/Consultant"),
                                   ("Senior Manager/Manager", "Associate/Junior")],
                      title="Adoption by Role",
                      ylabel="Adoption (1–5)",
                      fname="box_adopt_by_role.png")

if GROUP_COLS.get("Firm") in df_analytic.columns:
    boxplot_with_sigs(df_analytic, GROUP_COLS["Firm"], "Adopt_mean",
                      comparisons=[("Firm A", "Firm B")],
                      title="Adoption by Firm (anonymized)",
                      ylabel="Adoption (1–5)",
                      fname="box_adopt_by_firm.png")

# -----------------------------
# 9) SAVE A SNAPSHOT OF THE ANALYTIC N
# -----------------------------
with open(os.path.join(FIG_DIR, "sample_size.txt"), "w") as f:
    f.write(f"Analytic sample size (n): {len(df_analytic)}\n")

print("Dept labels:", sorted(df_analytic["Dept"].dropna().unique()))
print("Role labels:", sorted(df_analytic["Role"].dropna().unique()))
print("FirmA labels:", sorted(df_analytic["FirmA"].dropna().unique()))

# End of script
