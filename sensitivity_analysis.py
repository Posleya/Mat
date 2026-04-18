"""
Sensitivity Analysis for MathorCup 2026 C Question
====================================================
Generates three figures (Fig11–Fig13) quantifying how robust the
composite risk-warning model's outputs are to:

  Fig11 – Weight perturbation sensitivity
          Vary each of the four component weights (±10 %, ±20 %, ±30 %)
          and record the fraction of subjects whose THREE-LEVEL risk label
          changes relative to the base model.

  Fig12 – Cut-point percentile sensitivity
          Vary the (Low/Mid) and (Mid/High) percentile thresholds
          simultaneously across a grid (p_lo ∈ {20,25,30,33,40},
          p_hi ∈ {60,67,70,75,80}).  Heat-map shows fraction of
          subjects whose risk label changes vs. the (33/67) baseline.

  Fig13 – Bootstrap AUC stability
          200 bootstrap resamples of the Stage-A non-lipid XGBoost
          early-screening model; 95 % CI and distribution for AUC.

All text / labels are in English and suitable for inclusion in an
academic paper.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker
import scipy.stats as stats

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ── style ──────────────────────────────────────────────────────────────────────
rcParams["font.family"]     = "DejaVu Sans"
rcParams["axes.titlesize"]  = 13
rcParams["axes.labelsize"]  = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9
rcParams["figure.dpi"]      = 100

OUT_DIR = "/home/runner/work/Mat/Mat/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

COLORS = {
    "blue":   "#2E5E9A",
    "red":    "#E05A42",
    "green":  "#4A9E6B",
    "orange": "#E8952B",
    "gray":   "#888888",
    "light":  "#AEC6E8",
    "purple": "#9B59B6",
}

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load data (same processing as analysis_q2_risk_model.py)
# ══════════════════════════════════════════════════════════════════════════════
RAW = "/home/runner/work/Mat/Mat/附件1：样例数据.xlsx"
df_raw = pd.read_excel(RAW)

COL_EN = {
    "样本ID": "SampleID", "体质标签": "ConstitutionLabel",
    "平和质": "Balanced", "气虚质": "QiDeficiency",
    "阳虚质": "YangDeficiency", "阴虚质": "YinDeficiency",
    "痰湿质": "PhlegmDampness", "湿热质": "DampHeat",
    "血瘀质": "BloodStasis", "气郁质": "QiStagnation",
    "特禀质": "SpecialIntrinsic",
    "ADL用厕": "ADL_Toilet", "ADL吃饭": "ADL_Eating",
    "ADL步行": "ADL_Walking", "ADL穿衣": "ADL_Dressing",
    "ADL洗澡": "ADL_Bathing", "ADL总分": "ADL_Total",
    "IADL购物": "IADL_Shopping", "IADL做饭": "IADL_Cooking",
    "IADL理财": "IADL_Finance", "IADL交通": "IADL_Transport",
    "IADL服药": "IADL_Medication", "IADL总分": "IADL_Total",
    "活动量表总分（ADL总分+IADL总分）": "Activity_Total",
    "HDL-C（高密度脂蛋白）": "HDL_C",
    "LDL-C（低密度脂蛋白）": "LDL_C",
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "空腹血糖": "FastingGlucose", "血尿酸": "UricAcid", "BMI": "BMI",
    "高血脂症二分类标签": "Hyperlipidemia",
    "血脂异常分型标签（确诊病例）": "DyslipidemiaType",
    "年龄组": "AgeGroup", "性别": "Sex",
    "吸烟史": "Smoking", "饮酒史": "Alcohol",
}
df = df_raw.rename(columns=COL_EN).copy()

CONSTITUTION_SCORES = ["Balanced","QiDeficiency","YangDeficiency","YinDeficiency",
                        "PhlegmDampness","DampHeat","BloodStasis","QiStagnation",
                        "SpecialIntrinsic"]
ADL_ITEMS   = ["ADL_Toilet","ADL_Eating","ADL_Walking","ADL_Dressing","ADL_Bathing"]
IADL_ITEMS  = ["IADL_Shopping","IADL_Cooking","IADL_Finance","IADL_Transport",
                "IADL_Medication"]
ADL_SCORES  = ["ADL_Total","IADL_Total","Activity_Total"]
ASSOC_VARS  = ["FastingGlucose","UricAcid","BMI"]
DEMO_VARS   = ["AgeGroup","Sex","Smoking","Alcohol"]
BLOOD_LIPID = ["TC","TG","LDL_C","HDL_C"]

LIPID_REF = {
    "TC":    (3.1,  6.2),
    "TG":    (0.56, 1.7),
    "LDL_C": (2.07, 3.1),
    "HDL_C": (1.04, 1.55),
}

def count_lipid_abnormal(row):
    cnt = 0
    for col, (lo, hi) in LIPID_REF.items():
        if col in row.index and not pd.isna(row[col]):
            cnt += int(row[col] < lo) if col == "HDL_C" else int(row[col] > hi)
    return cnt

df["LipidAbnormalCount"] = df.apply(count_lipid_abnormal, axis=1)
df["LipidAbnormal"]      = (df["LipidAbnormalCount"] >= 1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Re-derive base model (identical to analysis_q2_risk_model.py)
# ══════════════════════════════════════════════════════════════════════════════
y = df["Hyperlipidemia"].values

SCREEN_FEATURES = (
    CONSTITUTION_SCORES + ADL_SCORES + ADL_ITEMS + IADL_ITEMS +
    ASSOC_VARS + DEMO_VARS
)
SCREEN_FEATURES = [c for c in SCREEN_FEATURES if c in df.columns]
X_sc_raw = df[SCREEN_FEATURES].values

n_neg, n_pos = (y==0).sum(), (y==1).sum()
spw = n_neg / max(n_pos, 1)

xgb_A = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric="logloss", verbosity=0, random_state=42
)
cal_A    = CalibratedClassifierCV(xgb_A, method="sigmoid", cv=3)
pipe_A   = Pipeline([("scaler", StandardScaler()), ("clf", cal_A)])
cv5      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob_screen = cross_val_predict(pipe_A, X_sc_raw, y, cv=cv5,
                                  method="predict_proba")[:, 1]
auc_base = roc_auc_score(y, y_prob_screen)
print(f"Base model OOF AUC = {auc_base:.4f}")

# Excess-AUC weights (base)
auc_A_val = roc_auc_score(y, y_prob_screen)
auc_B_val = roc_auc_score(y, df["PhlegmDampness"].values)
auc_C_val = roc_auc_score(y, -df["Activity_Total"].values)
auc_D_val = roc_auc_score(y, df["LipidAbnormalCount"].values)

excess_base = np.array([
    max(auc_A_val - 0.5, 1e-6),
    max(auc_B_val - 0.5, 1e-6),
    max(auc_C_val - 0.5, 1e-6),
    max(auc_D_val - 0.5, 1e-6),
])
W_BASE = excess_base / excess_base.sum() * 100.0   # [W_A, W_B, W_C, W_D]
print(f"Base weights: Screen={W_BASE[0]:.2f}  PD={W_BASE[1]:.2f}  "
      f"Act={W_BASE[2]:.2f}  Lipid={W_BASE[3]:.2f}")

# Tier functions
def phlegm_tier_norm(v):
    if v < 40:  return 0.0
    if v < 60:  return 0.33
    if v < 80:  return 0.67
    return 1.0

def activity_tier_norm(v):
    if v >= 70: return 0.0
    if v >= 55: return 0.33
    if v >= 40: return 0.67
    return 1.0

def lipid_score_norm(cnt):
    return min(int(cnt), 4) / 4.0

df["P_screen"]    = y_prob_screen
df["t_phlegm"]    = df["PhlegmDampness"].apply(phlegm_tier_norm)
df["t_activity"]  = df["Activity_Total"].apply(activity_tier_norm)
df["t_lipid"]     = df["LipidAbnormalCount"].apply(lipid_score_norm)

def compute_risk_labels(w, pct_lo=33, pct_hi=67):
    """Compute composite score and 3-level labels for weight vector w."""
    score = (df["P_screen"] * w[0] +
             df["t_phlegm"]   * w[1] +
             df["t_activity"] * w[2] +
             df["t_lipid"]    * w[3])
    lo_cut  = float(np.percentile(score, pct_lo))
    hi_cut  = float(np.percentile(score, pct_hi))
    labels  = np.where(score < lo_cut, 0, np.where(score < hi_cut, 1, 2))
    return labels

BASE_LABELS = compute_risk_labels(W_BASE, 33, 67)
N = len(BASE_LABELS)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 – Weight perturbation sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig 11: Weight perturbation sensitivity …")

PERTURB_LEVELS = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]
COMP_NAMES     = ["Screen\n(XGBoost)", "Phlegm-\nDampness", "Physical\nActivity", "Lipid\nAbnormal"]
COMP_KEYS      = [0, 1, 2, 3]

records = []
for ci in COMP_KEYS:
    row = {"Component": COMP_NAMES[ci]}
    for delta in PERTURB_LEVELS:
        w_new = W_BASE.copy()
        w_new[ci] = max(W_BASE[ci] * (1 + delta), 0.001)
        # re-normalise so weights still sum to 100
        w_new = w_new / w_new.sum() * 100.0
        new_labels  = compute_risk_labels(w_new, 33, 67)
        frac_change = (new_labels != BASE_LABELS).mean()
        row[f"{delta:+.0%}"] = frac_change * 100
    records.append(row)

df_pert = pd.DataFrame(records).set_index("Component")
pct_cols = [f"{d:+.0%}" for d in PERTURB_LEVELS]

fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(pct_cols))
width  = 0.18
colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]
for i, (comp_name, color) in enumerate(zip(COMP_NAMES, colors)):
    vals = [df_pert.loc[comp_name, c] for c in pct_cols]
    ax.bar(x + i * width, vals, width,
           label=comp_name.replace("\n", " "), color=color, alpha=0.82,
           edgecolor="white")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(pct_cols, fontsize=9)
ax.set_xlabel("Weight Perturbation (relative to baseline)", fontsize=11)
ax.set_ylabel("Risk-Level Change Rate (%)", fontsize=11)
ax.set_title("Fig 11 – Sensitivity to Component Weight Perturbation\n"
             "(Fraction of subjects whose risk tier changes)", fontweight="bold")
ax.legend(title="Component", fontsize=8, title_fontsize=8,
          loc="upper left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig11_WeightSensitivity.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig 11 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 – Cut-point percentile sensitivity heat-map
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 12: Cut-point sensitivity heat-map …")

P_LO_VALS = [20, 25, 30, 33, 40]
P_HI_VALS = [60, 67, 70, 75, 80]

change_grid = np.zeros((len(P_LO_VALS), len(P_HI_VALS)))
hld_hi_grid = np.zeros_like(change_grid)

for i, p_lo in enumerate(P_LO_VALS):
    for j, p_hi in enumerate(P_HI_VALS):
        if p_lo >= p_hi:
            change_grid[i, j] = np.nan
            hld_hi_grid[i, j] = np.nan
            continue
        new_labels = compute_risk_labels(W_BASE, p_lo, p_hi)
        change_grid[i, j]   = (new_labels != BASE_LABELS).mean() * 100
        # HLD prevalence in High-risk group
        high_mask = new_labels == 2
        hld_hi_grid[i, j] = (df.loc[high_mask, "Hyperlipidemia"].mean() * 100
                               if high_mask.sum() > 0 else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

def draw_heatmap(ax, data, title, cmap, fmt=".1f", cbar_label=""):
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(len(P_HI_VALS)))
    ax.set_xticklabels([f"p_hi={v}" for v in P_HI_VALS], rotation=30, ha="right")
    ax.set_yticks(range(len(P_LO_VALS)))
    ax.set_yticklabels([f"p_lo={v}" for v in P_LO_VALS])
    ax.set_xlabel("High/Medium cut-point (percentile)", fontsize=10)
    ax.set_ylabel("Low/Medium cut-point (percentile)", fontsize=10)
    ax.set_title(title, fontweight="bold")
    # Annotate cells
    for r in range(len(P_LO_VALS)):
        for c in range(len(P_HI_VALS)):
            if not np.isnan(data[r, c]):
                ax.text(c, r, f"{data[r, c]:{fmt}}",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if data[r, c] > (np.nanmax(data)+np.nanmin(data))/2 else "black")
    # Highlight the baseline cell (33/67)
    try:
        r0 = P_LO_VALS.index(33)
        c0 = P_HI_VALS.index(67)
        ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), 1, 1,
                                   fill=False, edgecolor="yellow", lw=2.5))
    except ValueError:
        pass
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

draw_heatmap(axes[0], change_grid,
             "Risk-Label Change Rate vs. Baseline (33/67)\n[Yellow box = baseline cut-points]",
             cmap="RdYlGn_r", fmt=".1f", cbar_label="% subjects changed")

draw_heatmap(axes[1], hld_hi_grid,
             "HLD Prevalence in High-Risk Tier\nunder Different Cut-points",
             cmap="YlOrRd", fmt=".1f", cbar_label="HLD prevalence (%)")

plt.suptitle("Fig 12 – Sensitivity to Risk-Stratification Cut-point Percentiles",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig12_CutpointSensitivity.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig 12 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 – Bootstrap AUC stability of the early-screening model
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 13: Bootstrap AUC stability (200 rounds) …")

N_BOOT = 200
boot_aucs = []

scaler_boot = StandardScaler()
X_sc_boot   = scaler_boot.fit_transform(X_sc_raw)

for seed in range(N_BOOT):
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(N, N, replace=True)
    Xb   = X_sc_boot[idx]
    yb   = y[idx]
    if yb.sum() < 5 or (yb==0).sum() < 5:
        continue
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(1.0, (yb==0).sum()/max((yb==1).sum(),1)),
        eval_metric="logloss", verbosity=0, random_state=seed
    )
    m.fit(Xb, yb)
    # evaluate on out-of-bag samples
    oob_mask = np.ones(N, dtype=bool)
    oob_mask[np.unique(idx)] = False
    if oob_mask.sum() < 20:
        continue
    X_oob = X_sc_boot[oob_mask]
    y_oob = y[oob_mask]
    if y_oob.sum() < 3 or (y_oob==0).sum() < 3:
        continue
    try:
        p_oob = m.predict_proba(X_oob)[:, 1]
        boot_aucs.append(roc_auc_score(y_oob, p_oob))
    except Exception:
        continue

boot_aucs = np.array(boot_aucs)
mean_auc  = boot_aucs.mean()
ci_lo     = np.percentile(boot_aucs, 2.5)
ci_hi     = np.percentile(boot_aucs, 97.5)
print(f"  Bootstrap AUC: {mean_auc:.4f}  95%CI [{ci_lo:.4f}, {ci_hi:.4f}]"
      f"  n_valid={len(boot_aucs)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: histogram
ax = axes[0]
ax.hist(boot_aucs, bins=30, color=COLORS["blue"], alpha=0.75,
        edgecolor="white", density=True)
ax.axvline(mean_auc, color="black",  lw=1.5, linestyle="-",
           label=f"Mean AUC = {mean_auc:.4f}")
ax.axvline(ci_lo,    color=COLORS["red"], lw=1.5, linestyle="--",
           label=f"95% CI  [{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi,    color=COLORS["red"], lw=1.5, linestyle="--")
ax.axvline(auc_base, color=COLORS["orange"], lw=1.5, linestyle=":",
           label=f"OOF AUC (base) = {auc_base:.4f}")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
                 ci_lo, ci_hi, alpha=0.10, color=COLORS["blue"])
ax.set_xlabel("OOB AUC (Bootstrap)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Bootstrap AUC Distribution\n(200 Resamples, OOB Evaluation)",
             fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# Right: sorted AUC with 95% CI band
ax = axes[1]
sorted_aucs = np.sort(boot_aucs)
idx_arr     = np.arange(1, len(sorted_aucs) + 1)
ax.plot(idx_arr, sorted_aucs, color=COLORS["blue"], lw=1.2, alpha=0.8)
ax.fill_between(idx_arr, ci_lo, ci_hi, alpha=0.15, color=COLORS["blue"],
                label=f"95% CI band [{ci_lo:.3f}, {ci_hi:.3f}]")
ax.axhline(mean_auc, color="black", lw=1, linestyle="--",
           label=f"Mean = {mean_auc:.4f}")
ax.set_xlabel("Bootstrap Iteration (sorted by AUC)", fontsize=11)
ax.set_ylabel("AUC", fontsize=11)
ax.set_title("Sorted Bootstrap AUC Values\n(Stability Assessment)",
             fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Fig 13 – Bootstrap Stability of the Early-Screening XGBoost Model",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig13_BootstrapAUC.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Fig 13 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Summary table for inline use in the report
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Sensitivity Summary ===")
print("\n[Fig 11] Max change rate per component (across all perturbation levels):")
for comp in COMP_NAMES:
    vals = [df_pert.loc[comp, c] for c in pct_cols if comp in df_pert.index]
    print(f"  {comp.replace(chr(10),' '):20s}: max={max(vals):.2f}%  min={min(vals):.2f}%")

print(f"\n[Fig 12] Change rate at baseline (33/67) = 0.0% (reference)")
print(f"         Max change rate across all valid grids: "
      f"{np.nanmax(change_grid):.1f}%")
print(f"         Min HLD prevalence in High-risk tier: "
      f"{np.nanmin(hld_hi_grid):.1f}%")
print(f"         Max HLD prevalence in High-risk tier: "
      f"{np.nanmax(hld_hi_grid):.1f}%")

print(f"\n[Fig 13] Bootstrap AUC: {mean_auc:.4f}  "
      f"95%CI [{ci_lo:.4f}, {ci_hi:.4f}]  "
      f"Std={boot_aucs.std():.4f}")

print(f"\n✓ All sensitivity figures saved to {OUT_DIR}/")
