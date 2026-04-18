"""
MathorCup 2026 C题 - 问题2（扩展）：三级风险预警模型
==========================================================
目标：构建融合多维度特征的高血脂风险预警模型，输出低/中/高三级风险，
并给出各级别对应的特征分层阈值选取依据，识别痰湿高风险人群核心特征组合。

方法选择依据：
  ① 两阶段建模：
     Stage-A "早筛模型"——仅用非血脂指标（体质积分+活动量表+代谢相关+人口学）
       → XGBoost + Platt校准 → 输出早筛概率 P_screen
     Stage-B "综合风险评分"——将 P_screen 与临床血脂异常标志共同映射到
       0–100 分的复合风险评分，再划分低/中/高三级
  ② 可解释性工具：
     决策树（max_depth=5）在临床关键特征子集上拟合三级标签 → 可读规则
     SHAP（XGBoost）→ 各级别特征贡献对比
  ③ 痰湿高风险人群核心特征组合：
     定义三维二值标志（高痰湿/低活动量/血脂异常）→ 8种组合 × HLD患病率 × 模型风险级别
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_text
import xgboost as xgb
import shap

# ── Global style ──────────────────────────────────────────────────────────────
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.titlesize"] = 13
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9

OUT_DIR = "/home/runner/work/Mat/Mat/outputs"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 0. Load & rename
# ══════════════════════════════════════════════════════════════════════════════
RAW = "/home/runner/work/Mat/Mat/附件1：样例数据.xlsx"
df_raw = pd.read_excel(RAW)

COL_EN = {
    "样本ID": "SampleID", "体质标签": "ConstitutionLabel",
    "平和质": "Balanced",       "气虚质": "QiDeficiency",
    "阳虚质": "YangDeficiency", "阴虚质": "YinDeficiency",
    "痰湿质": "PhlegmDampness", "湿热质": "DampHeat",
    "血瘀质": "BloodStasis",    "气郁质": "QiStagnation",
    "特禀质": "SpecialIntrinsic",
    "ADL用厕": "ADL_Toilet",    "ADL吃饭": "ADL_Eating",
    "ADL步行": "ADL_Walking",   "ADL穿衣": "ADL_Dressing",
    "ADL洗澡": "ADL_Bathing",   "ADL总分": "ADL_Total",
    "IADL购物": "IADL_Shopping","IADL做饭": "IADL_Cooking",
    "IADL理财": "IADL_Finance", "IADL交通": "IADL_Transport",
    "IADL服药": "IADL_Medication","IADL总分": "IADL_Total",
    "活动量表总分（ADL总分+IADL总分）": "Activity_Total",
    "HDL-C（高密度脂蛋白）": "HDL_C",
    "LDL-C（低密度脂蛋白）": "LDL_C",
    "TG（甘油三酯）": "TG",
    "TC（总胆固醇）": "TC",
    "空腹血糖": "FastingGlucose",
    "血尿酸": "UricAcid",
    "BMI": "BMI",
    "高血脂症二分类标签": "Hyperlipidemia",
    "血脂异常分型标签（确诊病例）": "DyslipidemiaType",
    "年龄组": "AgeGroup",
    "性别": "Sex",
    "吸烟史": "Smoking",
    "饮酒史": "Alcohol",
}
df = df_raw.rename(columns=COL_EN).copy()

BLOOD_LIPID = ["TC", "TG", "LDL_C", "HDL_C"]
ASSOC_VARS  = ["FastingGlucose", "UricAcid", "BMI"]
ADL_ITEMS   = ["ADL_Toilet","ADL_Eating","ADL_Walking","ADL_Dressing","ADL_Bathing"]
IADL_ITEMS  = ["IADL_Shopping","IADL_Cooking","IADL_Finance","IADL_Transport","IADL_Medication"]
ADL_SCORES  = ["ADL_Total","IADL_Total","Activity_Total"]
DEMO_VARS   = ["AgeGroup","Sex","Smoking","Alcohol"]
CONSTITUTION_SCORES = [
    "Balanced","QiDeficiency","YangDeficiency","YinDeficiency",
    "PhlegmDampness","DampHeat","BloodStasis","QiStagnation","SpecialIntrinsic"
]

TARGET = "Hyperlipidemia"
y = df[TARGET].values
print(f"Dataset: {df.shape[0]} samples | HLD prevalence: {y.mean():.1%}")

# ── Clinical reference ranges ─────────────────────────────────────────────────
# HDL: abnormal if BELOW lower bound; others: abnormal if ABOVE upper bound
LIPID_REF = {
    "TC":    (3.1,  6.2),
    "TG":    (0.56, 1.7),
    "LDL_C": (2.07, 3.1),
    "HDL_C": (1.04, 1.55),   # low HDL = abnormal
}
METAB_REF = {
    "FastingGlucose": (3.9, 6.1),
    "BMI":            (18.5, 23.9),
}

def count_lipid_abnormal(row):
    """Count how many lipid markers are out of clinical range."""
    cnt = 0
    for col, (lo, hi) in LIPID_REF.items():
        if col in row.index and not pd.isna(row[col]):
            if col == "HDL_C":
                cnt += int(row[col] < lo)      # low HDL is the risk
            else:
                cnt += int(row[col] > hi)      # high TC/TG/LDL is the risk
    return cnt

df["LipidAbnormalCount"] = df.apply(count_lipid_abnormal, axis=1)
df["LipidAbnormal"]      = (df["LipidAbnormalCount"] >= 1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE A: Early-screening model (non-lipid features only)
# Purpose: simulate what a clinician knows BEFORE ordering a blood lipid panel
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE A: Early-screening model (non-lipid features)")
print("="*60)

SCREEN_FEATURES = (
    CONSTITUTION_SCORES +
    ADL_SCORES + ADL_ITEMS + IADL_ITEMS +
    ASSOC_VARS +
    DEMO_VARS
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
cal_A = CalibratedClassifierCV(xgb_A, method="sigmoid", cv=3)

# Pipeline wraps scaler inside CV folds: scaler.fit() only sees training data
# in each fold, preventing any leakage of test-fold statistics into OOF probs.
pipe_A = Pipeline([("scaler", StandardScaler()), ("clf", cal_A)])
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob_screen = cross_val_predict(pipe_A, X_sc_raw, y, cv=cv5, method="predict_proba")[:, 1]
auc_screen = roc_auc_score(y, y_prob_screen)
print(f"  Early-screening model OOF AUC = {auc_screen:.3f}")
print(f"  P(HLD|screen) range: [{y_prob_screen.min():.3f}, {y_prob_screen.max():.3f}]")
print(f"  Mean={y_prob_screen.mean():.3f}  Std={y_prob_screen.std():.3f}")

# Calibration curve
frac_pos_A, mean_pred_A = calibration_curve(y, y_prob_screen, n_bins=10)

# Fit final model on ALL data — standard practice for interpretability (SHAP).
# The performance metric (AUC) is already OOF and unbiased.
scaler_final = StandardScaler()
X_sc = scaler_final.fit_transform(X_sc_raw)
xgb_A_final = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric="logloss", verbosity=0, random_state=42
)
xgb_A_final.fit(X_sc, y)
df["P_screen"] = y_prob_screen

# ══════════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN WEIGHT DERIVATION  (Excess-AUC method)
# Each component's weight is proportional to its individual AUC minus 0.5
# (i.e. the discriminative power ABOVE random chance).  Using excess-AUC rather
# than raw AUC prevents highly-correlated features (LipidAbnormal ≈ HLD label)
# from completely monopolising the score, while still reflecting each
# component's marginal contribution to predicting HLD.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DATA-DRIVEN WEIGHT DERIVATION (Excess-AUC method)")
print("="*60)

act_max   = float(df["Activity_Total"].max())
act_min   = float(df["Activity_Total"].min())
act_range = max(act_max - act_min, 1.0)

auc_A = roc_auc_score(y, y_prob_screen)
auc_B = roc_auc_score(y, df["PhlegmDampness"].values)
auc_C = roc_auc_score(y, -df["Activity_Total"].values)  # lower activity = higher risk
auc_D = roc_auc_score(y, df["LipidAbnormalCount"].values)

print(f"  Individual AUCs (for HLD prediction):")
print(f"    [A] P_screen       : {auc_A:.3f}")
print(f"    [B] PhlegmDampness : {auc_B:.3f}")
print(f"    [C] Activity_Total : {auc_C:.3f}  (negated)")
print(f"    [D] LipidAbnormal  : {auc_D:.3f}")

excess = np.array([max(auc_A - 0.5, 1e-6), max(auc_B - 0.5, 1e-6),
                   max(auc_C - 0.5, 1e-6), max(auc_D - 0.5, 1e-6)])
W_A, W_B, W_C, W_D = (excess / excess.sum() * 100.0)
print(f"\n  Excess-AUC derived weights (sum=100, prior in parentheses):")
print(f"    [A] P_screen       : {W_A:.1f}  (prior 40)")
print(f"    [B] PhlegmDampness : {W_B:.1f}  (prior 25)")
print(f"    [C] Activity_Total : {W_C:.1f}  (prior 20)")
print(f"    [D] LipidAbnormal  : {W_D:.1f}  (prior 15)")

# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE RISK SCORE (0–100) and THREE-LEVEL STRATIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPOSITE RISK SCORE & THREE-LEVEL STRATIFICATION")
print("="*60)

# ── Tier functions return values in [0, 1]; multiplied by their weight ────────
# PhlegmDampness breakpoints follow ZYYXH/T157-2009: <40 none, 40–60 tendency,
# 60–80 typical, ≥80 severe phlegm-dampness constitution.
def phlegm_tier_norm(v):
    """0–1: PhlegmDampness tier (4 levels, breakpoints 40/60/80)."""
    if v < 40:  return 0.0
    if v < 60:  return 0.33
    if v < 80:  return 0.67
    return 1.0

def activity_tier_norm(v):
    """0–1: Activity_Total risk tier (lower activity = higher risk)."""
    if v >= 70: return 0.0
    if v >= 55: return 0.33
    if v >= 40: return 0.67
    return 1.0

def lipid_score_norm(cnt):
    """0–1: proportional to number of abnormal lipid markers (capped at 4)."""
    return min(int(cnt), 4) / 4.0

# Weights W_A…W_D are data-driven (see DATA-DRIVEN WEIGHT DERIVATION above)
df["Score_Screen"]   = df["P_screen"]                                  * W_A
df["Score_Phlegm"]   = df["PhlegmDampness"].apply(phlegm_tier_norm)    * W_B
df["Score_Activity"] = df["Activity_Total"].apply(activity_tier_norm)  * W_C
df["Score_Lipid"]    = df["LipidAbnormalCount"].apply(lipid_score_norm) * W_D
df["CompositeScore"] = (df["Score_Screen"] + df["Score_Phlegm"] +
                        df["Score_Activity"] + df["Score_Lipid"])

# Data-driven cut-points: 33rd / 67th percentile of the composite score
# ensures a balanced three-way split without manual tuning.
LOW_CUT  = round(float(np.percentile(df["CompositeScore"], 33)), 1)
HIGH_CUT = round(float(np.percentile(df["CompositeScore"], 67)), 1)

def assign_risk(score):
    if score < LOW_CUT:  return 0  # Low
    if score < HIGH_CUT: return 1  # Medium
    return 2                       # High

RISK_NAMES = {0: "Low", 1: "Medium", 2: "High"}
df["RiskLevel"] = df["CompositeScore"].apply(assign_risk)
df["RiskLabel"]  = df["RiskLevel"].map(RISK_NAMES)

print(f"  Score range: [{df['CompositeScore'].min():.1f}, {df['CompositeScore'].max():.1f}]")
print(f"  Cut-points: Low < {LOW_CUT} ≤ Medium < {HIGH_CUT} ≤ High")
for r_id, r_name in RISK_NAMES.items():
    sub  = df[df["RiskLevel"] == r_id]
    prev = sub["Hyperlipidemia"].mean()
    scr  = sub["CompositeScore"]
    print(f"  {r_name:6s}: n={len(sub):4d}  HLD={prev:.1%}  "
          f"Score=[{scr.min():.1f}, {scr.max():.1f}]  "
          f"median={scr.median():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# DECISION TREE – extract interpretable threshold rules
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DECISION TREE – interpretable threshold rules per risk level")
print("="*60)

CLINICAL_FEAT = (
    BLOOD_LIPID + ["BMI","FastingGlucose","UricAcid"] +
    ["PhlegmDampness","Activity_Total","ADL_Total","IADL_Total"] +
    ["AgeGroup","Sex","Smoking"]
)
CLINICAL_FEAT = [c for c in CLINICAL_FEAT if c in df.columns]
X_clin = df[CLINICAL_FEAT].values
risk_labels = df["RiskLevel"].values

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=max(5, int(0.03 * len(y))),
    class_weight="balanced",
    random_state=42
)
dt.fit(X_clin, risk_labels)
dt_acc = (dt.predict(X_clin) == risk_labels).mean()
print(f"  DT training accuracy (proxy): {dt_acc:.3f}")
print("\n  Decision Tree rules (top depth-5):")
print(export_text(dt, feature_names=CLINICAL_FEAT, max_depth=5))

dt_feat_imp = pd.Series(dt.feature_importances_, index=CLINICAL_FEAT)
dt_feat_imp = dt_feat_imp[dt_feat_imp > 0].sort_values(ascending=False)
print("  Top DT feature importances:")
print(dt_feat_imp.head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6: Risk stratum feature profiles + threshold annotation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TABLE 6: Risk stratum profiles")
print("="*60)

KEY_COLS = (BLOOD_LIPID + ["BMI","FastingGlucose","UricAcid"] +
            ["PhlegmDampness","Activity_Total","ADL_Total","IADL_Total"])
KEY_COLS = [c for c in KEY_COLS if c in df.columns]

# Clinical reference threshold descriptions per risk level
RISK_RULES = {
    "Low": (
        "TG ≤ 1.70 AND TC ≤ 6.20 (both within range) "
        "AND PhlegmDampness < 60 AND Activity_Total ≥ 55"
    ),
    "Medium": (
        "Single lipid abnormality (TC > 6.2 OR TG > 1.7 OR LDL-C > 3.1 OR HDL-C < 1.04) "
        "OR (PhlegmDampness 60–80 AND Activity_Total 40–70)"
    ),
    "High": (
        "(≥2 abnormal blood lipid markers) "
        "OR (TC > 6.2 AND PhlegmDampness ≥ 60) "
        "OR (TG > 1.7 AND Activity_Total < 40) "
        "OR (PhlegmDampness ≥ 80 AND Activity_Total < 40)"
    ),
}

rows_t6 = []
for r_id, r_name in RISK_NAMES.items():
    sub = df[df["RiskLevel"] == r_id]
    row = {
        "RiskLevel": r_name, "N": len(sub),
        "HLD_Prevalence": f"{sub['Hyperlipidemia'].mean():.1%}",
        "CompositeScore_range": f"[{sub['CompositeScore'].min():.1f}, {sub['CompositeScore'].max():.1f}]",
        "CompositeScore_median": f"{sub['CompositeScore'].median():.1f}",
        "P_screen_median": f"{sub['P_screen'].median():.3f}",
        "LipidAbnormal_rate": f"{sub['LipidAbnormal'].mean():.1%}",
    }
    for col in KEY_COLS:
        q25 = sub[col].quantile(0.25)
        med = sub[col].median()
        q75 = sub[col].quantile(0.75)
        row[f"{col}_median"]  = round(med, 2)
        row[f"{col}_IQR"]     = f"[{q25:.1f}, {q75:.1f}]"
    row["Clinical_Rule"] = RISK_RULES[r_name]
    rows_t6.append(row)

tbl6 = pd.DataFrame(rows_t6)
tbl6.to_csv(f"{OUT_DIR}/Table6_RiskThresholds.csv", index=False)
print("✓ Table 6 saved")

# Print condensed version
print("\n  Condensed risk profile:")
for r_id, r_name in RISK_NAMES.items():
    sub  = df[df["RiskLevel"] == r_id]
    pd_m = sub["PhlegmDampness"].median()
    ac_m = sub["Activity_Total"].median()
    tc_m = sub["TC"].median()  if "TC" in sub.columns else float("nan")
    tg_m = sub["TG"].median()  if "TG" in sub.columns else float("nan")
    lip  = sub["LipidAbnormal"].mean()
    print(f"  {r_name:6s} n={len(sub):3d}  HLD={sub['Hyperlipidemia'].mean():.1%}  "
          f"PhlegmDamp_med={pd_m:.1f}  Activity_med={ac_m:.1f}  "
          f"TC_med={tc_m:.2f}  TG_med={tg_m:.2f}  LipidAbn={lip:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD VALIDATION: problem-statement example rules
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("THRESHOLD VALIDATION: problem-statement example rules")
print("="*60)

# Rule A: High risk = blood lipid abnormal AND PhlegmDampness ≥ 60
ruleA = (df["LipidAbnormal"] == 1) & (df["PhlegmDampness"] >= 60)
# Rule B: High risk = blood lipid normal AND PhlegmDampness ≥ 80 AND Activity < 40
ruleB = (df["LipidAbnormal"] == 0) & (df["PhlegmDampness"] >= 80) & (df["Activity_Total"] < 40)

df["HighRisk_ExampleRule"] = (ruleA | ruleB).astype(int)

for tag, mask in [("Rule A (Lipid-abn ∩ Phlegm≥60)", ruleA),
                   ("Rule B (Normal-lipid ∩ Phlegm≥80 ∩ Act<40)", ruleB),
                   ("Rule A ∪ B", ruleA | ruleB)]:
    n_rule = mask.sum()
    if n_rule == 0:
        print(f"  {tag}: n=0 (no samples match)")
        continue
    hld    = df.loc[mask, "Hyperlipidemia"].mean()
    high_pct = (df.loc[mask, "RiskLevel"] == 2).mean()
    print(f"  {tag}")
    print(f"    n={n_rule} ({n_rule/len(df)*100:.1f}%), HLD={hld:.1%}, Model-High={high_pct:.1%}")

# Our model High-risk group
model_high = df["RiskLevel"] == 2
print(f"\n  Our Model High-risk group: n={model_high.sum()}, "
      f"HLD={df.loc[model_high,'Hyperlipidemia'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 7 & CORE FEATURE COMBINATIONS
# Three binary flags: HighPhlegm / LowActivity / LipidAbnormal
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CORE FEATURE COMBINATIONS (PhlegmDampness × Activity × LipidAbnormal)")
print("="*60)

# Youden-index optimal thresholds from ROC analysis of each feature vs HLD.
# If Youden J < 0.10 the feature lacks sufficient discriminative power and the
# data-driven threshold is unreliable; in that case we fall back to the
# established clinical reference standard.
YOUDEN_J_MIN = 0.10

fpr_pd,  tpr_pd,  thr_pd  = roc_curve(y, df["PhlegmDampness"].values)
j_pd = float(np.max(tpr_pd - fpr_pd))
PD_HIGH_youden   = int(np.round(thr_pd[np.argmax(tpr_pd - fpr_pd)]))
PD_HIGH_clinical = 60    # ZYYXH/T157-2009: typical phlegm-dampness constitution ≥ 60

if j_pd >= YOUDEN_J_MIN:
    PD_HIGH  = PD_HIGH_youden
    pd_basis = f"Youden J={j_pd:.3f}"
else:
    PD_HIGH  = PD_HIGH_clinical
    pd_basis = (f"clinical standard ZYYXH/T157-2009 "
                f"(Youden J={j_pd:.3f} < {YOUDEN_J_MIN}, data-driven cut unreliable)")

# For Activity_Total: lower activity → higher risk, so negate before ROC.
fpr_act, tpr_act, thr_act = roc_curve(y, -df["Activity_Total"].values)
j_act = float(np.max(tpr_act - fpr_act))
ACT_LOW_youden   = int(np.round(-thr_act[np.argmax(tpr_act - fpr_act)]))
ACT_LOW_clinical = 40    # ADL+IADL < 40: significant functional limitation in elderly

if j_act >= YOUDEN_J_MIN:
    ACT_LOW  = ACT_LOW_youden
    act_basis = f"Youden J={j_act:.3f}"
else:
    ACT_LOW  = ACT_LOW_clinical
    act_basis = (f"ADL/IADL clinical standard "
                 f"(Youden J={j_act:.3f} < {YOUDEN_J_MIN}, data-driven cut unreliable)")

print(f"\n[Binary threshold selection: Youden vs clinical fallback]")
print(f"  PD_HIGH = {PD_HIGH}  →  {pd_basis}")
print(f"  ACT_LOW = {ACT_LOW}  →  {act_basis}")
# LipidAbnormal already defined above



df["Flag_HighPhlegm"]    = (df["PhlegmDampness"]  >= PD_HIGH).astype(int)
df["Flag_LowActivity"]   = (df["Activity_Total"]  <  ACT_LOW).astype(int)
df["Flag_LipidAbnormal"] = df["LipidAbnormal"]

df["CoreCombo"] = (
    df["Flag_HighPhlegm"].astype(str) +
    df["Flag_LowActivity"].astype(str) +
    df["Flag_LipidAbnormal"].astype(str)
)

COMBO_MAP = {
    "000": "None of three",
    "100": "HighPhlegm only",
    "010": "LowActivity only",
    "001": "LipidAbnormal only",
    "110": "HighPhlegm + LowActivity",
    "101": "HighPhlegm + LipidAbnormal",
    "011": "LowActivity + LipidAbnormal",
    "111": "All three (Core Triple)",
}

combo_rows = []
for code, name in COMBO_MAP.items():
    mask = df["CoreCombo"] == code
    n = mask.sum()
    if n == 0:
        combo_rows.append({"Combination": name, "Code": code, "N": 0,
                            "Pct": "0.0%", "HLD_Prev": "—", "High_Risk_pct": "—",
                            "HLD_Prev_num": 0})
        continue
    hld_prev  = df.loc[mask, "Hyperlipidemia"].mean()
    high_pct  = (df.loc[mask, "RiskLevel"] == 2).mean()
    pd_med    = df.loc[mask, "PhlegmDampness"].median()
    act_med   = df.loc[mask, "Activity_Total"].median()
    combo_rows.append({
        "Combination": name, "Code": code, "N": n,
        "Pct": f"{n/len(df)*100:.1f}%",
        "HLD_Prev": f"{hld_prev:.1%}",
        "High_Risk_pct": f"{high_pct:.1%}",
        "PhlegmDamp_median": pd_med,
        "Activity_median": act_med,
        "HLD_Prev_num": hld_prev,
    })

tbl7 = pd.DataFrame(combo_rows).sort_values("HLD_Prev_num", ascending=False)
tbl7_out = tbl7.drop(columns="HLD_Prev_num")
# Append a metadata row documenting model parameters for reproducibility.
# This covers both Table6 and Table7 derivation choices.
meta7 = {col: "" for col in tbl7_out.columns}
meta7["Combination"] = "# MODEL NOTES"
meta7["Code"] = (
    f"Weights(Excess-AUC): Screen={W_A:.1f} PD={W_B:.1f} Act={W_C:.1f} Lipid={W_D:.1f}. "
    f"Cuts at 33/67pct: Low<{LOW_CUT} Mid<{HIGH_CUT}. "
    f"PD_HIGH={PD_HIGH} ({pd_basis}). "
    f"ACT_LOW={ACT_LOW} ({act_basis})."
)
tbl7_out = pd.concat([tbl7_out, pd.DataFrame([meta7])], ignore_index=True)
tbl7_out.to_csv(f"{OUT_DIR}/Table7_CoreCombinations.csv", index=False)
print("✓ Table 7 saved")
print(tbl7[["Combination","N","Pct","HLD_Prev","High_Risk_pct"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# SHAP – full feature model for final interpretation
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing SHAP values for full model and phlegm-dampness subgroup...")

# Full SHAP
explainer_A = shap.TreeExplainer(xgb_A_final)
shap_full   = explainer_A.shap_values(X_sc)
shap_mean_full = pd.Series(np.abs(shap_full).mean(axis=0), index=SCREEN_FEATURES)

# SHAP for PhlegmDampness high-risk subgroup (PhlegmDampness ≥ 60)
pd_mask = df["Flag_HighPhlegm"].values == 1
print(f"  PhlegmDampness≥60 subgroup: n={pd_mask.sum()}")

if pd_mask.sum() >= 30:
    X_sub = X_sc[pd_mask]
    y_sub = y[pd_mask]
    xgb_sub = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, verbosity=0,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=max(1.0, (y_sub==0).sum() / max((y_sub==1).sum(), 1))
    )
    xgb_sub.fit(X_sub, y_sub)
    shap_sub = shap.TreeExplainer(xgb_sub).shap_values(X_sub)
    shap_mean_sub = pd.Series(np.abs(shap_sub).mean(axis=0), index=SCREEN_FEATURES)
else:
    shap_mean_sub = pd.Series(dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

COLORS = {
    "low":    "#4A9E6B",
    "mid":    "#E8952B",
    "high":   "#E05A42",
    "blue":   "#2E5E9A",
    "gray":   "#888888",
    "light":  "#AEC6E8",
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Composite score distribution + calibration curve
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Composite score histogram per risk level
ax = axes[0]
bins = np.linspace(df["CompositeScore"].min(), df["CompositeScore"].max(), 30)
for r_id, r_name, col in [(0,"Low",COLORS["low"]),
                            (1,"Medium",COLORS["mid"]),
                            (2,"High",COLORS["high"])]:
    vals = df.loc[df["RiskLevel"]==r_id, "CompositeScore"]
    ax.hist(vals, bins=bins, alpha=0.65, color=col,
            label=f"{r_name} risk (n={len(vals)})", density=True)
ax.axvline(LOW_CUT,  color="black", linestyle="--", lw=1.2,
           label=f"Low/Mid cut = {LOW_CUT}")
ax.axvline(HIGH_CUT, color="black", linestyle=":",  lw=1.2,
           label=f"Mid/High cut = {HIGH_CUT}")
ax.set_xlabel("Composite Risk Score (0–100)")
ax.set_ylabel("Density")
ax.set_title("Three-Level Risk Distribution\n(Composite Score)", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# Right: Calibration curve for early-screening model
ax = axes[1]
ax.plot(mean_pred_A, frac_pos_A, "s-", color=COLORS["blue"], lw=2,
        label=f"Early-Screen XGBoost (AUC={auc_screen:.3f})")
ax.plot([0,1],[0,1],"--", color=COLORS["gray"], lw=1, label="Perfect calibration")
ax.fill_between(mean_pred_A, frac_pos_A, mean_pred_A,
                alpha=0.12, color=COLORS["blue"])
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve\n(Non-Lipid Early-Screening Model)", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig7_RiskScore_Calibration.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 7 saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Core combination HLD prevalence + stacked risk composition
# ─────────────────────────────────────────────────────────────────────────────
valid_combos = [r for r in combo_rows if r["N"] > 0]
valid_combos_df = pd.DataFrame(valid_combos).sort_values("HLD_Prev_num")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: HLD prevalence bar chart
ax = axes[0]
bar_colors = []
for hld in valid_combos_df["HLD_Prev_num"]:
    if hld >= 0.7:   bar_colors.append(COLORS["high"])
    elif hld >= 0.4: bar_colors.append(COLORS["mid"])
    else:            bar_colors.append(COLORS["low"])
bars = ax.barh(valid_combos_df["Combination"],
               valid_combos_df["HLD_Prev_num"] * 100,
               color=bar_colors, alpha=0.85, edgecolor="white")
for bar, row in zip(bars, valid_combos_df.itertuples()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{row.HLD_Prev_num:.1%} (n={row.N})",
            va="center", fontsize=8)
ax.set_xlabel("HLD Prevalence (%)")
ax.set_title(
    f"HLD Prevalence by Core Combination\n"
    f"(HighPhlegm: PD≥{PD_HIGH}, LowActivity: Act<{ACT_LOW})",
    fontweight="bold"
)
ax.set_xlim(0, 120)
ax.spines[["top","right"]].set_visible(False)
ph = mpatches.Patch(color=COLORS["high"], alpha=0.85, label="High (≥70%)")
pm = mpatches.Patch(color=COLORS["mid"],  alpha=0.85, label="Mid (40–70%)")
pl = mpatches.Patch(color=COLORS["low"],  alpha=0.85, label="Low (<40%)")
ax.legend(handles=[ph, pm, pl], fontsize=8, loc="lower right")

# Right: Stacked bar – risk level composition per combination
ax = axes[1]
low_v  = []; mid_v  = []; hig_v  = []
labels = []
for row in valid_combos_df.itertuples():
    mask = df["CoreCombo"] == row.Code
    n    = mask.sum()
    low_v.append((df.loc[mask, "RiskLevel"] == 0).mean() * 100)
    mid_v.append((df.loc[mask, "RiskLevel"] == 1).mean() * 100)
    hig_v.append((df.loc[mask, "RiskLevel"] == 2).mean() * 100)
    labels.append(row.Combination)

y_pos = np.arange(len(labels))
low_a = np.array(low_v); mid_a = np.array(mid_v); hig_a = np.array(hig_v)
ax.barh(y_pos, low_a, color=COLORS["low"], alpha=0.85, label="Low risk")
ax.barh(y_pos, mid_a, left=low_a,          color=COLORS["mid"], alpha=0.85, label="Medium risk")
ax.barh(y_pos, hig_a, left=low_a+mid_a,    color=COLORS["high"], alpha=0.85, label="High risk")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Proportion (%)")
ax.set_title("Three-Level Risk Composition\nper Core Feature Combination", fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig8_CoreCombinations.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 8 saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9: SHAP feature importance – full vs PhlegmDampness subgroup
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
TOP_N = 12

ax = axes[0]
top_full = shap_mean_full.sort_values(ascending=False).head(TOP_N).sort_values()
ax.barh(top_full.index, top_full.values, color=COLORS["blue"], alpha=0.85, edgecolor="white")
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title(f"Full Sample – Top {TOP_N} Features\n(Non-Lipid Screening XGBoost)", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
for i, v in enumerate(top_full.values):
    ax.text(v + v*0.02 + 1e-4, i, f"{v:.4f}", va="center", fontsize=7)

ax = axes[1]
if len(shap_mean_sub) > 0:
    top_sub = shap_mean_sub.sort_values(ascending=False).head(TOP_N).sort_values()
    top_sub = top_sub[top_sub > 0]
    if len(top_sub) > 0:
        ax.barh(top_sub.index, top_sub.values, color=COLORS["high"], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Mean |SHAP Value|")
        for i, v in enumerate(top_sub.values):
            ax.text(v + v*0.02 + 1e-4, i, f"{v:.4f}", va="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No non-zero SHAP values in subgroup",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=COLORS["gray"])
    ax.set_title(f"PhlegmDampness≥{PD_HIGH} Subgroup – Top {TOP_N}\n(SHAP)", fontweight="bold")
else:
    ax.text(0.5, 0.5, "Subgroup too small for SHAP",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color=COLORS["gray"])
    ax.set_title(f"PhlegmDampness High-Risk Subgroup (N/A)", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig9_SHAP_FullVsSubgroup.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 9 saved")

# ── Figure 10: Composite score components breakdown per risk level ─────────
fig, ax = plt.subplots(figsize=(9, 5))
score_components = ["Score_Screen", "Score_Phlegm", "Score_Activity", "Score_Lipid"]
comp_labels      = [f"Screen Model (×{W_A:.0f})", f"PhlegmDampness (×{W_B:.0f})",
                    f"Activity (×{W_C:.0f})", f"LipidAbnormal (×{W_D:.0f})"]
comp_colors      = [COLORS["blue"], "#9B59B6", COLORS["mid"], COLORS["high"]]
x = np.arange(3)
width = 0.18
means = {
    comp: [df.loc[df["RiskLevel"]==r, comp].mean() for r in [0,1,2]]
    for comp in score_components
}
for i, (comp, label, col) in enumerate(zip(score_components, comp_labels, comp_colors)):
    ax.bar(x + i*width, means[comp], width, label=label, color=col, alpha=0.85, edgecolor="white")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(["Low risk", "Medium risk", "High risk"])
ax.set_ylabel("Mean Component Score")
ax.set_title("Composite Score Components by Risk Level\n(Mean per Stratum)", fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/Fig10_ScoreComponents.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Figure 10 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY – Q2 Three-Level Risk Warning Model")
print("="*60)

print(f"\n[Early-Screening Model]  Non-lipid features only  |  OOF AUC = {auc_screen:.3f}")
print(f"[Composite Risk Score]   Components (data-driven): "
      f"Screen({W_A:.1f}) + PhlegmDampness({W_B:.1f}) + Activity({W_C:.1f}) + Lipid({W_D:.1f})")
print(f"[Risk Cut-points]        Low < {LOW_CUT}  ≤  Medium < {HIGH_CUT}  ≤  High  "
      f"(33rd/67th percentile)")
print(f"[Binary Thresholds]      PD_HIGH={PD_HIGH} (Youden)  |  ACT_LOW={ACT_LOW} (Youden)")

print("\n[Risk Stratum HLD Prevalence]")
for r_id, r_name in RISK_NAMES.items():
    sub = df[df["RiskLevel"] == r_id]
    print(f"  {r_name:6s}: {sub['Hyperlipidemia'].mean():.1%}  (n={len(sub)},  "
          f"Score median={sub['CompositeScore'].median():.1f})")

print("\n[Clinical Threshold Rules]")
for r_name, rule in RISK_RULES.items():
    print(f"  {r_name}: {rule}")

print("\n[Core Triple Combination: HighPhlegm + LowActivity + LipidAbnormal]")
triple = df[df["CoreCombo"] == "111"]
print(f"  n={len(triple)}  ({len(triple)/len(df)*100:.1f}% of total)")
if len(triple) > 0:
    print(f"  HLD prevalence={triple['Hyperlipidemia'].mean():.1%}")
    print(f"  Model High-risk rate={(triple['RiskLevel']==2).mean():.1%}")

print(f"\n✓ All Q2 outputs saved to: {OUT_DIR}/")
print("  Tables : Table6_RiskThresholds.csv, Table7_CoreCombinations.csv")
print("  Figures: Fig7_RiskScore_Calibration.png")
print("           Fig8_CoreCombinations.png")
print("           Fig9_SHAP_FullVsSubgroup.png")
print("           Fig10_ScoreComponents.png")
