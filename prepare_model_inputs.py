"""
Stage 2 — Make the master dataset actually trainable.
======================================================
Input : master_learning_dataset.csv  (clean, null-free, but still raw/text)
Output: model-ready train/test splits, fully numeric, scaled, encoded.

IMPORTANT DESIGN DECISION (read this):
---------------------------------------
The problem statement's expected output is "recommend next topic," but OULAD
has no ground-truth "student clicked recommendation X and it worked" label —
no recommender ever ran on this historical data. So there is no honest
supervised target for "which topic to recommend next."

Two defensible paths, both supported by this script's output:
  (A) SUPERVISED PROXY TASK: predict `final_result` (Withdrawn/Fail/Pass/
      Distinction) from engagement + score features. This gives you a real,
      gradable, leakage-checked target to train/evaluate a model on, and the
      same feature set doubles as the input to a rule-based or similarity-based
      topic recommender (e.g., "students who ended up like this one engaged
      most with these topics"). This is what `y_target` is set to below.
  (B) UNSUPERVISED / CONTENT-BASED RECOMMENDER: use the per-activity-type
      click columns (clicks_*) directly as a student-topic engagement matrix
      and do similarity-based or collaborative-filtering recommendation —
      no train/test split needed for this path, just the scaled matrix.

Both paths are legitimate to write up under "Explainability/Innovation" —
just be explicit in your report about which one you chose and why the raw
data doesn't support a direct supervised "next topic" label.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

IN_PATH = "/home/claude/work/master_learning_dataset.csv"
OUT = "/home/claude/work"

df = pd.read_csv(IN_PATH)

# ---------------------------------------------------------------------------
# 1. Separate identifiers from features (never train on raw IDs)
# ---------------------------------------------------------------------------
id_cols = ["id_student", "code_module", "code_presentation"]
ids = df[id_cols].copy()

y_target_col = "final_result"          # Path (A) target — see docstring
y = df[y_target_col].copy()

# --- LEAKAGE FIX 1: drop features that encode the outcome by construction ---
# `dropped_out` is date_unregistration.notna() -- 10,063/10,063 rows where it's
# 1 are literally Withdrawn. That's not a predictor, it's the label renamed.
# `engagement_days` / `progress_ratio` / `last_active_day`-derived features are
# computed over each student's FULL observation window -- but for withdrawn
# students that window IS bounded by the withdrawal event itself, so these
# features leak "how the course ended" into "predict how the course ends."
leaky_cols = ["dropped_out", "engagement_days", "progress_ratio"]
X = df.drop(columns=id_cols + [y_target_col] + leaky_cols)

# ---------------------------------------------------------------------------
# 2. Encode categoricals
# ---------------------------------------------------------------------------
# Ordinal columns: real order exists, so map to integers (preserves it,
# unlike one-hot which would throw the ordering away)
ordinal_maps = {
    "highest_education": {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4,
    },
    "age_band": {"0-35": 0, "35-55": 1, "55<=": 2},
    "imd_band": {
        "Unknown": -1, "0-10%": 0, "10-20%": 1, "20-30%": 2, "30-40%": 3,
        "40-50%": 4, "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8,
        "90-100%": 9,
    },
}
for col, mapping in ordinal_maps.items():
    X[col] = X[col].map(mapping)
    # anything unmapped (typo/unexpected category) -> treat as unknown, not NaN
    X[col] = X[col].fillna(-1).astype(int)

# Nominal columns: no inherent order -> one-hot encode
nominal_cols = ["gender", "region", "disability", "last_recommended_topic_proxy"]
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)

# Target encoding (label encode the 4 outcome classes, ordered by severity)
target_map = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}
y_encoded = y.map(target_map)

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
# LEAKAGE FIX 2: split by STUDENT, not by row. 1,209 students take more than
# one course presentation, so a plain random row split puts the same student
# in both train and test -- the model can partly recognize them by their
# demographic/behavioral fingerprint instead of generalizing. GroupShuffleSplit
# guarantees every id_student lands entirely in train OR entirely in test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=ids["id_student"]))

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y_encoded.iloc[train_idx].copy(), y_encoded.iloc[test_idx].copy()
ids_train, ids_test = ids.iloc[train_idx].copy(), ids.iloc[test_idx].copy()

assert set(ids_train["id_student"]) & set(ids_test["id_student"]) == set(), \
    "Student leakage across split!"

# ---------------------------------------------------------------------------
# 4. Scale numeric features (fit ONLY on train, apply to both — this is the
#    single most common leakage bug: never fit a scaler on test data)
# ---------------------------------------------------------------------------
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# don't scale the one-hot / ordinal-flag columns that are effectively binary/discrete small ints
binary_like = [c for c in numeric_cols if X_train[c].nunique() <= 2]
cols_to_scale = [c for c in numeric_cols if c not in binary_like]

scaler = StandardScaler()
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# ---------------------------------------------------------------------------
# 5. Save everything
# ---------------------------------------------------------------------------
X_train.to_csv(f"{OUT}/X_train.csv", index=False)
X_test.to_csv(f"{OUT}/X_test.csv", index=False)
y_train.to_csv(f"{OUT}/y_train.csv", index=False)
y_test.to_csv(f"{OUT}/y_test.csv", index=False)
ids_train.to_csv(f"{OUT}/ids_train.csv", index=False)
ids_test.to_csv(f"{OUT}/ids_test.csv", index=False)
joblib.dump(scaler, f"{OUT}/scaler.joblib")

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("Nulls in X_train:", X_train.isna().sum().sum())
print("Nulls in X_test:", X_test.isna().sum().sum())
print("Target class balance (train):\n", y_train.value_counts(normalize=True).round(3))
print("Target class balance (test):\n", y_test.value_counts(normalize=True).round(3))
print("\nFinal feature columns (", X_train.shape[1], "):\n", list(X_train.columns))
