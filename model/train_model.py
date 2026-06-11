"""
RailPulse - Model Training
Trains XGBoost classifiers for overcrowding and ticketless risk prediction.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

print("🧠 RailPulse — Model Training Pipeline")
print("=" * 50)

# ─── Load Data ────────────────────────────────────────────────────────────────
print("\n📂 Loading training data...")
df = pd.read_csv("data/train_data.csv")
print(f"   Loaded {len(df):,} records")

# ─── Feature Engineering ──────────────────────────────────────────────────────
print("\n⚙️  Engineering features...")

# Encode coach type
coach_type_map = {"SL": 0, "3A": 1, "2A": 2, "1A": 3}
df["coach_type_enc"] = df["coach_type"].map(coach_type_map)

# Encode day type
day_type_map = {"weekday": 0, "weekend": 1, "festival": 2}
df["day_type_enc"] = df["day_type"].map(day_type_map).fillna(0)

# Extract coach number from coach_id
df["coach_number"] = df["coach_id"].str.extract(r"(\d+)").astype(int)

# Route popularity risk factor
df["route_risk"] = df["route_popularity"] * df["historical_avg_occupancy"]

# Combined pressure score
df["pressure_score"] = (
    df["occupancy_ratio"] * 0.4 +
    df["overbooking_index"] * 0.25 +
    df["event_pressure_score"] * 0.15 +
    df["is_holiday"] * 0.1 +
    df["is_festival_surge"] * 0.1
)

# Ticketless risk composite
df["ticketless_pressure"] = (
    (df["past_ticketless_incidents"] / df["total_seats"]) * 0.5 +
    df["occupancy_ratio"] * 0.3 +
    (df["coach_type"] == "SL").astype(int) * 0.2
)

# ─── Feature Sets ─────────────────────────────────────────────────────────────
OVERCROWD_FEATURES = [
    "day_of_week", "is_holiday", "is_festival_surge", "is_weekend",
    "event_nearby", "event_pressure_score",
    "occupancy_ratio", "overbooking_index", "cancellation_rate",
    "historical_avg_occupancy", "route_popularity",
    "coach_type_enc", "coach_number", "day_type_enc",
    "route_risk", "pressure_score",
]

TICKETLESS_FEATURES = [
    "day_of_week", "is_holiday", "is_festival_surge", "is_weekend",
    "event_nearby",
    "occupancy_ratio", "overbooking_index",
    "past_ticketless_incidents", "cancellation_rate",
    "historical_avg_occupancy", "route_popularity",
    "coach_type_enc", "coach_number",
    "ticketless_pressure",
]

print(f"   Overcrowding features: {len(OVERCROWD_FEATURES)}")
print(f"   Ticketless features:   {len(TICKETLESS_FEATURES)}")

# ─── Train Overcrowding Model ─────────────────────────────────────────────────
print("\n🔴 Training Overcrowding Risk Model (XGBoost)...")

X_oc = df[OVERCROWD_FEATURES]
y_oc = df["overcrowding_label"]

X_oc_train, X_oc_test, y_oc_train, y_oc_test = train_test_split(
    X_oc, y_oc, test_size=0.2, random_state=42, stratify=y_oc
)

scaler = StandardScaler()
X_oc_train_s = scaler.fit_transform(X_oc_train)
X_oc_test_s = scaler.transform(X_oc_test)

overcrowd_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
overcrowd_model.fit(X_oc_train_s, y_oc_train, eval_set=[(X_oc_test_s, y_oc_test)], verbose=50)

oc_preds = overcrowd_model.predict(X_oc_test_s)
oc_proba = overcrowd_model.predict_proba(X_oc_test_s)[:, 1]

print(f"\n   Overcrowding Model — AUC: {roc_auc_score(y_oc_test, oc_proba):.4f}")
print(classification_report(y_oc_test, oc_preds, target_names=["Low Risk", "High Risk"]))

# ─── Train Ticketless Model ───────────────────────────────────────────────────
print("\n⚠️  Training Ticketless Risk Model (XGBoost)...")

X_tl = df[TICKETLESS_FEATURES]
y_tl = df["ticketless_label"]

X_tl_train, X_tl_test, y_tl_train, y_tl_test = train_test_split(
    X_tl, y_tl, test_size=0.2, random_state=42, stratify=y_tl
)

scaler_tl = StandardScaler()
X_tl_train_s = scaler_tl.fit_transform(X_tl_train)
X_tl_test_s = scaler_tl.transform(X_tl_test)

ticketless_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
ticketless_model.fit(X_tl_train_s, y_tl_train, eval_set=[(X_tl_test_s, y_tl_test)], verbose=50)

tl_preds = ticketless_model.predict(X_tl_test_s)
tl_proba = ticketless_model.predict_proba(X_tl_test_s)[:, 1]

print(f"\n   Ticketless Model — AUC: {roc_auc_score(y_tl_test, tl_proba):.4f}")
print(classification_report(y_tl_test, tl_preds, target_names=["Low Risk", "High Risk"]))

# ─── Feature Importance ───────────────────────────────────────────────────────
print("\n📈 Top Feature Importances (Overcrowding):")
fi = pd.Series(overcrowd_model.feature_importances_, index=OVERCROWD_FEATURES)
for feat, imp in fi.sort_values(ascending=False).head(8).items():
    bar = "█" * int(imp * 100)
    print(f"   {feat:<35} {imp:.4f} {bar}")

# ─── Save Artifacts ───────────────────────────────────────────────────────────
print("\n💾 Saving model artifacts...")
os.makedirs("model", exist_ok=True)

with open("model/overcrowd_model.pkl", "wb") as f:
    pickle.dump(overcrowd_model, f)

with open("model/ticketless_model.pkl", "wb") as f:
    pickle.dump(ticketless_model, f)

with open("model/scaler_oc.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/scaler_tl.pkl", "wb") as f:
    pickle.dump(scaler_tl, f)

# Save feature lists for inference
import json
model_meta = {
    "overcrowd_features": OVERCROWD_FEATURES,
    "ticketless_features": TICKETLESS_FEATURES,
    "coach_type_map": coach_type_map,
    "trained_at": pd.Timestamp.now().isoformat(),
    "train_samples": len(X_oc_train),
    "overcrowd_auc": round(roc_auc_score(y_oc_test, oc_proba), 4),
    "ticketless_auc": round(roc_auc_score(y_tl_test, tl_proba), 4),
}
with open("model/model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print("   ✅ model/overcrowd_model.pkl")
print("   ✅ model/ticketless_model.pkl")
print("   ✅ model/scaler_oc.pkl")
print("   ✅ model/scaler_tl.pkl")
print("   ✅ model/model_meta.json")
print("\n🎉 Training complete!")
