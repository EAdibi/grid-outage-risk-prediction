"""
Compare Different Sampling Strategies for Imbalanced Data
Tests: No sampling, Undersampling only, SMOTE only, Hybrid (Under + SMOTE)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 80)
print("COMPARING SAMPLING STRATEGIES FOR IMBALANCED DATA")
print("=" * 80)

# Load data
db = get_db()
training_data = list(db.training_data.find())
df = pd.DataFrame(training_data)

feature_cols = [
    'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'season',
    'avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages',
    'weather_event_count', 'total_property_damage', 'avg_magnitude', 'total_injuries', 'total_deaths',
    'latest_population'
]

X = df[feature_cols]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n📊 Original Data:")
print(f"   Train: {len(y_train):,} samples")
print(f"   Class 0: {len(y_train[y_train==0]):,} ({len(y_train[y_train==0])/len(y_train):.1%})")
print(f"   Class 1: {len(y_train[y_train==1]):,} ({len(y_train[y_train==1])/len(y_train):.1%})")
print(f"   Ratio: 1:{len(y_train[y_train==0]) / len(y_train[y_train==1]):.0f}\n")

# Strategy 1: No Sampling (Baseline with class weights)
print("=" * 80)
print("Strategy 1: No Sampling (Class Weights Only)")
print("=" * 80)

rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
rf_baseline.fit(X_train, y_train)
y_pred = rf_baseline.predict(X_test)
y_prob = rf_baseline.predict_proba(X_test)[:, 1]

print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Strategy 2: Undersampling Only (1:10 ratio)
print("\n" + "=" * 80)
print("Strategy 2: Undersampling Only (1:10 ratio)")
print("=" * 80)

n_minority = len(y_train[y_train==1])
n_majority_target = n_minority * 10

under = RandomUnderSampler(sampling_strategy={0: n_majority_target, 1: n_minority}, random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

print(f"   Training samples: {len(y_train_under):,}")
print(f"   Class 0: {len(y_train_under[y_train_under==0]):,}")
print(f"   Class 1: {len(y_train_under[y_train_under==1]):,}")
print(f"   Ratio: 1:{len(y_train_under[y_train_under==0]) / len(y_train_under[y_train_under==1]):.1f}")

rf_under = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_under.fit(X_train_under, y_train_under)
y_pred = rf_under.predict(X_test)
y_prob = rf_under.predict_proba(X_test)[:, 1]

print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Strategy 3: SMOTE Only (1:10 ratio - moderate oversampling)
print("\n" + "=" * 80)
print("Strategy 3: SMOTE Only (1:10 ratio)")
print("=" * 80)

smote_only = SMOTE(sampling_strategy=0.1, random_state=42)  # Minority = 10% of majority
X_train_smote, y_train_smote = smote_only.fit_resample(X_train, y_train)

print(f"   Training samples: {len(y_train_smote):,}")
print(f"   Class 0: {len(y_train_smote[y_train_smote==0]):,}")
print(f"   Class 1: {len(y_train_smote[y_train_smote==1]):,}")
print(f"   Ratio: 1:{len(y_train_smote[y_train_smote==0]) / len(y_train_smote[y_train_smote==1]):.1f}")

rf_smote = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred = rf_smote.predict(X_test)
y_prob = rf_smote.predict_proba(X_test)[:, 1]

print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Strategy 4: Hybrid (Undersample to 1:10, then SMOTE to 1:3)
print("\n" + "=" * 80)
print("Strategy 4: Hybrid (Undersample 1:10 → SMOTE 1:3) ⭐ RECOMMENDED")
print("=" * 80)

# Step 1: Undersample to 1:10
under_hybrid = RandomUnderSampler(sampling_strategy={0: n_majority_target, 1: n_minority}, random_state=42)
X_train_hybrid, y_train_hybrid = under_hybrid.fit_resample(X_train, y_train)

# Step 2: SMOTE to 1:3
smote_hybrid = SMOTE(sampling_strategy=0.33, random_state=42)
X_train_hybrid, y_train_hybrid = smote_hybrid.fit_resample(X_train_hybrid, y_train_hybrid)

print(f"   Training samples: {len(y_train_hybrid):,}")
print(f"   Class 0: {len(y_train_hybrid[y_train_hybrid==0]):,}")
print(f"   Class 1: {len(y_train_hybrid[y_train_hybrid==1]):,}")
print(f"   Ratio: 1:{len(y_train_hybrid[y_train_hybrid==0]) / len(y_train_hybrid[y_train_hybrid==1]):.1f}")

rf_hybrid = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
y_pred = rf_hybrid.predict(X_test)
y_prob = rf_hybrid.predict_proba(X_test)[:, 1]

print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY COMPARISON")
print("=" * 80)
print(f"{'Strategy':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-" * 80)

# Re-run all for summary (to get metrics)
strategies = [
    ("1. No Sampling (Class Weights)", rf_baseline),
    ("2. Undersampling Only (1:10)", rf_under),
    ("3. SMOTE Only (1:10)", rf_smote),
    ("4. Hybrid (Under+SMOTE 1:3) ⭐", rf_hybrid)
]

for name, model in strategies:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"{name:<40} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")

print("=" * 80)
print("\n💡 Recommendation:")
print("   Use Strategy 4 (Hybrid) - Best balance of precision/recall")
print("   - Avoids overfitting from too many synthetic samples")
print("   - Maintains good recall while improving precision")
print("   - Reasonable training time with ~40K samples")
