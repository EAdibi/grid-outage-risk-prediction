"""
Ensemble Model Training - Stacking RF + XGBoost + LightGBM
Goal: Improve precision while maintaining high recall (threshold = 0.5)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 80)
print("ENSEMBLE MODEL TRAINING - STACKING APPROACH")
print("=" * 80)
print("Goal: Improve precision while maintaining high recall (>90%)")
print("Strategy: Stack RF + XGBoost + LightGBM with Logistic Regression meta-learner")
print("=" * 80)

# Load data
print("\n📊 Step 1: Loading features from MongoDB...")
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

print(f"   Loaded {len(df):,} records")
print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution: 0={len(y[y==0]):,} ({len(y[y==0])/len(y):.1%}), 1={len(y[y==1]):,} ({len(y[y==1])/len(y):.1%})")

# Split data
print("\n✂️  Step 2: Creating train/test/validation splits...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")
print(f"   Val:   {len(X_val):,} samples")

# Smart sampling
print("\n⚖️  Step 3: Applying smart sampling (Under 1:10 → SMOTE 1:3)...")
n_minority = len(y_train[y_train==1])
n_majority_target = n_minority * 10

under = RandomUnderSampler(sampling_strategy={0: n_majority_target, 1: n_minority}, random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

smote = SMOTE(sampling_strategy=0.33, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_under, y_train_under)

print(f"   Balanced: 0={len(y_train_balanced[y_train_balanced==0]):,}, 1={len(y_train_balanced[y_train_balanced==1]):,}")
print(f"   Ratio: 1:{len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1]):.1f}")

# ============================================================================
# APPROACH 1: INDIVIDUAL MODELS (BASELINE)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 1: INDIVIDUAL MODELS (BASELINE)")
print("=" * 80)

# Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print(f"   Precision: {rf_precision:.4f} | Recall: {rf_recall:.4f} | F1: {rf_f1:.4f} | ROC-AUC: {rf_auc:.4f}")

# XGBoost
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)

print(f"   Precision: {xgb_precision:.4f} | Recall: {xgb_recall:.4f} | F1: {xgb_f1:.4f} | ROC-AUC: {xgb_auc:.4f}")

# LightGBM
print("\n💡 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

lgb_precision = precision_score(y_test, y_pred_lgb, zero_division=0)
lgb_recall = recall_score(y_test, y_pred_lgb, zero_division=0)
lgb_f1 = f1_score(y_test, y_pred_lgb, zero_division=0)
lgb_auc = roc_auc_score(y_test, y_prob_lgb)

print(f"   Precision: {lgb_precision:.4f} | Recall: {lgb_recall:.4f} | F1: {lgb_f1:.4f} | ROC-AUC: {lgb_auc:.4f}")

# ============================================================================
# APPROACH 2: STACKING ENSEMBLE
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 2: STACKING ENSEMBLE ⭐")
print("=" * 80)

print("\n🎯 Building stacking ensemble...")
print("   Base models: Random Forest + XGBoost + LightGBM")
print("   Meta-learner: Logistic Regression with balanced class weights")

# Define base models
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )),
    ('xgb', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
]

# Meta-learner with class weights to maintain recall
meta_learner = LogisticRegression(
    class_weight='balanced',  # Prioritize recall
    max_iter=1000,
    random_state=42
)

# Create stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,  # 5-fold cross-validation for meta-features
    n_jobs=-1
)

print("\n🔄 Training stacking ensemble (this may take a few minutes)...")
stacking_model.fit(X_train_balanced, y_train_balanced)

print("✅ Ensemble training complete!")

# Evaluate ensemble
print("\n📊 Evaluating ensemble on test set...")
y_pred_ensemble = stacking_model.predict(X_test)
y_prob_ensemble = stacking_model.predict_proba(X_test)[:, 1]

ensemble_precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
ensemble_recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
ensemble_auc = roc_auc_score(y_test, y_prob_ensemble)

print(f"   Precision: {ensemble_precision:.4f} | Recall: {ensemble_recall:.4f} | F1: {ensemble_f1:.4f} | ROC-AUC: {ensemble_auc:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("📊 PERFORMANCE COMPARISON (Threshold = 0.5)")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble ⭐'],
    'Precision': [rf_precision, xgb_precision, lgb_precision, ensemble_precision],
    'Recall': [rf_recall, xgb_recall, lgb_recall, ensemble_recall],
    'F1': [rf_f1, xgb_f1, lgb_f1, ensemble_f1],
    'ROC-AUC': [rf_auc, xgb_auc, lgb_auc, ensemble_auc]
})

print("\n" + results.to_string(index=False))

# Calculate improvements
best_individual_f1 = max(rf_f1, xgb_f1, lgb_f1)
f1_improvement = ((ensemble_f1 - best_individual_f1) / best_individual_f1) * 100

best_individual_precision = max(rf_precision, xgb_precision, lgb_precision)
precision_improvement = ((ensemble_precision - best_individual_precision) / best_individual_precision) * 100

print("\n" + "=" * 80)
print("💡 ENSEMBLE IMPROVEMENTS")
print("=" * 80)
print(f"   Precision: {best_individual_precision:.4f} → {ensemble_precision:.4f} ({precision_improvement:+.1f}%)")
print(f"   Recall:    {xgb_recall:.4f} → {ensemble_recall:.4f} ({((ensemble_recall - xgb_recall) / xgb_recall) * 100:+.1f}%)")
print(f"   F1 Score:  {best_individual_f1:.4f} → {ensemble_f1:.4f} ({f1_improvement:+.1f}%)")
print(f"   ROC-AUC:   {xgb_auc:.4f} → {ensemble_auc:.4f} ({((ensemble_auc - xgb_auc) / xgb_auc) * 100:+.1f}%)")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)

models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(exist_ok=True)

# Save individual models
joblib.dump(rf_model, models_dir / "random_forest.pkl")
joblib.dump(xgb_model, models_dir / "xgboost.pkl")
joblib.dump(lgb_model, models_dir / "lightgbm.pkl")

# Save ensemble
joblib.dump(stacking_model, models_dir / "ensemble_stacking.pkl")

print(f"   ✅ Saved all models to {models_dir}")

# Save predictions with ensemble
print("\n💾 Saving ensemble predictions to MongoDB...")
df['predicted_outage'] = stacking_model.predict(X)
df['outage_probability'] = stacking_model.predict_proba(X)[:, 1]
df['model_used'] = 'Stacking Ensemble (RF+XGB+LGB)'

predictions = df[['county_fips', 'date', 'predicted_outage', 'outage_probability', 'model_used']].to_dict('records')
db.predictions.delete_many({})
db.predictions.insert_many(predictions)

print(f"   ✅ Saved {len(predictions):,} predictions to 'predictions' collection")

# ============================================================================
# DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📋 DETAILED CLASSIFICATION REPORT (Ensemble)")
print("=" * 80)
print(classification_report(y_test, y_pred_ensemble, target_names=['No Outage', 'Outage']))

print("\n" + "=" * 80)
print("✅ ENSEMBLE MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎯 Final Performance (Threshold = 0.5):")
print(f"   Precision: {ensemble_precision:.2%} - {int(ensemble_precision * 100)} out of 100 warnings are real")
print(f"   Recall:    {ensemble_recall:.2%} - Catches {int(ensemble_recall * 100)} out of 100 actual outages")
print(f"   F1 Score:  {ensemble_f1:.2%} - Balanced metric")
print(f"   ROC-AUC:   {ensemble_auc:.2%} - Discrimination ability")

if ensemble_recall >= 0.90:
    print(f"\n✅ SUCCESS: Recall ≥ 90% maintained! ({ensemble_recall:.1%})")
else:
    print(f"\n⚠️  WARNING: Recall dropped below 90% ({ensemble_recall:.1%})")

print(f"\nNext: Launch dashboard to see results!")
print(f"   cd app && streamlit run main.py")
