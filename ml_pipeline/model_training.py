"""
Model Training for Outage Prediction
Trains Random Forest, XGBoost, and LightGBM models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import sys
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 70)
print("MODEL TRAINING FOR OUTAGE PREDICTION")
print("=" * 70)

# Step 1: Load features from MongoDB
print("\n📊 Step 1: Loading features from MongoDB...")
db = get_db()
training_data = list(db.training_data.find())
print(f"   Loaded {len(training_data):,} records\n")

# Convert to DataFrame
df = pd.DataFrame(training_data)

# Step 2: Prepare features and target
print("🎯 Step 2: Preparing features and target...")
feature_cols = [
    'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'season',
    'avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages',
    'weather_event_count', 'total_property_damage', 'avg_magnitude', 'total_injuries', 'total_deaths',
    'latest_population'
]

X = df[feature_cols]
y = df['target']

print(f"   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]:,}")
print(f"   Target distribution:")
print(f"      No outage (0): {(y == 0).sum():,} ({(y == 0).mean():.1%})")
print(f"      Outage (1): {(y == 1).sum():,} ({(y == 1).mean():.1%})\n")

# Step 3: Train/Test/Validation Split
print("✂️  Step 3: Creating train/test/validation splits...")
# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Second split: 15% test, 15% validation
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
print(f"   Train class distribution: {y_train.value_counts().to_dict()}\n")

# Step 3b: Handle Class Imbalance with Smart Sampling Strategy
print("⚖️  Step 3b: Handling class imbalance with smart sampling...")
print(f"   Original train distribution: 0={len(y_train[y_train==0]):,}, 1={len(y_train[y_train==1]):,}")
print(f"   Original ratio: 1:{len(y_train[y_train==0]) / len(y_train[y_train==1]):.0f}")

# Strategy: First undersample majority to 1:10, then SMOTE minority to 1:3
# This avoids creating too many synthetic samples (overfitting)

# Step 1: Undersample majority class to 10x minority (1:10 ratio)
n_minority = len(y_train[y_train==1])
n_majority_target = n_minority * 10  # Target 10x minority

under = RandomUnderSampler(sampling_strategy={0: n_majority_target, 1: n_minority}, random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

print(f"   After undersampling: 0={len(y_train_under[y_train_under==0]):,}, 1={len(y_train_under[y_train_under==1]):,}")
print(f"   Ratio: 1:{len(y_train_under[y_train_under==0]) / len(y_train_under[y_train_under==1]):.1f}")

# Step 2: SMOTE minority class to 1:3 ratio (less synthetic samples = less overfitting)
smote = SMOTE(sampling_strategy=0.33, random_state=42)  # Minority will be 33% of majority (1:3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_under, y_train_under)

print(f"   After SMOTE: 0={len(y_train_balanced[y_train_balanced==0]):,}, 1={len(y_train_balanced[y_train_balanced==1]):,}")
print(f"   Final ratio: 1:{len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1]):.1f}")
print(f"   Minority class: {len(y_train_balanced[y_train_balanced==1])/len(y_train_balanced):.1%} of total\n")

# Step 4: Train Random Forest
print("🌲 Step 4: Training Random Forest on balanced data...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight=None  # Already balanced with SMOTE
)
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on original test set
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print(f"   Accuracy:  {rf_acc:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall:    {rf_recall:.4f}")
print(f"   F1 Score:  {rf_f1:.4f}")
print(f"   ROC-AUC:   {rf_auc:.4f}")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 Features:")
for idx, row in rf_importance.head(5).iterrows():
    print(f"      {row['feature']:30s}: {row['importance']:.4f}")
print()

# Step 5: Train XGBoost
print("🚀 Step 5: Training XGBoost on balanced data...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=1  # Already balanced with SMOTE
)
xgb_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on original test set
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb, zero_division=0)
xgb_recall = recall_score(y_test, y_pred_xgb, zero_division=0)
xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)

print(f"   Accuracy:  {xgb_acc:.4f}")
print(f"   Precision: {xgb_precision:.4f}")
print(f"   Recall:    {xgb_recall:.4f}")
print(f"   F1 Score:  {xgb_f1:.4f}")
print(f"   ROC-AUC:   {xgb_auc:.4f}\n")

# Step 6: Train LightGBM
print("💡 Step 6: Training LightGBM on balanced data...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    class_weight=None,  # Already balanced with SMOTE
    verbose=-1
)
lgb_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on original test set
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

lgb_acc = accuracy_score(y_test, y_pred_lgb)
lgb_precision = precision_score(y_test, y_pred_lgb, zero_division=0)
lgb_recall = recall_score(y_test, y_pred_lgb, zero_division=0)
lgb_f1 = f1_score(y_test, y_pred_lgb, zero_division=0)
lgb_auc = roc_auc_score(y_test, y_prob_lgb)

print(f"   Accuracy:  {lgb_acc:.4f}")
print(f"   Precision: {lgb_precision:.4f}")
print(f"   Recall:    {lgb_recall:.4f}")
print(f"   F1 Score:  {lgb_f1:.4f}")
print(f"   ROC-AUC:   {lgb_auc:.4f}\n")

# Step 7: Model Comparison
print("📊 Step 7: Model Comparison")
print("=" * 100)
print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-" * 100)
print(f"{'Random Forest':<15} {rf_acc:<12.4f} {rf_precision:<12.4f} {rf_recall:<12.4f} {rf_f1:<12.4f} {rf_auc:<12.4f}")
print(f"{'XGBoost':<15} {xgb_acc:<12.4f} {xgb_precision:<12.4f} {xgb_recall:<12.4f} {xgb_f1:<12.4f} {xgb_auc:<12.4f}")
print(f"{'LightGBM':<15} {lgb_acc:<12.4f} {lgb_precision:<12.4f} {lgb_recall:<12.4f} {lgb_f1:<12.4f} {lgb_auc:<12.4f}")
print("=" * 100)

# Select best model based on F1 score (better for imbalanced data than accuracy)
best_model_name = max(
    [('Random Forest', rf_f1), ('XGBoost', xgb_f1), ('LightGBM', lgb_f1)],
    key=lambda x: x[1]
)[0]
print(f"\n🏆 Best Model (by F1 Score): {best_model_name}")
print(f"   Note: F1 score is better than accuracy for imbalanced datasets\n")

# Step 8: Save models
print("💾 Step 8: Saving models...")
models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(rf_model, models_dir / "random_forest.pkl")
joblib.dump(xgb_model, models_dir / "xgboost.pkl")
joblib.dump(lgb_model, models_dir / "lightgbm.pkl")

# Save feature importance
rf_importance.to_csv(models_dir / "feature_importance.csv", index=False)

print(f"   ✅ Saved 3 models to {models_dir}")
print(f"   ✅ Saved feature importance to feature_importance.csv\n")

# Step 9: Save predictions to MongoDB
print("💾 Step 9: Saving predictions to MongoDB...")
# Use best model for predictions
if best_model_name == 'Random Forest':
    best_model = rf_model
elif best_model_name == 'XGBoost':
    best_model = xgb_model
else:
    best_model = lgb_model

# Predict on all data
y_pred_all = best_model.predict(X)
y_prob_all = best_model.predict_proba(X)[:, 1]

# Add predictions to dataframe
df['predicted_outage'] = y_pred_all
df['outage_probability'] = y_prob_all
df['model_used'] = best_model_name

# Save to MongoDB
predictions = df[['county_fips', 'date', 'predicted_outage', 'outage_probability', 'model_used']].to_dict('records')
db.predictions.delete_many({})
db.predictions.insert_many(predictions)

print(f"   ✅ Saved {len(predictions):,} predictions to 'predictions' collection\n")

print("=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModels saved to: {models_dir}")
print(f"Feature importance saved to: {models_dir / 'feature_importance.csv'}")
print(f"Predictions saved to MongoDB 'predictions' collection")
print(f"\nNext: Launch dashboard to see results!")
print(f"   cd app && streamlit run main.py")
