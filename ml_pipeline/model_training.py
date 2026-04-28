"""
Model Training for Outage Prediction
Trains Random Forest, XGBoost, and LightGBM models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import sys

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
print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%})\n")

# Step 4: Train Random Forest
print("🌲 Step 4: Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print(f"   Accuracy: {rf_acc:.4f}")
print(f"   ROC-AUC:  {rf_auc:.4f}")

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
print("🚀 Step 5: Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)

print(f"   Accuracy: {xgb_acc:.4f}")
print(f"   ROC-AUC:  {xgb_auc:.4f}\n")

# Step 6: Train LightGBM
print("💡 Step 6: Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
lgb_model.fit(X_train, y_train)

# Evaluate
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

lgb_acc = accuracy_score(y_test, y_pred_lgb)
lgb_auc = roc_auc_score(y_test, y_prob_lgb)

print(f"   Accuracy: {lgb_acc:.4f}")
print(f"   ROC-AUC:  {lgb_auc:.4f}\n")

# Step 7: Model Comparison
print("📊 Step 7: Model Comparison")
print("=" * 70)
print(f"{'Model':<20} {'Accuracy':<15} {'ROC-AUC':<15}")
print("-" * 70)
print(f"{'Random Forest':<20} {rf_acc:<15.4f} {rf_auc:<15.4f}")
print(f"{'XGBoost':<20} {xgb_acc:<15.4f} {xgb_auc:<15.4f}")
print(f"{'LightGBM':<20} {lgb_acc:<15.4f} {lgb_auc:<15.4f}")
print("=" * 70)

# Select best model
best_model_name = max(
    [('Random Forest', rf_auc), ('XGBoost', xgb_auc), ('LightGBM', lgb_auc)],
    key=lambda x: x[1]
)[0]
print(f"\n🏆 Best Model: {best_model_name}\n")

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
