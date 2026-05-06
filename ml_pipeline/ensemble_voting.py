"""
Ensemble Model - Weighted Voting Approach
Goal: Improve precision while maintaining recall >90% using soft voting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
print("ENSEMBLE MODEL - WEIGHTED VOTING")
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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Smart sampling
n_minority = len(y_train[y_train==1])
under = RandomUnderSampler(sampling_strategy={0: n_minority * 10, 1: n_minority}, random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)
smote = SMOTE(sampling_strategy=0.33, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_under, y_train_under)

print(f"\n📊 Training samples: {len(X_train_balanced):,}")
print(f"   Ratio: 1:{len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1]):.1f}")

# ============================================================================
# WEIGHTED VOTING ENSEMBLE
# ============================================================================
print("\n" + "=" * 80)
print("WEIGHTED VOTING ENSEMBLE")
print("=" * 80)

# Define models with different weights based on individual performance
# XGBoost had best F1 (0.4578), so give it highest weight
models = [
    ('rf', RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )),
    ('xgb', xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    ))
]

# Test different weighting schemes
weighting_schemes = {
    'Equal Weights': [1, 1, 1],
    'XGB Focused': [1, 2, 1],  # Give XGB double weight
    'Best Two': [1, 2, 0],     # Only RF and XGB
    'XGB Heavy': [1, 3, 1],    # Give XGB triple weight
}

results = []

for scheme_name, weights in weighting_schemes.items():
    print(f"\n🔬 Testing: {scheme_name} (weights: {weights})")
    
    voting_model = VotingClassifier(
        estimators=models,
        voting='soft',  # Use probability averaging
        weights=weights,
        n_jobs=-1
    )
    
    voting_model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = voting_model.predict(X_test)
    y_prob = voting_model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Scheme': scheme_name,
        'Weights': str(weights),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📊 VOTING ENSEMBLE COMPARISON")
print("=" * 80)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Find best scheme
best_scheme = df_results.loc[df_results['F1'].idxmax()]
print(f"\n🏆 Best Scheme: {best_scheme['Scheme']}")
print(f"   Weights: {best_scheme['Weights']}")
print(f"   Precision: {best_scheme['Precision']:.4f}")
print(f"   Recall: {best_scheme['Recall']:.4f}")
print(f"   F1: {best_scheme['F1']:.4f}")
print(f"   ROC-AUC: {best_scheme['ROC-AUC']:.4f}")

# Train final model with best weights
print("\n" + "=" * 80)
print("TRAINING FINAL ENSEMBLE")
print("=" * 80)

best_weights = weighting_schemes[best_scheme['Scheme']]
print(f"\nUsing weights: {best_weights}")

final_ensemble = VotingClassifier(
    estimators=models,
    voting='soft',
    weights=best_weights,
    n_jobs=-1
)

final_ensemble.fit(X_train_balanced, y_train_balanced)

# Save
models_dir = Path(__file__).parent.parent / "models"
joblib.dump(final_ensemble, models_dir / "ensemble_voting.pkl")
print(f"\n✅ Saved ensemble to {models_dir / 'ensemble_voting.pkl'}")

# Save predictions
df['predicted_outage'] = final_ensemble.predict(X)
df['outage_probability'] = final_ensemble.predict_proba(X)[:, 1]
df['model_used'] = f'Voting Ensemble ({best_scheme["Scheme"]})'

predictions = df[['county_fips', 'date', 'predicted_outage', 'outage_probability', 'model_used']].to_dict('records')
db.predictions.delete_many({})
db.predictions.insert_many(predictions)

print(f"✅ Saved {len(predictions):,} predictions to MongoDB")

print("\n" + "=" * 80)
print("✅ VOTING ENSEMBLE COMPLETE!")
print("=" * 80)
