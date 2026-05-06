"""
Advanced Model Improvements for Outage Prediction
Explores: Feature engineering, advanced sampling, ensembles, threshold tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 80)
print("ADVANCED MODEL IMPROVEMENTS")
print("=" * 80)

# Load data
print("\n📊 Loading data from MongoDB...")
db = get_db()
training_data = list(db.training_data.find())
df = pd.DataFrame(training_data)

# Base features
base_features = [
    'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'season',
    'avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages',
    'weather_event_count', 'total_property_damage', 'avg_magnitude', 'total_injuries', 'total_deaths',
    'latest_population'
]

print(f"   Loaded {len(df):,} samples")
print(f"   Base features: {len(base_features)}")

# ============================================================================
# IMPROVEMENT 1: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("IMPROVEMENT 1: ADVANCED FEATURE ENGINEERING")
print("=" * 80)

df_enhanced = df.copy()

# 1.1 Interaction Features (capture relationships)
print("\n1.1 Creating interaction features...")
df_enhanced['weather_x_population'] = df_enhanced['weather_event_count'] * df_enhanced['latest_population'] / 100000
df_enhanced['damage_per_capita'] = df_enhanced['total_property_damage'] / (df_enhanced['latest_population'] + 1)
df_enhanced['customers_per_outage'] = df_enhanced['avg_customers_affected'] / (df_enhanced['total_historical_outages'] + 1)
df_enhanced['severity_score'] = (
    df_enhanced['avg_magnitude'] * 
    df_enhanced['weather_event_count'] * 
    (df_enhanced['total_injuries'] + df_enhanced['total_deaths'] + 1)
)

# 1.2 Temporal Features (capture patterns)
print("1.2 Creating temporal pattern features...")
df_enhanced['is_summer'] = df_enhanced['season'].isin([2]).astype(int)  # Summer
df_enhanced['is_winter'] = df_enhanced['season'].isin([4]).astype(int)  # Winter
df_enhanced['is_storm_season'] = df_enhanced['month'].isin([6, 7, 8, 9]).astype(int)  # Jun-Sep
df_enhanced['quarter'] = ((df_enhanced['month'] - 1) // 3) + 1

# 1.3 Risk Indicators (domain knowledge)
print("1.3 Creating risk indicator features...")
df_enhanced['high_risk_county'] = (df_enhanced['total_historical_outages'] > df_enhanced['total_historical_outages'].quantile(0.75)).astype(int)
df_enhanced['extreme_weather'] = (df_enhanced['weather_event_count'] > 0).astype(int)
df_enhanced['high_population_density'] = (df_enhanced['latest_population'] > df_enhanced['latest_population'].quantile(0.75)).astype(int)

# 1.4 Polynomial Features (capture non-linear relationships)
print("1.4 Creating polynomial features...")
df_enhanced['outages_squared'] = df_enhanced['total_historical_outages'] ** 2
df_enhanced['population_log'] = np.log1p(df_enhanced['latest_population'])
df_enhanced['damage_log'] = np.log1p(df_enhanced['total_property_damage'])

enhanced_features = [col for col in df_enhanced.columns if col not in ['_id', 'county_fips', 'date', 'target']]
print(f"\n   Total features: {len(base_features)} → {len(enhanced_features)} (+{len(enhanced_features) - len(base_features)})")

# ============================================================================
# IMPROVEMENT 2: ADVANCED SAMPLING TECHNIQUES
# ============================================================================
print("\n" + "=" * 80)
print("IMPROVEMENT 2: ADVANCED SAMPLING TECHNIQUES")
print("=" * 80)

X = df_enhanced[enhanced_features]
y = df_enhanced['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nOriginal: 0={len(y_train[y_train==0]):,}, 1={len(y_train[y_train==1]):,}")

sampling_strategies = {}

# 2.1 ADASYN (Adaptive Synthetic Sampling)
print("\n2.1 Testing ADASYN (focuses on hard-to-learn samples)...")
try:
    under = RandomUnderSampler(sampling_strategy={0: len(y_train[y_train==1]) * 10, 1: len(y_train[y_train==1])}, random_state=42)
    X_temp, y_temp = under.fit_resample(X_train, y_train)
    adasyn = ADASYN(sampling_strategy=0.33, random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_temp, y_temp)
    sampling_strategies['ADASYN'] = (X_adasyn, y_adasyn)
    print(f"   Result: 0={len(y_adasyn[y_adasyn==0]):,}, 1={len(y_adasyn[y_adasyn==1]):,}")
except Exception as e:
    print(f"   ⚠️  ADASYN failed: {e}")

# 2.2 Borderline-SMOTE (focuses on borderline samples)
print("\n2.2 Testing Borderline-SMOTE (focuses on decision boundary)...")
try:
    under = RandomUnderSampler(sampling_strategy={0: len(y_train[y_train==1]) * 10, 1: len(y_train[y_train==1])}, random_state=42)
    X_temp, y_temp = under.fit_resample(X_train, y_train)
    borderline = BorderlineSMOTE(sampling_strategy=0.33, random_state=42, kind='borderline-1')
    X_borderline, y_borderline = borderline.fit_resample(X_temp, y_temp)
    sampling_strategies['Borderline-SMOTE'] = (X_borderline, y_borderline)
    print(f"   Result: 0={len(y_borderline[y_borderline==0]):,}, 1={len(y_borderline[y_borderline==1]):,}")
except Exception as e:
    print(f"   ⚠️  Borderline-SMOTE failed: {e}")

# 2.3 SMOTE + Tomek Links (removes noisy samples)
print("\n2.3 Testing SMOTE-Tomek (removes borderline noise)...")
try:
    under = RandomUnderSampler(sampling_strategy={0: len(y_train[y_train==1]) * 10, 1: len(y_train[y_train==1])}, random_state=42)
    X_temp, y_temp = under.fit_resample(X_train, y_train)
    smote_tomek = SMOTETomek(sampling_strategy=0.33, random_state=42)
    X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_temp, y_temp)
    sampling_strategies['SMOTE-Tomek'] = (X_smote_tomek, y_smote_tomek)
    print(f"   Result: 0={len(y_smote_tomek[y_smote_tomek==0]):,}, 1={len(y_smote_tomek[y_smote_tomek==1]):,}")
except Exception as e:
    print(f"   ⚠️  SMOTE-Tomek failed: {e}")

# 2.4 Current approach (baseline)
print("\n2.4 Current approach (Undersample + SMOTE)...")
under = RandomUnderSampler(sampling_strategy={0: len(y_train[y_train==1]) * 10, 1: len(y_train[y_train==1])}, random_state=42)
X_temp, y_temp = under.fit_resample(X_train, y_train)
smote = SMOTE(sampling_strategy=0.33, random_state=42)
X_current, y_current = smote.fit_resample(X_temp, y_temp)
sampling_strategies['Current (Under+SMOTE)'] = (X_current, y_current)
print(f"   Result: 0={len(y_current[y_current==0]):,}, 1={len(y_current[y_current==1]):,}")

# ============================================================================
# IMPROVEMENT 3: COMPARE SAMPLING STRATEGIES
# ============================================================================
print("\n" + "=" * 80)
print("IMPROVEMENT 3: EVALUATING ALL STRATEGIES")
print("=" * 80)

results = []

for strategy_name, (X_train_balanced, y_train_balanced) in sampling_strategies.items():
    print(f"\n🔬 Testing: {strategy_name}")
    
    # Train XGBoost (best model from previous experiments)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Strategy': strategy_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'Training Size': len(y_train_balanced)
    })
    
    print(f"   Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# IMPROVEMENT 4: THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "=" * 80)
print("IMPROVEMENT 4: THRESHOLD OPTIMIZATION")
print("=" * 80)

# Use best strategy
best_strategy = max(results, key=lambda x: x['F1'])
print(f"\nUsing best strategy: {best_strategy['Strategy']}")

X_train_best, y_train_best = sampling_strategies[best_strategy['Strategy']]

# Train final model
final_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_best, y_train_best)

# Get probabilities
y_prob = final_model.predict_proba(X_test)[:, 1]

# Find optimal thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

# Different optimization goals
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

print(f"\n📊 Threshold Analysis:")
print(f"   Default (0.5): Precision={precision_score(y_test, y_prob >= 0.5, zero_division=0):.4f}, Recall={recall_score(y_test, y_prob >= 0.5, zero_division=0):.4f}")
print(f"   Optimal F1 ({best_f1_threshold:.3f}): Precision={precisions[best_f1_idx]:.4f}, Recall={recalls[best_f1_idx]:.4f}, F1={f1_scores[best_f1_idx]:.4f}")

# Find threshold for 40% precision
target_precision_idx = np.where(precisions >= 0.4)[0]
if len(target_precision_idx) > 0:
    idx = target_precision_idx[0]
    threshold_40 = thresholds[idx] if idx < len(thresholds) else 0.5
    print(f"   40% Precision ({threshold_40:.3f}): Precision={precisions[idx]:.4f}, Recall={recalls[idx]:.4f}, F1={f1_scores[idx]:.4f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1', ascending=False)

print("\n" + df_results.to_string(index=False))

print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

best = df_results.iloc[0]
print(f"\n1️⃣  Best Sampling Strategy: {best['Strategy']}")
print(f"   - F1 Score: {best['F1']:.4f}")
print(f"   - Precision: {best['Precision']:.4f}")
print(f"   - Recall: {best['Recall']:.4f}")
print(f"   - ROC-AUC: {best['ROC-AUC']:.4f}")

print(f"\n2️⃣  Feature Engineering Impact:")
print(f"   - Added {len(enhanced_features) - len(base_features)} new features")
print(f"   - Includes: interactions, temporal patterns, risk indicators, polynomials")

print(f"\n3️⃣  Threshold Tuning:")
print(f"   - Use threshold {best_f1_threshold:.3f} for best F1 score")
print(f"   - Or use {threshold_40:.3f} if you need 40%+ precision")

print(f"\n4️⃣  Next Steps:")
print(f"   - Update model_training.py with best strategy: {best['Strategy']}")
print(f"   - Add enhanced features to feature_engineering.py")
print(f"   - Implement threshold tuning in predictions")

print("\n" + "=" * 80)
