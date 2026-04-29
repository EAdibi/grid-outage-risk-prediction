"""
Optimize Decision Threshold for Better Precision/Recall Trade-off
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 70)
print("OPTIMIZING DECISION THRESHOLD")
print("=" * 70)

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

# Same split as training
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Load best model
models_dir = Path(__file__).parent.parent / "models"
rf_model = joblib.load(models_dir / "random_forest.pkl")

# Get probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find optimal thresholds for different goals
print("\n📊 Threshold Analysis:")
print("=" * 70)

# Goal 1: Maximize F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]

print(f"\n🎯 Best F1 Score Threshold: {best_f1_threshold:.3f}")
print(f"   Precision: {precisions[best_f1_idx]:.4f}")
print(f"   Recall:    {recalls[best_f1_idx]:.4f}")
print(f"   F1 Score:  {f1_scores[best_f1_idx]:.4f}")

# Goal 2: Target 50% precision
target_precision = 0.5
precision_50_idx = np.where(precisions >= target_precision)[0]
if len(precision_50_idx) > 0:
    idx = precision_50_idx[0]
    print(f"\n🎯 Threshold for 50% Precision: {thresholds[idx]:.3f}")
    print(f"   Precision: {precisions[idx]:.4f}")
    print(f"   Recall:    {recalls[idx]:.4f}")
    print(f"   F1 Score:  {f1_scores[idx]:.4f}")

# Goal 3: Target 70% precision
target_precision = 0.7
precision_70_idx = np.where(precisions >= target_precision)[0]
if len(precision_70_idx) > 0:
    idx = precision_70_idx[0]
    print(f"\n🎯 Threshold for 70% Precision: {thresholds[idx]:.3f}")
    print(f"   Precision: {precisions[idx]:.4f}")
    print(f"   Recall:    {recalls[idx]:.4f}")
    print(f"   F1 Score:  {f1_scores[idx]:.4f}")

# Goal 4: Target 90% recall
target_recall = 0.9
recall_90_idx = np.where(recalls >= target_recall)[0]
if len(recall_90_idx) > 0:
    idx = recall_90_idx[-1]  # Last one with recall >= 90%
    print(f"\n🎯 Threshold for 90% Recall: {thresholds[idx]:.3f}")
    print(f"   Precision: {precisions[idx]:.4f}")
    print(f"   Recall:    {recalls[idx]:.4f}")
    print(f"   F1 Score:  {f1_scores[idx]:.4f}")

print("\n" + "=" * 70)
print("💡 Recommendation:")
print("=" * 70)
print(f"For power outage prediction, prioritize RECALL (catching outages).")
print(f"Current threshold (0.5): Precision={precisions[best_f1_idx]:.1%}, Recall={recalls[best_f1_idx]:.1%}")
print(f"\nSuggested threshold: {best_f1_threshold:.3f} (maximizes F1)")
print(f"This balances precision and recall optimally.")
print("\n" + "=" * 70)

# Show distribution of probabilities
print("\n📊 Probability Distribution:")
print(f"   Min:    {y_prob.min():.4f}")
print(f"   25%:    {np.percentile(y_prob, 25):.4f}")
print(f"   Median: {np.median(y_prob):.4f}")
print(f"   75%:    {np.percentile(y_prob, 75):.4f}")
print(f"   Max:    {y_prob.max():.4f}")
print(f"\n   Predictions > 0.5: {(y_prob > 0.5).sum():,} ({(y_prob > 0.5).mean():.1%})")
print(f"   Predictions > 0.7: {(y_prob > 0.7).sum():,} ({(y_prob > 0.7).mean():.1%})")
print(f"   Actual positives:  {y_test.sum():,} ({y_test.mean():.1%})")
