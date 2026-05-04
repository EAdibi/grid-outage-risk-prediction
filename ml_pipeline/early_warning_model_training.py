"""Train Early Warning models for next-window outage prediction.

This script is separate from `model_training.py`. It reads only
`early_warning_training_data` and writes only Early Warning model artifacts plus
the `early_warning_predictions` collection.
"""

from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db


FEATURE_COLS = [
    "window_hours",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "recent_outage_count_24h",
    "recent_outage_count_72h",
    "recent_outage_count_168h",
    "recent_customers_affected_24h",
    "recent_customers_affected_72h",
    "recent_duration_mean_168h",
    "weather_event_count_24h",
    "weather_event_count_72h",
    "weather_event_count_168h",
    "total_property_damage_72h",
    "avg_weather_magnitude_72h",
    "latest_population",
]


def _metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0,
    }


def _balance_training_data(X_train, y_train):
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 6:
        print("Skipping SMOTE because there are too few minority-class samples.")
        return X_train, y_train
    if class_counts.min() / class_counts.max() >= 0.4:
        print("Skipping SMOTE because Early Warning classes are already balanced.")
        return X_train, y_train

    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.75, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    X_balanced, y_balanced = under.fit_resample(X_balanced, y_balanced)
    return X_balanced, y_balanced


def _train_models(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "metrics": _metrics(y_test, y_pred, y_prob),
        }
        print(
            f"  F1={results[name]['metrics']['f1']:.4f} "
            f"ROC-AUC={results[name]['metrics']['roc_auc']:.4f}"
        )
    return results


def _feature_importance(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.zeros(len(feature_cols))
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": values,
    }).sort_values("importance", ascending=False)


def main():
    print("=" * 70)
    print("EARLY WARNING MODEL TRAINING")
    print("=" * 70)

    db = get_db(use_cache_fallback=False)
    rows = list(db.early_warning_training_data.find())
    if not rows:
        raise ValueError(
            "No early_warning_training_data found. "
            "Run python ml_pipeline/early_warning_feature_engineering.py first."
        )

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].fillna(0)
    y = df["target"].astype(int)
    print(f"Loaded {len(df):,} rows")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    stratify = y if len(y.value_counts()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    X_train_balanced, y_train_balanced = _balance_training_data(X_train, y_train)
    print(f"Balanced target distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

    results = _train_models(X_train_balanced, y_train_balanced, X_test, y_test)
    best_name = max(results.items(), key=lambda item: item[1]["metrics"]["f1"])[0]
    best_model = results[best_name]["model"]
    print(f"Best Early Warning model by F1: {best_name}")

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(results["Random Forest"]["model"], models_dir / "early_warning_random_forest.pkl")
    joblib.dump(results["XGBoost"]["model"], models_dir / "early_warning_xgboost.pkl")
    joblib.dump(results["LightGBM"]["model"], models_dir / "early_warning_lightgbm.pkl")
    joblib.dump(best_model, models_dir / "early_warning_best_model.pkl")

    importance = _feature_importance(best_model, FEATURE_COLS)
    importance.to_csv(models_dir / "early_warning_feature_importance.csv", index=False)
    print(f"Saved Early Warning model artifacts to {models_dir}")

    probabilities = best_model.predict_proba(X)[:, 1]
    predictions = best_model.predict(X)
    output = df[[
        "county_fips",
        "county_name",
        "state",
        "prediction_time",
        "window_hours",
    ]].copy()
    output["outage_probability"] = probabilities
    output["predicted_outage"] = predictions
    output["risk_level"] = pd.cut(
        output["outage_probability"],
        bins=[-0.01, 0.5, 0.7, 0.8, 1.0],
        labels=["Low", "Elevated", "High", "Critical"],
    ).astype(str)
    output["model_used"] = best_name
    output["model_version"] = "early_warning_v1"

    records = output.to_dict("records")
    db.early_warning_predictions.delete_many({})
    db.early_warning_predictions.insert_many(records)
    print(f"Saved {len(records):,} rows to 'early_warning_predictions'")

    metrics_doc = {
        "model_version": "early_warning_v1",
        "best_model": best_name,
        "feature_columns": FEATURE_COLS,
        "metrics": {
            name: {key: float(value) for key, value in result["metrics"].items()}
            for name, result in results.items()
        },
        "feature_importances": {
            feature: float(value)
            for feature, value in zip(importance["feature"], importance["importance"])
        },
    }
    db.early_warning_model_metadata.delete_many({"model_version": "early_warning_v1"})
    db.early_warning_model_metadata.insert_one(metrics_doc)
    print("Saved Early Warning metadata")


if __name__ == "__main__":
    main()
