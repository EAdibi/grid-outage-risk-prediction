"""Feature engineering for the Early Warning next-window outage model.

This is intentionally separate from `feature_engineering.py`. It writes only to
the `early_warning_training_data` collection and does not touch the existing
daily model's `training_data` collection.
"""

from datetime import timedelta
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db


WINDOWS = [1, 3, 6, 12, 24]
RANDOM_STATE = 42
MAX_POSITIVE_EVENTS = int(os.getenv("EARLY_WARNING_MAX_POSITIVE_EVENTS", "5000"))
NEGATIVE_MULTIPLIER = int(os.getenv("EARLY_WARNING_NEGATIVE_MULTIPLIER", "1"))

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


def _as_time_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            values = pd.to_datetime(df[col], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(pd.NaT, index=df.index)


def _load_outages(db):
    rows = list(db.outages.find(
        {"location.county_fips": {"$ne": None}},
        {
            "event_began": 1,
            "start_time": 1,
            "location.county_fips": 1,
            "location.county_name": 1,
            "location.state": 1,
            "max_customers": 1,
            "duration_hours": 1,
        },
    ))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No outages found in MongoDB.")

    df["event_time"] = _as_time_column(df, ["event_began", "start_time"])
    df["county_fips"] = df["location"].apply(lambda x: str(x.get("county_fips", "")).zfill(5))
    df["county_name"] = df["location"].apply(lambda x: x.get("county_name"))
    df["state"] = df["location"].apply(lambda x: x.get("state"))
    df["max_customers"] = pd.to_numeric(df.get("max_customers"), errors="coerce").fillna(0)
    df["duration_hours"] = pd.to_numeric(df.get("duration_hours"), errors="coerce").fillna(0)
    return df.dropna(subset=["event_time", "county_fips"])


def _load_weather(db):
    rows = list(db.storm_events.find(
        {"location.county_fips": {"$ne": None}},
        {
            "begin_date": 1,
            "location.county_fips": 1,
            "damage_property": 1,
            "magnitude": 1,
        },
    ))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "event_time", "county_fips", "damage_property", "magnitude"
        ])

    df["event_time"] = _as_time_column(df, ["begin_date"])
    df["county_fips"] = df["location"].apply(lambda x: str(x.get("county_fips", "")).zfill(5))
    df["damage_property"] = pd.to_numeric(df.get("damage_property"), errors="coerce").fillna(0)
    df["magnitude"] = pd.to_numeric(df.get("magnitude"), errors="coerce").fillna(0)
    return df.dropna(subset=["event_time", "county_fips"])


def _load_population(db):
    rows = list(db.county_population.find({}, {
        "county_fips": 1,
        "latest_population": 1,
    }))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["latest_population"] = pd.to_numeric(df["latest_population"], errors="coerce").fillna(0)
    return dict(zip(df["county_fips"], df["latest_population"]))


def _build_county_lookup(outages):
    lookup = outages[["county_fips", "county_name", "state"]].drop_duplicates("county_fips")
    return lookup.set_index("county_fips").to_dict("index")


def _build_series(df, value_cols):
    series = {}
    for county, group in df.sort_values("event_time").groupby("county_fips"):
        item = {"time": group["event_time"].astype("int64").to_numpy()}
        for col in value_cols:
            item[col] = pd.to_numeric(group[col], errors="coerce").fillna(0).to_numpy()
        series[county] = item
    return series


def _slice(series, county, start, end):
    item = series.get(county)
    if item is None:
        return None, slice(0, 0)
    times = item["time"]
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    left = np.searchsorted(times, start_ns, side="left")
    right = np.searchsorted(times, end_ns, side="left")
    return item, slice(left, right)


def _count_events(series, county, start, end):
    item, idx = _slice(series, county, start, end)
    if item is None:
        return 0
    return idx.stop - idx.start


def _sum_values(series, county, start, end, col):
    item, idx = _slice(series, county, start, end)
    if item is None or idx.stop == idx.start:
        return 0.0
    return float(item[col][idx].sum())


def _mean_values(series, county, start, end, col):
    item, idx = _slice(series, county, start, end)
    if item is None or idx.stop == idx.start:
        return 0.0
    return float(item[col][idx].mean())


def _has_future_outage(outage_series, county, prediction_time, window_hours):
    return _count_events(
        outage_series,
        county,
        prediction_time,
        prediction_time + timedelta(hours=int(window_hours)),
    ) > 0


def _features_for_row(row, outage_series, weather_series, population):
    county = row["county_fips"]
    t = row["prediction_time"]
    features = {
        "window_hours": int(row["window_hours"]),
        "hour": t.hour,
        "day_of_week": t.dayofweek,
        "month": t.month,
        "is_weekend": int(t.dayofweek >= 5),
        "latest_population": float(population.get(county, 0)),
    }

    for hours in [24, 72, 168]:
        start = t - timedelta(hours=hours)
        suffix = f"{hours}h"
        features[f"recent_outage_count_{suffix}"] = _count_events(outage_series, county, start, t)
        features[f"weather_event_count_{suffix}"] = _count_events(weather_series, county, start, t)

    features["recent_customers_affected_24h"] = _sum_values(
        outage_series, county, t - timedelta(hours=24), t, "max_customers"
    )
    features["recent_customers_affected_72h"] = _sum_values(
        outage_series, county, t - timedelta(hours=72), t, "max_customers"
    )
    features["recent_duration_mean_168h"] = _mean_values(
        outage_series, county, t - timedelta(hours=168), t, "duration_hours"
    )
    features["total_property_damage_72h"] = _sum_values(
        weather_series, county, t - timedelta(hours=72), t, "damage_property"
    )
    features["avg_weather_magnitude_72h"] = _mean_values(
        weather_series, county, t - timedelta(hours=72), t, "magnitude"
    )
    return features


def _build_candidate_rows(outages, outage_series):
    rng = np.random.default_rng(RANDOM_STATE)
    sampled = outages.sample(
        n=min(MAX_POSITIVE_EVENTS, len(outages)),
        random_state=RANDOM_STATE,
    ).copy()

    rows = []
    lookup = _build_county_lookup(outages)
    for _, outage in sampled.iterrows():
        for window in WINDOWS:
            offset = rng.uniform(0.25, max(float(window), 0.5))
            prediction_time = outage["event_time"] - timedelta(hours=float(offset))
            county = outage["county_fips"]
            meta = lookup.get(county, {})
            rows.append({
                "county_fips": county,
                "county_name": meta.get("county_name"),
                "state": meta.get("state"),
                "prediction_time": prediction_time,
                "window_hours": window,
                "target": 1,
            })

    counties = outages["county_fips"].dropna().unique()
    min_time = outages["event_time"].min()
    max_time = outages["event_time"].max() - timedelta(hours=max(WINDOWS))
    negative_count = max(1, len(rows) * NEGATIVE_MULTIPLIER)
    total_hours = max(1, int((max_time - min_time).total_seconds() // 3600))

    attempts = 0
    while negative_count > 0 and attempts < negative_count * 20:
        attempts += 1
        county = str(rng.choice(counties))
        prediction_time = min_time + timedelta(hours=int(rng.integers(0, total_hours)))
        window = int(rng.choice(WINDOWS))
        if _has_future_outage(outage_series, county, prediction_time, window):
            continue
        meta = lookup.get(county, {})
        rows.append({
            "county_fips": county,
            "county_name": meta.get("county_name"),
            "state": meta.get("state"),
            "prediction_time": prediction_time,
            "window_hours": window,
            "target": 0,
        })
        negative_count -= 1

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("EARLY WARNING FEATURE ENGINEERING")
    print("=" * 70)

    db = get_db(use_cache_fallback=False)
    print("Loading source collections...")
    outages = _load_outages(db)
    weather = _load_weather(db)
    population = _load_population(db)
    print(f"Loaded {len(outages):,} outage events")
    print(f"Loaded {len(weather):,} weather events")

    outage_series = _build_series(outages, ["max_customers", "duration_hours"])
    weather_series = _build_series(weather, ["damage_property", "magnitude"])

    print("Building next-window labels...")
    candidates = _build_candidate_rows(outages, outage_series)
    print(f"Candidate rows: {len(candidates):,}")
    print(f"Target distribution: {candidates['target'].value_counts().to_dict()}")

    print("Creating features...")
    feature_rows = []
    for _, row in candidates.iterrows():
        features = _features_for_row(row, outage_series, weather_series, population)
        feature_rows.append({**row.to_dict(), **features})

    training_df = pd.DataFrame(feature_rows)
    training_df["prediction_time"] = pd.to_datetime(training_df["prediction_time"])
    training_df = training_df[[
        "county_fips", "county_name", "state", "prediction_time",
        *FEATURE_COLS,
        "target",
    ]]

    records = training_df.to_dict("records")
    print("Saving to MongoDB collection 'early_warning_training_data'...")
    db.early_warning_training_data.delete_many({})
    db.early_warning_training_data.insert_many(records)
    print(f"Saved {len(records):,} rows")
    print("Next: python ml_pipeline/early_warning_model_training.py")


if __name__ == "__main__":
    main()
