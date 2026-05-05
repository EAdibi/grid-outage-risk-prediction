from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from db import get_db


@st.cache_data(ttl=3600)
def outages_by_county():
    db = get_db()
    pipeline = [
        {"$match": {"location.county_fips": {"$exists": True, "$ne": None}}},
        {"$group": {
            "_id": "$location.county_fips",
            "county_name": {"$first": "$location.county_name"},
            "state": {"$first": "$location.state"},
            "outage_count": {"$sum": 1},
        }},
    ]
    rows = list(db.outages.aggregate(pipeline))
    return pd.DataFrame([{
        "county_fips": r["_id"],
        "county_name": r.get("county_name") or "Unknown",
        "state": r.get("state") or "",
        "outage_count": r["outage_count"],
    } for r in rows])


OUTAGES_BY_COUNTY_SQL = """
SELECT
    county_fips,
    FIRST(county_name) AS county_name,
    FIRST(state)       AS state,
    COUNT(*)           AS outage_count
FROM outages
GROUP BY county_fips
""".strip()


@st.cache_data(ttl=3600, show_spinner=False)
def outages_by_county_spark():
    from spark import get_spark
    from spark_data import outages_df
    spark = get_spark()
    outages_df()
    return spark.sql(OUTAGES_BY_COUNTY_SQL).toPandas()


@st.cache_data(ttl=3600)
def texas_2021_outages():
    db = get_db()
    rows = list(db.outages.find({
        "location.state": "Texas",
        "start_time": {
            "$gte": datetime(2021, 2, 1),
            "$lte": datetime(2021, 2, 28),
        },
    }, {
        "location.county_name": 1,
        "location.county_fips": 1,
        "max_customers": 1,
        "duration_hours": 1,
        "start_time": 1,
    }))
    return pd.DataFrame([{
        "county": r["location"].get("county_name", "Unknown"),
        "county_fips": r["location"].get("county_fips", ""),
        "max_customers": r.get("max_customers") or 0,
        "duration_hours": r.get("duration_hours") or 0,
        "start_time": r.get("start_time"),
    } for r in rows])


@st.cache_data(ttl=3600)
def state_list():
    db = get_db()
    states = db.outages.distinct("location.state")
    return sorted(s for s in states if s)


@st.cache_data(ttl=600)
def latest_predictions(week_start: Optional[datetime] = None):
    db = get_db()
    if week_start is None:
        latest = db.predictions.find_one(sort=[("week_start", -1)])
        if not latest:
            return pd.DataFrame(columns=[
                "county_fips", "state", "week_start",
                "risk_score", "risk_level", "top_factors",
            ])
        week_start = latest["week_start"]
    rows = list(db.predictions.find({"week_start": week_start}))
    return pd.DataFrame([{
        "county_fips": r.get("county_fips"),
        "state": r.get("state"),
        "week_start": r.get("week_start"),
        "risk_score": r.get("risk_score"),
        "risk_level": r.get("risk_level"),
        "top_factors": r.get("top_factors"),
    } for r in rows])


@st.cache_data(ttl=3600)
def feature_importances(model_version="rf_v1"):
    db = get_db()
    doc = db.model_metadata.find_one({"model_version": model_version}) \
        if "model_metadata" in db.list_collection_names() else None
    if not doc or "feature_importances" not in doc:
        return pd.DataFrame(columns=["feature", "importance"])
    items = doc["feature_importances"]
    return pd.DataFrame(
        sorted(items.items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "importance"],
    )


@st.cache_data(ttl=600)
def early_warning_predictions() -> pd.DataFrame:
    """Latest next-window outage probabilities for the warning dashboard.

    Uses the separate `early_warning_predictions` collection written by
    `early_warning_model_training.py`. It does not read or modify the existing
    daily model's `predictions` collection.

    Columns: county_fips, county_name, state, prediction_time, window_hours,
    outage_probability, predicted_outage, risk_level, model_used
    """
    db = get_db()
    rows = list(db.early_warning_predictions.find({}, {
        "county_fips": 1,
        "county_name": 1,
        "state": 1,
        "prediction_time": 1,
        "window_hours": 1,
        "outage_probability": 1,
        "predicted_outage": 1,
        "risk_level": 1,
        "model_used": 1,
    }))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "county_fips", "county_name", "state", "prediction_time",
            "window_hours", "outage_probability", "predicted_outage",
            "risk_level", "model_used",
        ])

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df = df.sort_values("prediction_time").drop_duplicates(
        ["county_fips", "window_hours"],
        keep="last",
    )

    return df[[
        "county_fips", "county_name", "state", "prediction_time",
        "window_hours", "outage_probability", "predicted_outage",
        "risk_level", "model_used",
    ]]


@st.cache_data(ttl=600)
def warning_probability_history() -> pd.DataFrame:
    """Early Warning probability history by county/window for the line chart."""
    db = get_db()
    rows = list(db.early_warning_predictions.find({}, {
        "county_fips": 1,
        "county_name": 1,
        "state": 1,
        "prediction_time": 1,
        "window_hours": 1,
        "outage_probability": 1,
        "model_used": 1,
    }))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "county_fips", "county_name", "state", "prediction_time",
            "window_hours", "outage_probability", "model_used",
        ])

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    return df[[
        "county_fips", "county_name", "state", "prediction_time",
        "window_hours", "outage_probability", "model_used",
    ]]
