"""Cached query helpers — the only module that talks to MongoDB collections.

Sections should import from here, not call pymongo directly. Every helper
returns a `pd.DataFrame` with a documented column set, and is wrapped in
`@st.cache_data` so Streamlit doesn't re-query Atlas on every interaction.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from db import get_db


@st.cache_data(ttl=3600)
def outages_by_county() -> pd.DataFrame:
    """Outage counts per county across all years.

    Columns: county_fips, county_name, state, outage_count
    """
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


@st.cache_data(ttl=3600)
def texas_2021_outages() -> pd.DataFrame:
    """Texas outages during the Feb 2021 winter storm.

    Columns: county, county_fips, max_customers, duration_hours, start_time
    """
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
def state_list() -> list[str]:
    """Distinct states present in the outages collection (sorted)."""
    db = get_db()
    states = db.outages.distinct("location.state")
    return sorted(s for s in states if s)


@st.cache_data(ttl=600)
def latest_predictions(week_start: datetime | None = None) -> pd.DataFrame:
    """Predictions from the `predictions` collection.

    Returns rows for the given `week_start`, or for the most recent week if None.
    Columns: county_fips, state, week_start, risk_score, risk_level, top_factors

    The `predictions` collection is populated by Phase 2 (Spark MLlib). Returns
    an empty DataFrame until that pipeline runs.
    """
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
def feature_importances(model_version: str = "rf_v1") -> pd.DataFrame:
    """Feature importances for a trained model.

    Columns: feature, importance
    Empty until the model pipeline writes a `model_metadata` document.
    """
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
