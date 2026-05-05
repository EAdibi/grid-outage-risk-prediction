import pandas as pd
import streamlit as st

from db import get_db
from spark import get_spark


@st.cache_resource(show_spinner=False)
def outages_df():
    spark = get_spark()
    db = get_db()
    docs = list(db.outages.find(
        {"location.county_fips": {"$exists": True, "$ne": None}},
        {
            "location.county_fips": 1,
            "location.county_name": 1,
            "location.state": 1,
            "start_time": 1,
            "max_customers": 1,
            "duration_hours": 1,
        },
    ))
    rows = pd.DataFrame([{
        "county_fips": d["location"].get("county_fips"),
        "county_name": d["location"].get("county_name") or "Unknown",
        "state": d["location"].get("state") or "",
        "start_time": pd.to_datetime(d.get("start_time"), errors="coerce"),
        "max_customers": float(d.get("max_customers") or 0),
        "duration_hours": float(d.get("duration_hours") or 0),
    } for d in docs]).dropna(subset=["start_time"])
    df = spark.createDataFrame(rows).cache()
    df.createOrReplaceTempView("outages")
    df.count()
    return df


@st.cache_resource(show_spinner=False)
def storms_df():
    spark = get_spark()
    db = get_db()
    docs = list(db.storm_events.find(
        {
            "location.county_fips": {"$exists": True, "$ne": None},
            "location.cz_type": "C",
        },
        {
            "location.county_fips": 1,
            "location.state": 1,
            "begin_date": 1,
            "event_type": 1,
            "damage_property": 1,
            "magnitude": 1,
        },
    ))
    rows = pd.DataFrame([{
        "county_fips": d["location"].get("county_fips"),
        "state": d["location"].get("state") or "",
        "storm_time": pd.to_datetime(d.get("begin_date"), errors="coerce"),
        "storm_event_type": d.get("event_type") or "Unknown",
        "damage_property": float(d.get("damage_property") or 0),
        "magnitude": float(d.get("magnitude") or 0),
    } for d in docs]).dropna(subset=["storm_time"])
    df = spark.createDataFrame(rows).cache()
    df.createOrReplaceTempView("storms")
    df.count()
    return df
