import streamlit as st
import pandas as pd
from db import get_db


def show():
    st.header("Top 10 High-Risk Counties")
    st.markdown("Counties with the most outages historically (2014–2023)")

    with st.spinner("Loading data from MongoDB..."):
        db = get_db()
        outages = list(db.outages.find(
            {"location.county_fips": {"$exists": True}},
            {"location.county_fips": 1, "location.county_name": 1, "location.state": 1}
        ).limit(50000))

    df = pd.DataFrame([{
        "county_fips": o["location"]["county_fips"],
        "county_name": o["location"].get("county_name", "Unknown"),
        "state": o["location"].get("state", "")
    } for o in outages])

    top10 = df.groupby(["county_fips", "county_name", "state"]) \
              .size() \
              .reset_index(name="outage_count") \
              .sort_values("outage_count", ascending=False) \
              .head(10)

    top10.index = range(1, 11)

    st.dataframe(top10[["county_name", "state", "outage_count"]],
                 use_container_width=True)
