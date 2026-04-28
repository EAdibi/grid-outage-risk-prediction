import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from db import get_db


def show():
    st.header("Texas 2021 Winter Storm Case Study")
    st.markdown(
        "Actual outages recorded in Texas during the February 2021 winter storm.")

    with st.spinner("Loading data from MongoDB..."):
        db = get_db()
        outages = list(db.outages.find({
            "location.state": "Texas",
            "start_time": {
                "$gte": datetime(2021, 2, 1),
                "$lte": datetime(2021, 2, 28)
            }
        }, {
            "location.county_name": 1,
            "location.county_fips": 1,
            "max_customers": 1,
            "duration_hours": 1,
            "start_time": 1
        }))

    if not outages:
        st.warning("No data found for Texas February 2021.")
        return

    df = pd.DataFrame([{
        "county": o["location"].get("county_name", "Unknown"),
        "county_fips": o["location"].get("county_fips", ""),
        "max_customers": o.get("max_customers", 0),
        "duration_hours": o.get("duration_hours", 0),
        "start_time": o.get("start_time")
    } for o in outages])

    st.subheader(f"Total outage events: {len(df)}")

    top = df.groupby("county")["max_customers"].sum() \
            .reset_index() \
            .sort_values("max_customers", ascending=False) \
            .head(15)

    fig = px.bar(
        top,
        x="county",
        y="max_customers",
        title="Customers Affected by County — Texas Feb 2021",
        labels={"max_customers": "Max Customers Affected", "county": "County"},
        color="max_customers",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)
