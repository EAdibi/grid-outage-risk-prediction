import streamlit as st
import pandas as pd
import plotly.express as px
from db import get_db


def show():
    st.header("Outage Vulnerability Map")
    st.markdown("County-level outage counts across the U.S. (2014–2023)")

    with st.spinner("Loading data from MongoDB..."):
        db = get_db()
        pipeline = [
            {"$match": {"location.county_fips": {"$exists": True}}},
            {"$group": {
                "_id": "$location.county_fips",
                "outage_count": {"$sum": 1}
            }}
        ]
        results = list(db.outages.aggregate(pipeline))

    df = pd.DataFrame([{
        "county_fips": r["_id"],
        "outage_count": r["outage_count"]
    } for r in results])

    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="county_fips",
        color="outage_count",
        color_continuous_scale="Reds",
        scope="usa",
        labels={"outage_count": "Outage Count"},
        title="Outage Frequency by County"
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
