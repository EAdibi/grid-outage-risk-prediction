import plotly.express as px
import streamlit as st

from components import metric_row, section_header
from data import texas_2021_outages

TITLE = "Texas 2021 Case Study"
ICON = "❄️"
ORDER = 90
SECTION = "AI Analysis"


def show():
    section_header(TITLE, "Texas outages during the February 2021 winter storm.")

    df = texas_2021_outages()
    if df.empty:
        st.warning("No data found for Texas February 2021.")
        return

    metric_row({
        "Outage events": f"{len(df):,}",
        "Counties affected": f"{df['county'].nunique():,}",
        "Peak customers affected": f"{int(df['max_customers'].max()):,}",
    })

    by_county = (
        df.groupby("county")["max_customers"].sum()
          .reset_index()
          .sort_values("max_customers", ascending=False)
          .head(15)
    )
    fig = px.bar(
        by_county,
        x="county",
        y="max_customers",
        title="Customers affected by county, Texas Feb 2021",
        labels={"max_customers": "Max customers affected", "county": "County"},
        color="max_customers",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw data")
    st.dataframe(df, use_container_width=True)
