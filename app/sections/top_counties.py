import streamlit as st

from components import section_header, state_filter
from data import outages_by_county

TITLE = "Top 10 High-Risk Counties"
ICON = "🏆"
ORDER = 20
SECTION = "AI Analysis"


def show():
    section_header(TITLE, "Counties with the most recorded outages, 2014 to 2023.")

    state = state_filter()
    df = outages_by_county()
    if state != "ALL":
        df = df[df["state"] == state]

    top = df.sort_values("outage_count", ascending=False).head(10).reset_index(drop=True)
    top.index = range(1, len(top) + 1)
    st.dataframe(top[["county_name", "state", "outage_count"]], use_container_width=True)
