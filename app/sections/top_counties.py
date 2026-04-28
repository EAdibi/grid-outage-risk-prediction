import streamlit as st

from components import section_header, state_filter
from data import outages_by_county

TITLE = "Top 10 High-Risk Counties"
ICON = "🏆"
ORDER = 20


def show() -> None:
    section_header(TITLE, "Counties with the most recorded outages (2014–2023).")

    state = state_filter()
    with st.spinner("Loading data..."):
        df = outages_by_county()
    if state != "ALL":
        df = df[df["state"] == state]

    top10 = df.sort_values("outage_count", ascending=False) \
              .head(10) \
              .reset_index(drop=True)
    top10.index = range(1, len(top10) + 1)

    st.dataframe(
        top10[["county_name", "state", "outage_count"]],
        use_container_width=True,
    )
