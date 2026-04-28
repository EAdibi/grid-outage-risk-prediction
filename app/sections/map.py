import streamlit as st

from components import county_choropleth, metric_row, section_header, state_filter
from data import outages_by_county

TITLE = "Vulnerability Map"
ICON = "🗺️"
ORDER = 10


def show() -> None:
    section_header(TITLE, "County-level outage counts across the U.S. (2014–2023).")

    state = state_filter()
    with st.spinner("Loading data..."):
        df = outages_by_county()
    if state != "ALL":
        df = df[df["state"] == state]

    metric_row({
        "Counties shown": f"{len(df):,}",
        "Total outages": f"{int(df['outage_count'].sum()):,}",
        "Max in one county": f"{int(df['outage_count'].max()) if not df.empty else 0:,}",
    })

    fig = county_choropleth(df, value_col="outage_count", title="Outage Count")
    st.plotly_chart(fig, use_container_width=True)
