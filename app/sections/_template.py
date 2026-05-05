import streamlit as st

from components import metric_row, section_header, state_filter
from data import outages_by_county

TITLE = "My New Page"
ICON = "📊"
ORDER = 50
SECTION = "Live Analysis"  # or "AI Analysis"


def show():
    section_header(TITLE, "One-line description.")

    state = state_filter()
    df = outages_by_county()
    if state != "ALL":
        df = df[df["state"] == state]

    metric_row({
        "Counties": f"{len(df):,}",
        "Total outages": f"{int(df['outage_count'].sum()):,}",
    })
    st.dataframe(df.head(20), use_container_width=True)
