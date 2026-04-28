"""Template for a new dashboard page.

Copy this file to `sections/<your_page>.py` and edit. The leading underscore
keeps it out of the sidebar.

Conventions:
  - Export TITLE, ICON, ORDER, and a `show()` function.
  - Pull data through helpers in `data.py` — add a new helper there if needed.
  - Reuse widgets from `components.py` for consistent styling.
"""
import streamlit as st

from components import metric_row, section_header, state_filter
from data import outages_by_county

TITLE = "My New Page"
ICON = "📊"
ORDER = 50


def show() -> None:
    section_header(TITLE, "One-line description goes here.")

    state = state_filter()
    df = outages_by_county()
    if state != "ALL":
        df = df[df["state"] == state]

    metric_row({
        "Counties": f"{len(df):,}",
        "Total outages": f"{int(df['outage_count'].sum()):,}",
    })

    st.dataframe(df.head(20), use_container_width=True)
