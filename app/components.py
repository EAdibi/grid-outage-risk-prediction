"""Reusable UI widgets and chart wrappers.

Sections should compose from these so the dashboard has a consistent look.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

COUNTIES_GEOJSON = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)


def section_header(title: str, subtitle: str | None = None) -> None:
    st.header(title)
    if subtitle:
        st.markdown(subtitle)


def metric_row(metrics: dict[str, str | int | float]) -> None:
    """Render a row of KPI tiles. `metrics` maps label → value."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def state_filter(default: str = "ALL", key: str | None = None) -> str:
    """Sidebar dropdown of states. Returns the selected state or 'ALL'."""
    from data import state_list

    options = ["ALL"] + state_list()
    index = options.index(default) if default in options else 0
    return st.sidebar.selectbox("State filter", options, index=index, key=key)


def county_choropleth(
    df: pd.DataFrame,
    value_col: str,
    title: str | None = None,
    color_scale: str = "Reds",
    fips_col: str = "county_fips",
):
    """Plotly county-level choropleth keyed by 5-digit FIPS."""
    fig = px.choropleth(
        df,
        geojson=COUNTIES_GEOJSON,
        locations=fips_col,
        color=value_col,
        color_continuous_scale=color_scale,
        scope="usa",
        labels={value_col: title or value_col},
    )
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
    return fig
