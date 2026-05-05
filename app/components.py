from contextlib import contextmanager

import plotly.express as px
import streamlit as st


COUNTIES_GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def section_header(title, subtitle=None):
    st.header(title)
    if subtitle:
        st.markdown(subtitle)


@contextmanager
def loading(label="Loading..."):
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <style>
        .grid-loader {{
          display: flex; flex-direction: column;
          align-items: center; gap: 16px;
          padding: 60px 0;
        }}
        .grid-loader__spinner {{
          width: 56px; height: 56px; border-radius: 50%;
          border: 5px solid rgba(255, 75, 75, 0.18);
          border-top-color: #ff4b4b;
          animation: grid-loader-spin 0.9s linear infinite;
        }}
        .grid-loader__label {{ font-size: 0.95rem; opacity: 0.75; }}
        @keyframes grid-loader-spin {{ to {{ transform: rotate(360deg); }} }}
        </style>
        <div class="grid-loader">
          <div class="grid-loader__spinner"></div>
          <div class="grid-loader__label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def metric_row(metrics):
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def state_filter(default="ALL", key=None):
    from data import state_list
    options = ["ALL"] + state_list()
    idx = options.index(default) if default in options else 0
    return st.sidebar.selectbox("State filter", options, index=idx, key=key)


def county_choropleth(df, value_col, title=None, color_scale="Reds", fips_col="county_fips"):
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
