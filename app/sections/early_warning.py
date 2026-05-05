from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components import county_choropleth, metric_row, section_header
from data import early_warning_predictions, state_list, warning_probability_history

TITLE = "Early Warning"
ICON = "🚨"
ORDER = 6
SECTION = "AI Analysis"


def _risk_label(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    if probability >= 0.7:
        return "High"
    if probability >= 0.5:
        return "Elevated"
    return "Low"


def _load_feature_importance() -> pd.DataFrame:
    path = Path(__file__).parent.parent.parent / "models" / "early_warning_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(path)


def show() -> None:
    section_header(
        TITLE,
        "County-level next-window outage warning view from the separate Early Warning model.",
    )

    with st.spinner("Loading warning predictions..."):
        df = early_warning_predictions()

    if df.empty:
        st.warning(
            "No Early Warning predictions found. Run "
            "`python ml_pipeline/early_warning_feature_engineering.py` and then "
            "`python ml_pipeline/early_warning_model_training.py`."
        )
        return

    states = ["ALL"] + state_list()
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        region = st.selectbox("Region", states, index=0)
    with col2:
        window_hours = st.selectbox("Prediction window", [1, 3, 6, 12, 24], index=4)
    with col3:
        threshold = st.selectbox("Warning threshold", [0.7, 0.8], index=1)

    filtered = df.copy()
    filtered = filtered[filtered["window_hours"] == window_hours]
    if region != "ALL":
        filtered = filtered[filtered["state"] == region]

    filtered["outage_probability"] = filtered["outage_probability"].fillna(0)
    filtered["risk_level"] = filtered["outage_probability"].apply(_risk_label)
    warnings = filtered[filtered["outage_probability"] >= threshold]

    latest_time = pd.to_datetime(filtered["prediction_time"]).max() if not filtered.empty else None
    model_name = filtered["model_used"].dropna().iloc[0] if filtered["model_used"].notna().any() else "Existing ML model"

    metric_row({
        "Prediction window": f"Next {window_hours} hours",
        "Counties monitored": f"{len(filtered):,}",
        "Active warnings": f"{len(warnings):,}",
        "Max probability": f"{filtered['outage_probability'].max():.1%}" if not filtered.empty else "0.0%",
    })

    if latest_time is not None:
        st.caption(
            f"Using separate Early Warning {model_name} predictions generated for "
            f"{latest_time}."
        )

    left, right = st.columns([1.45, 1])
    with left:
        fig = county_choropleth(
            filtered,
            value_col="outage_probability",
            title="Outage Probability",
            color_scale="YlOrRd",
        )
        fig.update_layout(coloraxis_colorbar={"tickformat": ".0%"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("High-Risk Counties")
        table = warnings.sort_values("outage_probability", ascending=False).head(15)
        if table.empty:
            st.info("No counties exceed the selected threshold.")
        else:
            st.dataframe(
                table[["county_name", "state", "outage_probability", "risk_level"]]
                .rename(columns={
                    "county_name": "County",
                    "state": "State",
                    "outage_probability": "Probability",
                    "risk_level": "Risk",
                })
                .style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

    history = warning_probability_history()
    history = history[history["window_hours"] == window_hours]
    if region != "ALL":
        history = history[history["state"] == region]

    st.subheader("Risk Trend")
    if history.empty:
        st.info("No prediction history available for the selected region.")
    else:
        top_counties = (
            filtered.sort_values("outage_probability", ascending=False)
            .head(5)["county_fips"]
            .tolist()
        )
        trend = history[history["county_fips"].isin(top_counties)].copy()
        trend["prediction_time"] = pd.to_datetime(trend["prediction_time"])
        trend["label"] = trend["county_name"].fillna(trend["county_fips"])

        fig = px.line(
            trend.sort_values("prediction_time"),
            x="prediction_time",
            y="outage_probability",
            color="label",
            markers=True,
            labels={
                "prediction_time": "Prediction time",
                "outage_probability": "Outage probability",
                "label": "County",
            },
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    importance = _load_feature_importance()
    if not importance.empty:
        st.subheader("Model Feature Importance")
        fig = px.bar(
            importance.head(10).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
