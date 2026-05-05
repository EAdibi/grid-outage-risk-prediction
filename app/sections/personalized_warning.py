import base64
import io
import math
import wave

import plotly.graph_objects as go
import streamlit as st

from components import section_header
from data import early_warning_predictions

TITLE = "Personalized Warning"
ICON = "📣"
ORDER = 7
SECTION = "AI Analysis"


def _risk_label(probability: float) -> str:
    if probability >= 0.8:
        return "Critical"
    if probability >= 0.7:
        return "High"
    if probability >= 0.5:
        return "Elevated"
    return "Low"


def _alert_tone() -> bytes:
    sample_rate = 44100
    duration_seconds = 0.7
    frames = int(sample_rate * duration_seconds)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            envelope = 0.55 if (i // 6000) % 2 == 0 else 0.18
            value = int(32767 * envelope * math.sin(2 * math.pi * 880 * t))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))

    return buffer.getvalue()


def _autoplay_alert_tone() -> None:
    encoded = base64.b64encode(_alert_tone()).decode("ascii")
    st.markdown(
        f"""
        <audio autoplay loop>
            <source src="data:audio/wav;base64,{encoded}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def _alert_panel(status: str, message: str, probability: float, is_warning: bool) -> None:
    if is_warning:
        background = "linear-gradient(135deg, #991b1b, #ef4444)"
        shadow = "0 0 0 0 rgba(239, 68, 68, 0.65)"
        icon = "!"
    else:
        background = "linear-gradient(135deg, #065f46, #14b8a6)"
        shadow = "0 0 0 0 rgba(20, 184, 166, 0.45)"
        icon = "✓"

    st.markdown(
        f"""
        <style>
        @keyframes warningPulse {{
            0% {{ box-shadow: {shadow}; }}
            70% {{ box-shadow: 0 0 0 18px rgba(239, 68, 68, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .personal-alert {{
            background: {background};
            color: white;
            padding: 24px;
            border-radius: 8px;
            margin: 12px 0 18px 0;
            animation: warningPulse 1.5s infinite;
        }}
        .personal-alert-icon {{
            width: 44px;
            height: 44px;
            border: 2px solid rgba(255,255,255,0.85);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 28px;
            margin-right: 12px;
            vertical-align: middle;
        }}
        .personal-alert-title {{
            display: inline-block;
            font-size: 26px;
            font-weight: 800;
            vertical-align: middle;
        }}
        .personal-alert-message {{
            font-size: 16px;
            margin-top: 14px;
            line-height: 1.45;
        }}
        .personal-alert-prob {{
            font-size: 38px;
            font-weight: 800;
            margin-top: 8px;
        }}
        </style>
        <div class="personal-alert">
            <span class="personal-alert-icon">{icon}</span>
            <span class="personal-alert-title">{status}</span>
            <div class="personal-alert-message">{message}</div>
            <div class="personal-alert-prob">{probability:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _gauge(probability: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if probability >= threshold else "#14b8a6"},
            "steps": [
                {"range": [0, 50], "color": "#d1fae5"},
                {"range": [50, 70], "color": "#fef3c7"},
                {"range": [70, 80], "color": "#fed7aa"},
                {"range": [80, 100], "color": "#fecaca"},
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 4},
                "thickness": 0.78,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 20, "b": 10})
    return fig


def show() -> None:
    section_header(
        TITLE,
        "Interactive outage warning lookup for a selected county and prediction window.",
    )

    with st.spinner("Loading Early Warning predictions..."):
        df = early_warning_predictions()

    if df.empty:
        st.warning(
            "No Early Warning predictions found. Run "
            "`python ml_pipeline/early_warning_feature_engineering.py` and then "
            "`python ml_pipeline/early_warning_model_training.py`."
        )
        return

    df = df.dropna(subset=["state", "county_name", "window_hours"])
    states = sorted(df["state"].dropna().unique())

    with st.form("personalized_warning_form"):
        st.subheader("Your Alert Settings")
        name = st.text_input("Name", value="Demo User")
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox("State", states, index=states.index("Texas") if "Texas" in states else 0)
        with col2:
            window = st.selectbox("Prediction window", [1, 3, 6, 12, 24], index=4)

        county_options = (
            df[(df["state"] == state) & (df["window_hours"] == window)]
            .sort_values("county_name")["county_name"]
            .dropna()
            .unique()
            .tolist()
        )
        if county_options:
            county = st.selectbox("County", county_options)
        else:
            county = None
            st.warning("No counties are available for this state/window selection.")
        threshold = st.select_slider("Warning threshold", options=[0.5, 0.6, 0.7, 0.8], value=0.7)
        submitted = st.form_submit_button("Check My Outage Risk", use_container_width=True)

    if submitted:
        if county is None:
            st.warning("Pick a different state or prediction window.")
            return
        st.session_state["personalized_warning_selection"] = {
            "name": name,
            "state": state,
            "county": county,
            "window": window,
            "threshold": threshold,
        }
        st.session_state["personalized_warning_audio_stopped"] = False
    elif "personalized_warning_selection" in st.session_state:
        selection = st.session_state["personalized_warning_selection"]
        name = selection["name"]
        state = selection["state"]
        county = selection["county"]
        window = selection["window"]
        threshold = selection["threshold"]
    else:
        st.info("Choose your area and press Check My Outage Risk.")
        return

    match = df[
        (df["state"] == state)
        & (df["county_name"] == county)
        & (df["window_hours"] == window)
    ].sort_values("prediction_time").tail(1)

    if match.empty:
        st.warning("No Early Warning prediction is available for that selection.")
        return

    row = match.iloc[0]
    probability = float(row["outage_probability"])
    is_warning = probability >= threshold
    risk = _risk_label(probability)
    prediction_time = row["prediction_time"]
    model_used = row.get("model_used", "Early Warning model")

    if is_warning:
        status = f"{risk} Outage Warning"
        message = (
            f"Hey {name}! {county}, {state} is above your selected warning threshold "
            f"for the next {window} hours."
        )
    else:
        status = "No Active Warning"
        message = (
            f"Hey {name}! {county}, {state} is below your selected warning threshold "
            f"for the next {window} hours."
        )

    _alert_panel(status, message, probability, is_warning)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(_gauge(probability, threshold), use_container_width=True)
    with right:
        st.metric("Selected threshold", f"{threshold:.0%}")
        st.metric("Risk level", risk)
        st.metric("Prediction window", f"Next {window} hours")
        st.caption(f"Model: {model_used} | Prediction time: {prediction_time}")

    st.progress(min(max(probability, 0), 1), text=f"Outage probability: {probability:.1%}")

    if is_warning:
        if not st.session_state.get("personalized_warning_audio_stopped", False):
            _autoplay_alert_tone()
            if st.button("Stop Audible Cue", use_container_width=True):
                st.session_state["personalized_warning_audio_stopped"] = True
                st.rerun()
        st.error("Visual warning is active for this personalized selection.")
    else:
        st.session_state["personalized_warning_audio_stopped"] = False
        st.success("No personalized warning is active for this selection.")

    st.subheader("Selected Prediction Details")
    st.dataframe(
        match[[
            "county_name",
            "state",
            "window_hours",
            "outage_probability",
            "risk_level",
            "model_used",
        ]].rename(columns={
            "county_name": "County",
            "state": "State",
            "window_hours": "Window Hours",
            "outage_probability": "Probability",
            "risk_level": "Model Risk Level",
            "model_used": "Model",
        }).style.format({"Probability": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
