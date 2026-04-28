import streamlit as st

st.set_page_config(
    page_title="U.S. Power Grid Outage Risk",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ U.S. Power Grid Outage Risk Intelligence Platform ⚡ ")
st.markdown("Predicting county-level outage risk across all 50 states.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Vulnerability Map",
    "Top 10 High-Risk Counties",
    "Texas 2021 Case Study"
])

if page == "Vulnerability Map":
    from sections.map import show
    show()
elif page == "Top 10 High-Risk Counties":
    from sections.top_counties import show
    show()
elif page == "Texas 2021 Case Study":
    from sections.texas_2021 import show
    show()
