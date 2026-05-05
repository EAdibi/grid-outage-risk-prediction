import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from components import loading, section_header
from spark import get_spark, show_install_help, spark_ready
from spark_data import outages_df, storms_df


TITLE = "Cluster Console"
ICON = "🛰️"
ORDER = 5
SECTION = "Live Analysis"


def outage_hotspots(spark):
    outages_df()
    sql = """
        SELECT
            county_fips,
            FIRST(county_name) AS county_name,
            FIRST(state)       AS state,
            COUNT(*)           AS outage_count,
            ROUND(SUM(max_customers * duration_hours)) AS customer_hours,
            ROUND(AVG(duration_hours), 2) AS avg_duration_hours
        FROM outages
        GROUP BY county_fips
        ORDER BY customer_hours DESC
    """
    df = spark.sql(sql).toPandas()
    fig = px.bar(
        df.head(20),
        x="county_name",
        y="customer_hours",
        color="state",
        title="Top 20 counties by outage impact (customers x hours)",
    )
    return df, fig, sql


def storm_lift(spark):
    outages_df()
    storms_df()
    sql = """
        WITH paired AS (
            SELECT s.storm_event_type, o.county_fips, o.start_time
            FROM outages o
            JOIN storms s
              ON o.county_fips = s.county_fips
             AND s.storm_time <= o.start_time
             AND s.storm_time >= o.start_time - INTERVAL 7 DAYS
        )
        SELECT
            storm_event_type,
            COUNT(*) AS pairings,
            COUNT(DISTINCT county_fips) AS counties_seen
        FROM paired
        GROUP BY storm_event_type
        ORDER BY pairings DESC
    """
    df = spark.sql(sql).toPandas()
    fig = px.bar(
        df.head(15),
        x="storm_event_type",
        y="pairings",
        color="counties_seen",
        color_continuous_scale="Reds",
        title="Storm types preceding outages within 7 days",
    )
    return df, fig, sql


def seasonal_pattern(spark):
    outages_df()
    sql = """
        SELECT state, month(start_time) AS month, COUNT(*) AS outages
        FROM outages
        WHERE state != ''
        GROUP BY state, month(start_time)
    """
    df = spark.sql(sql).toPandas()
    top = df.groupby("state")["outages"].sum().nlargest(15).index
    df = df[df["state"].isin(top)]
    fig = px.density_heatmap(
        df, x="month", y="state", z="outages", histfunc="sum",
        color_continuous_scale="Reds",
        title="Outages by month and state (top 15)",
    )
    fig.update_xaxes(dtick=1)
    return df, fig, sql


QUERIES = {
    "Outage hotspots": outage_hotspots,
    "Storm lift (7-day window)": storm_lift,
    "Seasonal patterns": seasonal_pattern,
}


def recent_jobs(ui_url):
    try:
        apps = requests.get(f"{ui_url}/api/v1/applications", timeout=2).json()
        if not apps:
            return pd.DataFrame()
        app_id = apps[0]["id"]
        jobs = requests.get(
            f"{ui_url}/api/v1/applications/{app_id}/jobs", timeout=2
        ).json()[:10]
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Job": j.get("jobId"),
        "Status": j.get("status"),
        "Stages": j.get("numCompletedStages", 0),
        "Tasks": f"{j.get('numCompletedTasks', 0)}/{j.get('numTasks', 0)}",
    } for j in jobs])


def show():
    section_header(TITLE, "Spark SQL queries over the full outages and storm-events corpus.")

    if not spark_ready():
        show_install_help()
        return

    spark = get_spark()
    ui_url = spark.sparkContext.uiWebUrl

    left, right = st.columns([3, 2], gap="large")

    with left:
        choice = st.radio("Query", list(QUERIES.keys()), label_visibility="collapsed")
        run = st.button("Run on cluster", type="primary", use_container_width=True)

    with right:
        sc = spark.sparkContext
        st.caption(f"App `{sc.applicationId}` · {sc.master} · {sc.defaultParallelism} cores")
        if ui_url:
            st.link_button("Open Spark UI", ui_url, use_container_width=True)
        jobs = recent_jobs(ui_url)
        if not jobs.empty:
            st.dataframe(jobs, use_container_width=True, hide_index=True, height=200)

    if not run:
        return

    with loading(f"Running {choice} on the cluster..."):
        t0 = time.time()
        df, fig, sql = QUERIES[choice](spark)
        elapsed = time.time() - t0

    st.caption(f"{len(df):,} rows in {elapsed:.2f}s")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("SQL"):
        st.code(sql.strip(), language="sql")
    with st.expander("Rows"):
        st.dataframe(df, use_container_width=True)
