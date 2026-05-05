import streamlit as st

from components import county_choropleth, loading, metric_row, section_header
from data import OUTAGES_BY_COUNTY_SQL, outages_by_county, outages_by_county_spark
from spark import get_spark, spark_ready
from spark_data import outages_df

TITLE = "Vulnerability Map"
ICON = "🗺️"
ORDER = 10
SECTION = "Live Analysis"

YEARS = list(range(2014, 2024))


def year_sql(year):
    return (
        "SELECT\n"
        "    county_fips,\n"
        "    FIRST(county_name) AS county_name,\n"
        "    FIRST(state)       AS state,\n"
        "    COUNT(*)           AS outage_count\n"
        "FROM outages\n"
        f"WHERE year(start_time) = {year}\n"
        "GROUP BY county_fips"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def outages_for_year(year):
    spark = get_spark()
    outages_df()
    return spark.sql(year_sql(year)).toPandas()


def show():
    section_header(TITLE, "County-level outage counts, 2014 to 2023.")

    year_choice = st.selectbox("Year", ["All years"] + [str(y) for y in YEARS])
    year = None if year_choice == "All years" else int(year_choice)

    using_spark = spark_ready()
    if using_spark:
        if year is None:
            sql = OUTAGES_BY_COUNTY_SQL
            with loading("Aggregating outages with Spark..."):
                df = outages_by_county_spark()
        else:
            sql = year_sql(year)
            with loading(f"Aggregating {year} outages with Spark..."):
                df = outages_for_year(year)
    else:
        sql = None
        df = outages_by_county()

    metric_row({
        "Counties": f"{len(df):,}",
        "Total outages": f"{int(df['outage_count'].sum()):,}",
        "Max in one county": f"{int(df['outage_count'].max()) if not df.empty else 0:,}",
    })

    fig = county_choropleth(df, value_col="outage_count", title="Outage Count")
    st.plotly_chart(fig, use_container_width=True)

    if sql:
        with st.expander("SQL"):
            st.code(sql, language="sql")
