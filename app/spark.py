import os
from pathlib import Path

import streamlit as st


JAVA_PATHS = [
    "/opt/homebrew/opt/openjdk@17",
    "/opt/homebrew/opt/openjdk",
    "/usr/local/opt/openjdk@17",
    "/usr/local/opt/openjdk",
]


def set_java_home():
    if os.environ.get("JAVA_HOME"):
        return
    for p in JAVA_PATHS:
        if (Path(p) / "bin" / "java").exists():
            os.environ["JAVA_HOME"] = p
            os.environ["PATH"] = f"{p}/bin:" + os.environ.get("PATH", "")
            return


@st.cache_resource(show_spinner=False)
def get_spark():
    try:
        set_java_home()
        from pyspark.sql import SparkSession
        spark = (
            SparkSession.builder
            .appName("OutageRisk")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as e:
        return e


def spark_ready():
    return hasattr(get_spark(), "sql")


def show_install_help():
    st.error("PySpark is not installed.")
    st.markdown(
        "1. Install Java 17: `brew install openjdk@17`\n"
        "2. `pip install pyspark`\n"
        "3. Restart the app"
    )
    err = get_spark()
    if isinstance(err, Exception):
        with st.expander("Error"):
            st.code(f"{type(err).__name__}: {err}")
