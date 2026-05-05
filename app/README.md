# Dashboard (`app/`)

Streamlit dashboard reading from MongoDB Atlas. Live Analysis pages run Spark SQL queries; AI Analysis pages render model output.

## Run locally

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp app/.env.example app/.env       # set MONGO_USER + MONGO_PASSWORD
streamlit run app/main.py
```

`MONGO_USER` defaults to `fa2927` if unset; override it with your Atlas username (`wdc9645`, `sl12190`, `jp8081`).

## Spark setup

The Live Analysis pages need Java + pyspark. On Mac:

```bash
brew install openjdk@17
pip install pyspark   # already in requirements.txt
```

On the NYU JupyterHub Dataproc cluster Spark 4.1.1 is preinstalled. If Spark isn't available the page shows install guidance and the rest of the dashboard keeps working.

## Adding a page

1. Copy `sections/_template.py` to `sections/your_page.py`.
2. Set `TITLE`, `ICON`, `ORDER`, `SECTION` (`"Live Analysis"` or `"AI Analysis"`).
3. Write `show()`. Pull data through `data.py` (or `spark_data.py` for Spark DataFrames).
4. Save and refresh.

## Layout

- `main.py`: entrypoint, auto-discovers sections and groups them by `SECTION`.
- `db.py`: MongoDB client.
- `data.py`: cached pymongo helpers.
- `spark.py`: SparkSession factory and Java-home detection.
- `spark_data.py`: cached Spark DataFrames for outages and storm events.
- `components.py`: reusable widgets (`section_header`, `metric_row`, `county_choropleth`, `loading`).
- `cache_manager.py`: offline cache fallback when MongoDB Atlas is unreachable.
- `sections/`: one file per page; files starting with `_` are skipped.
