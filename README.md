# grid-outage-risk-prediction

Big data platform for predicting U.S. power grid outage risk using NYISO, NOAA, and DOE datasets with Spark, MongoDB, and machine learning.

See [project.md](project.md) for the full project plan, schemas, and team workflow.

## Quick start (dashboard)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp app/.env.example app/.env       # then fill in MONGO_USER + MONGO_PASSWORD in .env file
streamlit run app/main.py
```

See [app/README.md](app/README.md) for full dashboard docs and how to add new pages.
