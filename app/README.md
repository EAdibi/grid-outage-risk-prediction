# Dashboard (`app/`)

Streamlit dashboard for the U.S. Power Grid Outage Risk platform. Reads from MongoDB Atlas.

## Run locally

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp app/.env.example app/.env       # then edit: set MONGO_USER + MONGO_PASSWORD
streamlit run app/main.py
```

`MONGO_USER` defaults to `fa2927` if unset; override it with your own Atlas username (`wdc9645`, `sl12190`, `jp8081`).

## Adding a new page

1. Copy [sections/_template.py](sections/_template.py) → `sections/your_page.py`.
2. Edit `TITLE`, `ICON`, `ORDER`, and write your `show()` function.
3. If you need data, add a `@st.cache_data` helper to [data.py](data.py) — don't query Mongo from inside a section.
4. Reuse widgets from [components.py](components.py) (`county_choropleth`, `section_header`, `metric_row`, `state_filter`).
5. Save and refresh — the sidebar picks up the new page automatically.

## Layout

- [main.py](main.py) — entrypoint. Auto-discovers any `sections/*.py` that exports `show()`.
- [db.py](db.py) — MongoDB client (reads `MONGO_USER` + `MONGO_PASSWORD` from `app/.env`).
- [data.py](data.py) — cached query helpers. The only place that talks to collections.
- [components.py](components.py) — reusable UI widgets and chart wrappers.
- [sections/](sections/) — one file per page. Files starting with `_` are skipped.

## Conventions

Every section module exports:

| Name | Type | Meaning |
|------|------|---------|
| `TITLE` | `str` | Display name in the sidebar |
| `ICON` | `str` (emoji) | Icon shown next to the title |
| `ORDER` | `int` | Sort order (lower = higher in sidebar) |
| `show()` | callable | Renders the page |

Other rules:
- Helpers in `data.py` return `pd.DataFrame` with documented columns.
- Cache TTLs default to 1 hour for batch data, 10 minutes for predictions.
- Sections never import `pymongo` directly — go through `data.py`.

## Page assignments

Maps the five proposal deliverables (plus the case study) to section files.

| Deliverable | File | Status |
|-------------|------|--------|
| Vulnerability Map | `sections/map.py` | scaffolded |
| Top 10 High-Risk Counties | `sections/top_counties.py` | scaffolded |
| Early Warning System | `sections/early_warning.py` | TODO |
| Comparative Analysis | `sections/comparative.py` | TODO |
| Feature Importance | `sections/feature_importance.py` | TODO |
| Texas 2021 Case Study | `sections/texas_2021.py` | scaffolded |
