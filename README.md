# ⚡ U.S. Power Grid Outage Risk Intelligence Platform

Big data platform for predicting U.S. power grid outage risk using MongoDB, machine learning, and interactive visualizations.

## Team Members
- Seungeun Lee (sl12190)
- Will Chanania (wdc9645)
- Elina Adibi (fa2927)
- Jithendra Puppala (jp8081)

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure MongoDB credentials
cp app/.env.example app/.env
# Edit app/.env with YOUR MongoDB username and password

# 3. Verify setup (optional)
python scripts/check_setup.py

# 4. Explore data (optional)
python scripts/explore_data.py

# 5. Launch dashboard
cd app && streamlit run main.py
```

## MongoDB Setup

Each team member uses their own credentials in `app/.env`:

```bash
MONGO_USERNAME=your_netid      # e.g., jp8081, fa2927
MONGO_PASSWORD=your_password
MONGO_CLUSTER=cluster0.jphivpd.mongodb.net
MONGO_DATABASE=big_data
```

## ML Pipeline

```bash
# Train models (creates features and trains models)
python ml_pipeline/feature_engineering.py
python ml_pipeline/model_training.py
```

## Project Structure

```
├── app/                    # Streamlit dashboard
│   ├── main.py            # Auto-discovers sections
│   ├── db.py              # MongoDB connection
│   ├── data.py            # Data queries
│   └── sections/          # Dashboard pages
├── ml_pipeline/           # Feature engineering & model training
├── scripts/               # Utility scripts
└── requirements.txt       # Dependencies
```

## Dashboard Features

- **Vulnerability Map** - County-level outage frequency
- **Top 10 High-Risk Counties** - Ranked by historical outages
- **Texas 2021 Case Study** - February 2021 winter storm analysis
- **Feature Importance Analysis** - Correlation & model explainability
- **Comparative Analysis** - State/time/event/region comparisons

## Data Collections (MongoDB)

- `outages` (526K docs) - Historical outage events
- `storm_events` (831K docs) - Weather data
- `grid_demand` (5.8M docs) - Grid load data
- `county_population` (38K docs) - Demographics
- `generators` (272K docs) - Power infrastructure
- `utilities` (25K docs) - Utility companies
- `disasters` (70K docs) - FEMA disasters
- `training_data` - ML features (generated)
- `predictions` - Model predictions (generated)

See [app/README.md](app/README.md) for dashboard development guide.
