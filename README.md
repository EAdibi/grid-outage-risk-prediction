# ⚡ U.S. Power Grid Outage Risk Intelligence Platform

Big data platform for predicting U.S. power grid outage risk using MongoDB, machine learning, and interactive visualizations.

## Team Members
- **Seungeun Lee** (sl12190)
- **Will Chanania** (wdc9645)
- **Elina Adibi** (fa2927)
- **Jithendra Puppala** (jp8081)

## Project Overview

This platform integrates power grid outage data, weather events, and grid demand data to predict where the next major grid failure is most likely to occur.

## Dashboard Features

### Existing (by teammate)
- **Vulnerability Map**: County-level outage frequency choropleth
- **Top 10 High-Risk Counties**: Ranked by historical outage count
- **Texas 2021 Case Study**: February 2021 winter storm analysis

### New Visualizations
- **Feature Importance Analysis**: Model explainability and performance metrics
- **Comparative Analysis**: State-by-state outage pattern comparison

## MongoDB Collections

- **`outages`** (523K docs) - Historical outage events
- **`storm_events`** (833K docs) - Weather data
- **`grid_demand`** (5.8M docs) - Grid load data
- **`county_population`** (38K docs) - Population data
- **`generators`** (272K docs) - Power generation facilities
- **`utilities`** (25K docs) - Utility companies
- **`disasters`** (70K docs) - Disaster events
- **`predictions`** - Model predictions
- **`training_data`** - ML training data

## Setup

### Prerequisites
- Python 3.9+
- MongoDB Atlas access (credentials from team)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (use YOUR credentials)
cp app/.env.template app/.env
# Edit app/.env with YOUR MongoDB username and password

# Test connection
python explore_data.py

# Launch dashboard
cd app
streamlit run main.py
```

### MongoDB Credentials

Each team member uses their own MongoDB Atlas credentials:

```bash
# In app/.env
MONGO_USERNAME=your_netid      # e.g., jp8081, fa2927, sl12190, wdc9645
MONGO_PASSWORD=your_password
MONGO_CLUSTER=cluster0.jphivpd.mongodb.net
MONGO_DATABASE=big_data
```

See `TEAM_SETUP.md` for detailed instructions.

## Project Structure

```
grid-outage-risk-prediction/
├── app/                      # Streamlit dashboard
│   ├── main.py              # Main app
│   ├── db.py                # MongoDB connection
│   ├── .env.template        # Environment template
│   └── sections/            # Dashboard pages
│       ├── map.py
│       ├── top_counties.py
│       ├── texas_2021.py
│       ├── feature_importance.py
│       └── comparative_analysis.py
├── ml_pipeline/             # Machine learning
│   ├── feature_engineering.py
│   └── model_training.py
└── requirements.txt         # Python dependencies
```

## ML Pipeline

Train predictive models from MongoDB data:

```bash
# Generate features
python ml_pipeline/feature_engineering.py

# Train models
python ml_pipeline/model_training.py
```

## Technology Stack

- **Database**: MongoDB Atlas
- **Visualization**: Streamlit, Plotly
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy
