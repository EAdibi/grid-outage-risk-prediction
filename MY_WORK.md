# My Work: Visualization & ML Model Training

## 🎯 My Responsibilities

1. **Dashboard Visualizations** - Build new interactive pages
2. **ML Model Training** - Predict outage risk from MongoDB data

## 📊 Available Data (MongoDB)

### Collections Overview
- **`outages`** (523K docs) - Main dataset for predictions
- **`storm_events`** (833K docs) - Weather correlation
- **`grid_demand`** (5.8M docs) - Grid load patterns
- **`county_population`** (38K docs) - Demographics
- **`generators`** (272K docs) - Power infrastructure
- **`utilities`** (25K docs) - Utility info
- **`disasters`** (70K docs) - Disaster events
- **`predictions`** - Store model outputs here
- **`training_data`** - Store processed features here

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Copy environment template
cp app/.env.template app/.env

# Edit app/.env with MongoDB password (get from team)
# MONGO_PASSWORD=actual_password_here
```

### 2. Explore Data

```bash
# Run data exploration script
python explore_data.py
```

This will show you:
- All available collections
- Sample documents from each collection
- Field names and types

### 3. Launch Dashboard

```bash
cd app
streamlit run main.py
```

## 📈 Dashboard Pages

### ✅ Existing (by teammate - DO NOT MODIFY)
1. **Vulnerability Map** (`app/sections/map.py`)
2. **Top 10 High-Risk Counties** (`app/sections/top_counties.py`)
3. **Texas 2021 Case Study** (`app/sections/texas_2021.py`)

### 🆕 My New Pages
4. **Feature Importance Analysis** (`app/sections/feature_importance.py`)
   - Model performance metrics
   - Feature importance rankings
   - Correlation analysis

5. **Comparative Analysis** (`app/sections/comparative_analysis.py`)
   - State-by-state comparison
   - Temporal trends
   - Cause breakdown

### 💡 Ideas for More Visualizations

Based on the data, you could add:

1. **Time Series Analysis**
   - Outage trends over time
   - Seasonal patterns
   - Peak hours/days

2. **Weather Correlation Dashboard**
   - Outages vs. storm events
   - Weather type impact
   - Geographic correlation

3. **Grid Demand Analysis**
   - Load patterns vs. outages
   - Peak demand periods
   - Zone-level analysis

4. **Prediction Dashboard**
   - Real-time risk scores
   - County-level predictions
   - Confidence intervals

5. **Infrastructure Analysis**
   - Generator capacity vs. outages
   - Utility performance
   - Age of infrastructure

## 🤖 ML Model Training

### Current Pipeline

1. **Feature Engineering** (`ml_pipeline/feature_engineering.py`)
   - Loads data from MongoDB
   - Creates temporal, historical, and weather features
   - Saves to `features/outage_features.parquet`

2. **Model Training** (`ml_pipeline/model_training.py`)
   - Trains multiple models (Random Forest, XGBoost, LightGBM)
   - Classification: Predict risk level (very_low to very_high)
   - Regression: Predict risk score (continuous)
   - Saves models to `models/` directory

### Running the Pipeline

```bash
# Step 1: Generate features
python ml_pipeline/feature_engineering.py

# Step 2: Train models
python ml_pipeline/model_training.py
```

### Model Objectives

You can train models to predict:

1. **Outage Risk Level** (Classification)
   - Input: County, time, weather, historical data
   - Output: very_low, low, medium, high, very_high

2. **Customers Affected** (Regression)
   - Input: Same features
   - Output: Number of customers affected

3. **Outage Duration** (Regression)
   - Input: Same features
   - Output: Hours until restoration

4. **Outage Probability** (Binary Classification)
   - Input: Same features
   - Output: Will outage occur? (yes/no)

## 📝 Next Steps

### For Visualizations

1. Run `python explore_data.py` to see actual data structure
2. Pick a visualization idea from the list above
3. Create new file in `app/sections/your_page.py`
4. Add navigation in `app/main.py`
5. Test with `streamlit run main.py`

### For ML Model

1. Decide what to predict (risk level, customers affected, etc.)
2. Update `ml_pipeline/feature_engineering.py` with relevant features
3. Update `ml_pipeline/model_training.py` with target variable
4. Run training pipeline
5. Save predictions back to MongoDB `predictions` collection
6. Create dashboard page to visualize predictions

## 🔧 Useful Code Snippets

### Query MongoDB

```python
from app.db import get_db

db = get_db()

# Get all outages in Texas
texas_outages = list(db.outages.find({"location.state": "Texas"}))

# Count outages by state
pipeline = [
    {"$group": {"_id": "$location.state", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
results = list(db.outages.aggregate(pipeline))
```

### Create Plotly Chart

```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame(data)
fig = px.bar(df, x='state', y='count', title='Outages by State')
st.plotly_chart(fig, use_container_width=True)
```

### Add New Dashboard Page

```python
# 1. Create app/sections/my_page.py
import streamlit as st

def show():
    st.header("My New Page")
    st.write("Content here")

# 2. Update app/main.py
page = st.sidebar.radio("Go to", [
    "Vulnerability Map",
    "Top 10 High-Risk Counties",
    "Texas 2021 Case Study",
    "Feature Importance Analysis",
    "Comparative Analysis",
    "My New Page"  # Add this
])

if page == "My New Page":
    from sections.my_page import show
    show()
```

## 📊 Sample Queries to Try

```python
from app.db import get_db
import pandas as pd

db = get_db()

# 1. Outages over time
pipeline = [
    {"$group": {
        "_id": {"$year": "$start_time"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}}
]
yearly_outages = list(db.outages.aggregate(pipeline))

# 2. Top counties by outage count
pipeline = [
    {"$group": {
        "_id": "$location.county_fips",
        "county_name": {"$first": "$location.county_name"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"count": -1}},
    {"$limit": 20}
]
top_counties = list(db.outages.aggregate(pipeline))

# 3. Weather events by type
pipeline = [
    {"$group": {
        "_id": "$event_type",
        "count": {"$sum": 1}
    }},
    {"$sort": {"count": -1}}
]
weather_types = list(db.storm_events.aggregate(pipeline))
```

## 🤝 Team Collaboration

- **Your work**: Visualizations + ML model
- **Other teammates**: Data ingestion, Spark processing, Kafka streaming
- **Shared**: MongoDB database, requirements.txt, .env configuration

### Git Workflow

```bash
# Pull latest changes from team
git pull origin main

# Create your branch
git checkout -b visualization-ml

# Make your changes
# ... work on dashboard and model ...

# Commit your work
git add app/sections/ ml_pipeline/
git commit -m "Add new visualizations and ML model"

# Push to your branch
git push origin visualization-ml

# Create pull request for team review
```

## 📞 Need Help?

1. Check actual data structure: `python explore_data.py`
2. Test MongoDB connection: `python -c "from app.db import get_db; print(get_db().list_collection_names())"`
3. Review existing dashboard code in `app/sections/`
4. Ask teammates for data schema details
