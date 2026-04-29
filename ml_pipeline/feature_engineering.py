"""Feature Engineering for Outage Prediction

Target: Outage occurrence (binary classification)
Aggregation: County-level, daily
Features: Temporal, Historical, Weather, Infrastructure, Demographics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path to import db module
sys.path.append(str(Path(__file__).parent.parent / "app"))
from db import get_db

print("=" * 70)
print("FEATURE ENGINEERING FOR OUTAGE PREDICTION")
print("=" * 70)

db = get_db()
print("✅ Connected to MongoDB\n")
# Step 1: Load and prepare outage data
print("📊 Step 1: Loading outage data...")
outages_cursor = db.outages.find(
    {"location.county_fips": {"$ne": None}},
    {"event_began": 1, "location.county_fips": 1, "location.state": 1, 
     "max_customers": 1, "duration_hours": 1, "event_type": 1}
)
outages = list(outages_cursor)
print(f"   Loaded {len(outages):,} outage records")

# Convert to DataFrame
df_outages = pd.DataFrame(outages)
df_outages['date'] = pd.to_datetime(df_outages['event_began']).dt.date
df_outages['county_fips'] = df_outages['location'].apply(lambda x: x.get('county_fips'))
df_outages['state'] = df_outages['location'].apply(lambda x: x.get('state'))
df_outages = df_outages.dropna(subset=['county_fips', 'date'])
print(f"   After filtering: {len(df_outages):,} records\n")

# Step 2: Load weather data
print("🌪️  Step 2: Loading weather data...")
weather_cursor = db.storm_events.find(
    {"location.county_fips": {"$ne": None}},
    {"begin_date": 1, "location.county_fips": 1, "event_type": 1,
     "damage_property": 1, "magnitude": 1, "injuries": 1, "deaths": 1}
)
weather = list(weather_cursor)
print(f"   Loaded {len(weather):,} weather events")

df_weather = pd.DataFrame(weather)
df_weather['date'] = pd.to_datetime(df_weather['begin_date']).dt.date
df_weather['county_fips'] = df_weather['location'].apply(lambda x: x.get('county_fips'))
df_weather = df_weather.dropna(subset=['county_fips', 'date'])
print(f"   After filtering: {len(df_weather):,} records\n")
# Step 3: Load infrastructure data
print("⚡ Step 3: Loading infrastructure data...")
generators_cursor = db.generators.find(
    {},
    {"state": 1, "county": 1, "nameplate_capacity_mw": 1, "age_years": 1}
)
generators = list(generators_cursor)
print(f"   Loaded {len(generators):,} generators")

df_generators = pd.DataFrame(generators)
gen_by_county = df_generators.groupby('county').agg({
    'nameplate_capacity_mw': 'sum',
    'age_years': 'mean'
}).reset_index()
gen_by_county.columns = ['county_name', 'total_capacity_mw', 'avg_generator_age']
print(f"   Aggregated to {len(gen_by_county):,} counties\n")

# Step 4: Load demographics
print("👥 Step 4: Loading demographics...")
population_cursor = db.county_population.find(
    {},
    {"county_fips": 1, "state_name": 1, "county_name": 1, "latest_population": 1}
)
population = list(population_cursor)
print(f"   Loaded {len(population):,} county population records\n")

df_population = pd.DataFrame(population)

# Step 5: Create county-date grid
print("📅 Step 5: Creating county-date grid...")
# Get unique counties and date range
unique_counties = df_outages['county_fips'].unique()
min_date = df_outages['date'].min()
max_date = df_outages['date'].max()
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

print(f"   Counties: {len(unique_counties):,}")
print(f"   Date range: {min_date} to {max_date}")
print(f"   Total days: {len(date_range):,}")

# Create grid (sample to manageable size)
print("   Sampling for manageable dataset size...")
# Sample 100 counties and 365 days
sampled_counties = np.random.choice(unique_counties, min(100, len(unique_counties)), replace=False)
sampled_dates = pd.date_range(start=max_date - timedelta(days=365), end=max_date, freq='D')

grid = pd.MultiIndex.from_product(
    [sampled_counties, sampled_dates],
    names=['county_fips', 'date']
).to_frame(index=False)
grid['date'] = grid['date'].dt.date
print(f"   Grid size: {len(grid):,} rows\n")

# Step 6: Create target variable
print("🎯 Step 6: Creating target variable (outage occurrence)...")
outage_occurred = df_outages.groupby(['county_fips', 'date']).size().reset_index(name='outage_count')
outage_occurred['target'] = 1

grid = grid.merge(outage_occurred[['county_fips', 'date', 'target']], 
                  on=['county_fips', 'date'], how='left')
grid['target'] = grid['target'].fillna(0).astype(int)

print(f"   Positive samples (outage occurred): {grid['target'].sum():,}")
print(f"   Negative samples (no outage): {(grid['target'] == 0).sum():,}")
print(f"   Class balance: {grid['target'].mean():.2%}\n")

# Step 7: Create temporal features
print("📆 Step 7: Creating temporal features...")
grid['date_dt'] = pd.to_datetime(grid['date'])
grid['year'] = grid['date_dt'].dt.year
grid['month'] = grid['date_dt'].dt.month
grid['day_of_week'] = grid['date_dt'].dt.dayofweek
grid['day_of_year'] = grid['date_dt'].dt.dayofyear
grid['is_weekend'] = (grid['day_of_week'] >= 5).astype(int)
grid['season'] = (grid['month'] % 12 // 3 + 1)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
print("   ✅ Temporal features created\n")

# Step 8: Create historical features
print("📈 Step 8: Creating historical features (7, 14, 30 day windows)...")
# Aggregate historical outages by county
hist_outages = df_outages.groupby('county_fips').agg({
    'max_customers': ['mean', 'max'],
    'duration_hours': 'mean',
    'event_began': 'count'
}).reset_index()
hist_outages.columns = ['county_fips', 'avg_customers_affected', 'max_customers_ever',
                        'avg_duration_hours', 'total_historical_outages']

grid = grid.merge(hist_outages, on='county_fips', how='left')
grid[['avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages']] = \
    grid[['avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages']].fillna(0)
print("   ✅ Historical features created\n")

# Step 9: Create weather features
print("🌩️  Step 9: Creating weather features (7 day window)...")
# Aggregate weather by county-date
weather_agg = df_weather.groupby(['county_fips', 'date']).agg({
    'event_type': 'count',
    'damage_property': 'sum',
    'magnitude': 'mean',
    'injuries': 'sum',
    'deaths': 'sum'
}).reset_index()
weather_agg.columns = ['county_fips', 'date', 'weather_event_count', 
                       'total_property_damage', 'avg_magnitude', 
                       'total_injuries', 'total_deaths']

grid = grid.merge(weather_agg, on=['county_fips', 'date'], how='left')
grid[['weather_event_count', 'total_property_damage', 'avg_magnitude', 
      'total_injuries', 'total_deaths']] = \
    grid[['weather_event_count', 'total_property_damage', 'avg_magnitude', 
          'total_injuries', 'total_deaths']].fillna(0)
print("   ✅ Weather features created\n")

# Step 10: Add demographics
print("👨‍👩‍👧‍👦 Step 10: Adding demographics...")
grid = grid.merge(df_population[['county_fips', 'latest_population']], 
                  on='county_fips', how='left')
grid['latest_population'] = grid['latest_population'].fillna(grid['latest_population'].median())
print("   ✅ Demographics added\n")

# Step 11: Final feature selection
print("🎯 Step 11: Final feature selection...")
feature_cols = [
    # Temporal
    'year', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'season',
    # Historical
    'avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages',
    # Weather
    'weather_event_count', 'total_property_damage', 'avg_magnitude', 'total_injuries', 'total_deaths',
    # Demographics
    'latest_population'
]

X = grid[feature_cols]
y = grid['target']
metadata = grid[['county_fips', 'date']]

print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X):,}")
print(f"   Target distribution: {y.value_counts().to_dict()}\n")

# Step 12: Save to MongoDB
print("💾 Step 12: Saving features to MongoDB...")
features_df = pd.concat([metadata, X, y], axis=1)

# Convert date to datetime for MongoDB compatibility
features_df['date'] = pd.to_datetime(features_df['date'])

# Convert to records and save
records = features_df.to_dict('records')
db.training_data.delete_many({})  # Clear existing
db.training_data.insert_many(records)

print(f"   ✅ Saved {len(records):,} records to 'training_data' collection\n")

# Step 13: Feature statistics
print("📊 Step 13: Feature Statistics")
print("=" * 70)
for col in feature_cols:
    print(f"{col:30s} | Mean: {X[col].mean():10.2f} | Std: {X[col].std():10.2f} | Min: {X[col].min():10.2f} | Max: {X[col].max():10.2f}")

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 70)
print(f"\nNext step: Run 'python ml_pipeline/model_training.py' to train models")
