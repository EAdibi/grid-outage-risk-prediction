import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class FeatureEngineering:
    def __init__(self, mongo_uri=None):
        if mongo_uri is None:
            password = os.getenv("MONGO_PASSWORD")
            mongo_uri = f"mongodb+srv://fa2927:{password}@cluster0.jphivpd.mongodb.net/"
        
        self.client = MongoClient(mongo_uri)
        self.db = self.client["big_data"]
    
    def load_outage_data(self):
        """
        Load outage data from MongoDB
        """
        logger.info("Loading outage data from MongoDB")
        
        outages = list(self.db.outages.find())
        df = pd.DataFrame(outages)
        
        logger.info(f"Loaded {len(df)} outage records")
        return df
    
    def load_weather_data(self):
        """
        Load weather events data from MongoDB
        """
        logger.info("Loading weather events data from MongoDB")
        
        weather = list(self.db.weather_events.find())
        df = pd.DataFrame(weather)
        
        logger.info(f"Loaded {len(df)} weather event records")
        return df
    
    def create_temporal_features(self, df, timestamp_col='start_time'):
        """
        Create temporal features from timestamp
        """
        logger.info("Creating temporal features")
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['hour'] = df[timestamp_col].dt.hour
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['season'] = df['month'].apply(lambda x: 
            'winter' if x in [12, 1, 2] else
            'spring' if x in [3, 4, 5] else
            'summer' if x in [6, 7, 8] else
            'fall'
        )
        
        return df
    
    def create_historical_features(self, df, county_col='location.county_fips'):
        """
        Create historical outage features by county
        """
        logger.info("Creating historical features")
        
        if isinstance(df[county_col].iloc[0], dict):
            df['county_fips'] = df[county_col].apply(lambda x: x if isinstance(x, str) else '')
        else:
            df['county_fips'] = df[county_col]
        
        county_history = df.groupby('county_fips').agg({
            'event_id': 'count',
            'customers_affected': ['sum', 'mean', 'max'],
            'duration_hours': ['mean', 'max']
        }).reset_index()
        
        county_history.columns = [
            'county_fips',
            'historical_outage_count',
            'total_customers_affected',
            'avg_customers_affected',
            'max_customers_affected',
            'avg_duration_hours',
            'max_duration_hours'
        ]
        
        df = df.merge(county_history, on='county_fips', how='left')
        
        return df
    
    def create_weather_features(self, outage_df, weather_df):
        """
        Create weather-related features by joining with weather data
        """
        logger.info("Creating weather features")
        
        if 'location.county_fips' in outage_df.columns:
            outage_df['county_fips'] = outage_df['location.county_fips'].apply(
                lambda x: x if isinstance(x, str) else ''
            )
        
        if 'location.county_fips' in weather_df.columns:
            weather_df['weather_county_fips'] = weather_df['location.county_fips'].apply(
                lambda x: x if isinstance(x, str) else ''
            )
        
        weather_df['begin_date'] = pd.to_datetime(weather_df['begin_date'], errors='coerce')
        
        weather_agg = weather_df.groupby('weather_county_fips').agg({
            'event_id': 'count',
            'magnitude': 'mean',
            'damage_property': 'sum'
        }).reset_index()
        
        weather_agg.columns = [
            'county_fips',
            'weather_event_count',
            'avg_weather_magnitude',
            'total_property_damage'
        ]
        
        outage_df = outage_df.merge(weather_agg, on='county_fips', how='left')
        
        outage_df['weather_event_count'] = outage_df['weather_event_count'].fillna(0)
        outage_df['avg_weather_magnitude'] = outage_df['avg_weather_magnitude'].fillna(0)
        outage_df['total_property_damage'] = outage_df['total_property_damage'].fillna(0)
        
        return outage_df
    
    def create_risk_labels(self, df):
        """
        Create risk labels for supervised learning
        """
        logger.info("Creating risk labels")
        
        df['customers_affected'] = pd.to_numeric(df['customers_affected'], errors='coerce').fillna(0)
        
        df['risk_level'] = pd.cut(
            df['customers_affected'],
            bins=[0, 1000, 10000, 50000, 100000, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        df['risk_score'] = df['customers_affected'] / 1000
        
        return df
    
    def create_lag_features(self, df, county_col='county_fips', days=[7, 14, 30]):
        """
        Create lag features for time series prediction
        """
        logger.info("Creating lag features")
        
        df = df.sort_values(['county_fips', 'start_time'])
        
        for day in days:
            df[f'outages_last_{day}_days'] = df.groupby(county_col)['event_id'].transform(
                lambda x: x.rolling(window=day, min_periods=1).count()
            )
        
        return df
    
    def engineer_all_features(self):
        """
        Run complete feature engineering pipeline
        """
        logger.info("Starting complete feature engineering pipeline")
        
        outage_df = self.load_outage_data()
        
        outage_df = self.create_temporal_features(outage_df)
        
        outage_df = self.create_historical_features(outage_df)
        
        try:
            weather_df = self.load_weather_data()
            outage_df = self.create_weather_features(outage_df, weather_df)
        except Exception as e:
            logger.warning(f"Weather features skipped: {e}")
        
        outage_df = self.create_risk_labels(outage_df)
        
        outage_df = outage_df.drop(columns=['_id'], errors='ignore')
        
        logger.info("Feature engineering completed")
        
        return outage_df
    
    def save_features(self, df, output_path='features/outage_features.parquet'):
        """
        Save engineered features to file
        """
        logger.info(f"Saving features to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        
        logger.info("Features saved successfully")


if __name__ == "__main__":
    fe = FeatureEngineering()
    features_df = fe.engineer_all_features()
    fe.save_features(features_df)
    
    print(f"\nFeature Summary:")
    print(f"Total records: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"\nFeature columns: {list(features_df.columns)}")
