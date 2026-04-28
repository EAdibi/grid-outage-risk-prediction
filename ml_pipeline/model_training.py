import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
import joblib
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutageRiskModel:
    def __init__(self, features_path='features/outage_features.parquet'):
        self.features_path = features_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        
    def load_features(self):
        """
        Load engineered features
        """
        logger.info(f"Loading features from {self.features_path}")
        
        df = pd.read_parquet(self.features_path)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def prepare_data(self, df, target_col='risk_level', task='classification'):
        """
        Prepare data for model training
        """
        logger.info("Preparing data for training")
        
        feature_cols = [
            'year', 'month', 'day', 'day_of_week', 'hour', 'quarter', 'is_weekend',
            'historical_outage_count', 'avg_customers_affected', 'max_customers_affected',
            'avg_duration_hours', 'max_duration_hours',
            'weather_event_count', 'avg_weather_magnitude', 'total_property_damage'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        X = df[available_features].copy()
        
        X = X.fillna(0)
        
        if task == 'classification':
            y = df[target_col].fillna('very_low')
            y = self.label_encoder.fit_transform(y)
        else:
            y = df['risk_score'].fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple classification models for risk level prediction
        """
        logger.info("Training classification models")
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}")
            
            print(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred, 
                                       target_names=self.label_encoder.classes_))
        
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple regression models for risk score prediction
        """
        logger.info("Training regression models")
        
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost_reg': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm_reg': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return results
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from trained model
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        return None
    
    def save_models(self, models, output_dir='models'):
        """
        Save trained models to disk
        """
        logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, result in models.items():
            model_path = os.path.join(output_dir, f"{name}_{timestamp}.pkl")
            joblib.dump(result['model'], model_path)
            logger.info(f"Saved {name} to {model_path}")
        
        scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        encoder_path = os.path.join(output_dir, f"label_encoder_{timestamp}.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        logger.info("All models saved successfully")
    
    def run_training_pipeline(self, task='classification'):
        """
        Run complete model training pipeline
        """
        logger.info(f"Starting model training pipeline (task: {task})")
        
        df = self.load_features()
        
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df, task=task)
        
        if task == 'classification':
            results = self.train_classification_models(X_train, X_test, y_train, y_test)
        else:
            results = self.train_regression_models(X_train, X_test, y_train, y_test)
        
        best_model_name = max(results.items(), 
                             key=lambda x: x[1].get('accuracy', x[1].get('r2', 0)))[0]
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name}")
        
        importance_df = self.get_feature_importance(best_model, feature_names)
        if importance_df is not None:
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
        
        self.save_models(results)
        
        return results, feature_names


if __name__ == "__main__":
    trainer = OutageRiskModel()
    
    print("Training Classification Models...")
    classification_results, feature_names = trainer.run_training_pipeline(task='classification')
    
    print("\n" + "="*80 + "\n")
    
    print("Training Regression Models...")
    regression_results, _ = trainer.run_training_pipeline(task='regression')
