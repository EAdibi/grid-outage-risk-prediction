"""Feature Importance Analysis - Real Data"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from db import get_db

TITLE = "Feature Importance Analysis"
ICON = "🔍"
ORDER = 4


def show():
    st.header("🔍 Feature Importance Analysis")
    st.markdown("Understanding which factors most strongly predict power outages")
    
    db = get_db()
    
    # Check if model has been trained
    models_dir = Path(__file__).parent.parent.parent / "models"
    feature_importance_file = models_dir / "feature_importance.csv"
    
    if not feature_importance_file.exists():
        st.warning("⚠️ No trained model found. Please run the ML pipeline first:")
        st.code("""
# Run these commands:
python ml_pipeline/feature_engineering.py
python ml_pipeline/model_training.py
        """, language="bash")
        
        st.info("📊 Showing correlation analysis from raw data instead...")
        show_correlation_analysis(db)
    else:
        # Show both correlation and model-based importance
        tab1, tab2, tab3 = st.tabs(["📊 Correlation Analysis", "🤖 Model Feature Importance", "📈 Model Performance"])
        
        with tab1:
            show_correlation_analysis(db)
        
        with tab2:
            show_model_importance(feature_importance_file)
        
        with tab3:
            show_model_performance(db)


def show_correlation_analysis(db):
    """Show correlation between features and outages"""
    st.subheader("Correlation Analysis (Raw Data)")
    
    # Load sample data
    with st.spinner("Loading data from MongoDB..."):
        # Get outages - aggregate by county
        pipeline = [
            {"$match": {"location.county_fips": {"$ne": None}, "max_customers": {"$ne": None}}},
            {"$group": {
                "_id": "$location.county_fips",
                "outage_count": {"$sum": 1},
                "avg_customers": {"$avg": "$max_customers"},
                "max_customers": {"$max": "$max_customers"},
                "avg_duration": {"$avg": "$duration_hours"},
                "total_customers": {"$sum": "$max_customers"}
            }},
            {"$limit": 2000}
        ]
        outages_agg = list(db.outages.aggregate(pipeline))
        
        if not outages_agg:
            st.error("No outage data found")
            return
        
        df_outages = pd.DataFrame(outages_agg)
        df_outages.rename(columns={"_id": "county_fips"}, inplace=True)
        
        # Get weather data by county
        weather_pipeline = [
            {"$match": {"location.county_fips": {"$ne": None}}},
            {"$group": {
                "_id": "$location.county_fips",
                "weather_event_count": {"$sum": 1},
                "avg_damage": {"$avg": "$damage_property"},
                "total_injuries": {"$sum": "$injuries"},
                "total_deaths": {"$sum": "$deaths"}
            }}
        ]
        weather_agg = list(db.storm_events.aggregate(weather_pipeline))
        df_weather = pd.DataFrame(weather_agg)
        df_weather.rename(columns={"_id": "county_fips"}, inplace=True)
        
        # Get population
        population = list(db.county_population.find({}, {"county_fips": 1, "latest_population": 1}))
        df_pop = pd.DataFrame(population)
        
    # Merge data
    df_merged = df_outages.merge(df_weather, on="county_fips", how="left")
    df_merged = df_merged.merge(df_pop, on="county_fips", how="left")
    
    # Fill NaN with 0 for numeric columns only
    numeric_cols = ['weather_event_count', 'avg_damage', 'total_injuries', 'total_deaths', 'latest_population']
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
    
    # Calculate correlations with outage count
    feature_cols = ['avg_customers', 'avg_duration', 'weather_event_count', 
                    'avg_damage', 'total_injuries', 'total_deaths', 'latest_population']
    
    correlations = []
    for col in feature_cols:
        if col in df_merged.columns and df_merged[col].notna().sum() > 0:
            # Calculate correlation with outage count
            corr = df_merged[col].corr(df_merged['outage_count'])
            if not pd.isna(corr):
                correlations.append({'Feature': col, 'Correlation': abs(corr)})
    
    if not correlations:
        st.warning("Not enough data to calculate correlations")
        return
    
    df_corr = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    
    # Plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_corr,
            x='Correlation',
            y='Feature',
            orientation='h',
            title="Feature Correlations with Outage Occurrence",
            color='Correlation',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Counties Analyzed", f"{len(df_outages):,}")
        st.metric("Total Outages", f"{df_outages['outage_count'].sum():,.0f}")
        st.metric("Avg Customers/Outage", f"{df_outages['avg_customers'].mean():,.0f}")
        
        st.markdown("**Top Correlating Factors:**")
        for idx, row in df_corr.head(5).iterrows():
            st.write(f"- **{row['Feature']}**: {row['Correlation']:.3f}")


def show_model_importance(feature_importance_file):
    """Show feature importance from trained model"""
    st.subheader("Model-Based Feature Importance")
    
    # Load feature importance
    df_importance = pd.read_csv(feature_importance_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot feature importance
        fig = px.bar(
            df_importance.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 15 Most Important Features",
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Feature Categories")
        
        # Categorize features
        temporal = ['year', 'month', 'day_of_week', 'day_of_year', 'is_weekend', 'season']
        historical = ['avg_customers_affected', 'max_customers_ever', 'avg_duration_hours', 'total_historical_outages']
        weather = ['weather_event_count', 'total_property_damage', 'avg_magnitude', 'total_injuries', 'total_deaths']
        demographics = ['latest_population']
        
        categories = {
            'Temporal': df_importance[df_importance['feature'].isin(temporal)]['importance'].sum(),
            'Historical': df_importance[df_importance['feature'].isin(historical)]['importance'].sum(),
            'Weather': df_importance[df_importance['feature'].isin(weather)]['importance'].sum(),
            'Demographics': df_importance[df_importance['feature'].isin(demographics)]['importance'].sum()
        }
        
        fig_pie = px.pie(
            values=list(categories.values()),
            names=list(categories.keys()),
            title="Importance by Category"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("### Top 5 Features")
        for idx, row in df_importance.head(5).iterrows():
            st.metric(row['feature'], f"{row['importance']:.4f}")


def show_model_performance(db):
    """Show model performance metrics"""
    st.subheader("Model Performance")
    
    # Load predictions
    predictions = list(db.predictions.find().limit(1000))
    
    if not predictions:
        st.warning("No predictions found. Run model training first.")
        return
    
    df_pred = pd.DataFrame(predictions)
    
    # Load training data for actual values
    training_data = list(db.training_data.find().limit(1000))
    df_train = pd.DataFrame(training_data)
    
    # Merge
    df_eval = df_pred.merge(df_train[['county_fips', 'date', 'target']], 
                            on=['county_fips', 'date'], how='inner')
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(df_eval['target'], df_eval['predicted_outage'])
    precision = precision_score(df_eval['target'], df_eval['predicted_outage'], zero_division=0)
    recall = recall_score(df_eval['target'], df_eval['predicted_outage'], zero_division=0)
    f1 = f1_score(df_eval['target'], df_eval['predicted_outage'], zero_division=0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1 Score", f"{f1:.2%}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df_eval['target'], df_eval['predicted_outage'])
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Outage', 'Outage'],
        y=['No Outage', 'Outage'],
        title="Confusion Matrix",
        color_continuous_scale='Blues',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Probability distribution
    fig_prob = px.histogram(
        df_eval,
        x='outage_probability',
        color='target',
        title="Predicted Probability Distribution",
        labels={'target': 'Actual Outage', 'outage_probability': 'Predicted Probability'},
        barmode='overlay',
        nbins=50
    )
    st.plotly_chart(fig, use_container_width=True)
