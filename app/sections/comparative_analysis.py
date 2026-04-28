"""Comparative Analysis - Real Data"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from db import get_db
from datetime import datetime

TITLE = "Comparative Analysis"
ICON = "⚖️"
ORDER = 5


def show():
    st.header("⚖️ Comparative Analysis")
    st.markdown("Compare outage patterns across states, time periods, event types, and regions")
    
    db = get_db()
    
    # Tabs for different comparisons
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ State vs State",
        "📅 Time Period Comparison",
        "⚡ Event Type Analysis",
        "🌎 Regional Comparison"
    ])
    
    with tab1:
        show_state_comparison(db)
    
    with tab2:
        show_time_comparison(db)
    
    with tab3:
        show_event_type_comparison(db)
    
    with tab4:
        show_regional_comparison(db)


def show_state_comparison(db):
    """Compare two states"""
    st.subheader("State-by-State Comparison")
    
    # Get list of states
    states = db.outages.distinct("location.state")
    states = [s for s in states if s]  # Remove None values
    states.sort()
    
    col1, col2 = st.columns(2)
    with col1:
        state1 = st.selectbox("Select First State", states, index=states.index("California") if "California" in states else 0)
    with col2:
        state2 = st.selectbox("Select Second State", states, index=states.index("Texas") if "Texas" in states else 1)
    
    if state1 == state2:
        st.warning("Please select two different states")
        return
    
    # Load data for both states
    with st.spinner("Loading data..."):
        data1 = list(db.outages.find({"location.state": state1}))
        data2 = list(db.outages.find({"location.state": state2}))
    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Metrics comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"{state1} - Total Outages", f"{len(df1):,}")
        st.metric(f"{state2} - Total Outages", f"{len(df2):,}")
    
    with col2:
        avg_customers1 = df1['max_customers'].mean() if 'max_customers' in df1.columns else 0
        avg_customers2 = df2['max_customers'].mean() if 'max_customers' in df2.columns else 0
        st.metric(f"{state1} - Avg Customers Affected", f"{avg_customers1:,.0f}")
        st.metric(f"{state2} - Avg Customers Affected", f"{avg_customers2:,.0f}")
    
    with col3:
        avg_duration1 = df1['duration_hours'].mean() if 'duration_hours' in df1.columns else 0
        avg_duration2 = df2['duration_hours'].mean() if 'duration_hours' in df2.columns else 0
        st.metric(f"{state1} - Avg Duration (hrs)", f"{avg_duration1:.1f}")
        st.metric(f"{state2} - Avg Duration (hrs)", f"{avg_duration2:.1f}")
    
    # Time series comparison
    df1['date'] = pd.to_datetime(df1['event_began']).dt.to_period('M')
    df2['date'] = pd.to_datetime(df2['event_began']).dt.to_period('M')
    
    ts1 = df1.groupby('date').size().reset_index(name='count')
    ts1['state'] = state1
    ts2 = df2.groupby('date').size().reset_index(name='count')
    ts2['state'] = state2
    
    ts_combined = pd.concat([ts1, ts2])
    ts_combined['date'] = ts_combined['date'].astype(str)
    
    fig = px.line(
        ts_combined,
        x='date',
        y='count',
        color='state',
        title=f"Outage Trends: {state1} vs {state2}",
        labels={'count': 'Number of Outages', 'date': 'Month'}
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Event type breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        if 'event_type' in df1.columns:
            event_counts1 = df1['event_type'].value_counts().head(10)
            fig1 = px.pie(
                values=event_counts1.values,
                names=event_counts1.index,
                title=f"{state1} - Top Event Types"
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'event_type' in df2.columns:
            event_counts2 = df2['event_type'].value_counts().head(10)
            fig2 = px.pie(
                values=event_counts2.values,
                names=event_counts2.index,
                title=f"{state2} - Top Event Types"
            )
            st.plotly_chart(fig2, use_container_width=True)


def show_time_comparison(db):
    """Compare different time periods"""
    st.subheader("Time Period Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox("Select First Year", range(2014, 2025), index=6)  # 2020
    with col2:
        year2 = st.selectbox("Select Second Year", range(2014, 2025), index=7)  # 2021
    
    # Load data for both years
    with st.spinner("Loading data..."):
        data1 = list(db.outages.find({
            "event_began": {
                "$gte": datetime(year1, 1, 1),
                "$lt": datetime(year1 + 1, 1, 1)
            }
        }))
        data2 = list(db.outages.find({
            "event_began": {
                "$gte": datetime(year2, 1, 1),
                "$lt": datetime(year2 + 1, 1, 1)
            }
        }))
    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{year1} - Total Outages", f"{len(df1):,}")
    col2.metric(f"{year2} - Total Outages", f"{len(df2):,}", 
                delta=f"{((len(df2) - len(df1)) / len(df1) * 100):.1f}%" if len(df1) > 0 else "N/A")
    
    total_customers1 = df1['max_customers'].sum() if 'max_customers' in df1.columns else 0
    total_customers2 = df2['max_customers'].sum() if 'max_customers' in df2.columns else 0
    col3.metric(f"{year1} - Total Customers", f"{total_customers1:,.0f}")
    col4.metric(f"{year2} - Total Customers", f"{total_customers2:,.0f}",
                delta=f"{((total_customers2 - total_customers1) / total_customers1 * 100):.1f}%" if total_customers1 > 0 else "N/A")
    
    # Monthly breakdown
    df1['month'] = pd.to_datetime(df1['event_began']).dt.month
    df2['month'] = pd.to_datetime(df2['event_began']).dt.month
    
    monthly1 = df1.groupby('month').size().reset_index(name='count')
    monthly1['year'] = year1
    monthly2 = df2.groupby('month').size().reset_index(name='count')
    monthly2['year'] = year2
    
    monthly_combined = pd.concat([monthly1, monthly2])
    
    fig = px.bar(
        monthly_combined,
        x='month',
        y='count',
        color='year',
        barmode='group',
        title=f"Monthly Outage Distribution: {year1} vs {year2}",
        labels={'month': 'Month', 'count': 'Number of Outages'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_event_type_comparison(db):
    """Compare different event types"""
    st.subheader("Event Type Analysis")
    
    # Get event type distribution
    with st.spinner("Loading data..."):
        pipeline = [
            {"$group": {
                "_id": "$event_type",
                "count": {"$sum": 1},
                "avg_customers": {"$avg": "$max_customers"},
                "avg_duration": {"$avg": "$duration_hours"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 15}
        ]
        event_data = list(db.outages.aggregate(pipeline))
    
    df_events = pd.DataFrame(event_data)
    df_events.rename(columns={"_id": "event_type"}, inplace=True)
    
    # Event type frequency
    fig1 = px.bar(
        df_events,
        x='event_type',
        y='count',
        title="Top 15 Event Types by Frequency",
        labels={'count': 'Number of Outages', 'event_type': 'Event Type'},
        color='count',
        color_continuous_scale='Reds'
    )
    fig1.update_xaxis(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Impact comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.bar(
            df_events.head(10),
            x='event_type',
            y='avg_customers',
            title="Average Customers Affected by Event Type",
            labels={'avg_customers': 'Avg Customers', 'event_type': 'Event Type'},
            color='avg_customers',
            color_continuous_scale='Blues'
        )
        fig2.update_xaxis(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.bar(
            df_events.head(10),
            x='event_type',
            y='avg_duration',
            title="Average Duration by Event Type",
            labels={'avg_duration': 'Avg Duration (hours)', 'event_type': 'Event Type'},
            color='avg_duration',
            color_continuous_scale='Greens'
        )
        fig3.update_xaxis(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)


def show_regional_comparison(db):
    """Compare different regions"""
    st.subheader("Regional Comparison")
    
    # Define regions
    regions = {
        'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 
                     'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
        'Southeast': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 
                     'South Carolina', 'Virginia', 'West Virginia', 'Alabama', 'Kentucky', 
                     'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana'],
        'Midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 
                   'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
        'Southwest': ['Arizona', 'New Mexico', 'Oklahoma', 'Texas'],
        'West': ['Colorado', 'Idaho', 'Montana', 'Nevada', 'Utah', 'Wyoming', 
                'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
    }
    
    # Aggregate by region
    with st.spinner("Loading regional data..."):
        regional_data = []
        for region, states in regions.items():
            count = db.outages.count_documents({"location.state": {"$in": states}})
            
            # Get average customers affected
            pipeline = [
                {"$match": {"location.state": {"$in": states}}},
                {"$group": {
                    "_id": None,
                    "avg_customers": {"$avg": "$max_customers"},
                    "total_customers": {"$sum": "$max_customers"}
                }}
            ]
            agg_result = list(db.outages.aggregate(pipeline))
            
            avg_customers = agg_result[0]['avg_customers'] if agg_result else 0
            total_customers = agg_result[0]['total_customers'] if agg_result else 0
            
            regional_data.append({
                'region': region,
                'outage_count': count,
                'avg_customers_affected': avg_customers,
                'total_customers_affected': total_customers
            })
    
    df_regional = pd.DataFrame(regional_data)
    
    # Regional comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            df_regional,
            x='region',
            y='outage_count',
            title="Total Outages by Region",
            labels={'outage_count': 'Number of Outages', 'region': 'Region'},
            color='outage_count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            df_regional,
            x='region',
            y='total_customers_affected',
            title="Total Customers Affected by Region",
            labels={'total_customers_affected': 'Total Customers', 'region': 'Region'},
            color='total_customers_affected',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Regional metrics table
    st.dataframe(
        df_regional.style.format({
            'outage_count': '{:,.0f}',
            'avg_customers_affected': '{:,.0f}',
            'total_customers_affected': '{:,.0f}'
        }),
        use_container_width=True
    )
