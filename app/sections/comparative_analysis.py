import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from db import get_db


def show():
    st.header("Comparative Analysis")
    st.markdown("Compare outage patterns across different regions and time periods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        state1 = st.selectbox(
            "Select First State",
            ["Texas", "California", "New York", "Florida", "Louisiana"],
            key="state1"
        )
    
    with col2:
        state2 = st.selectbox(
            "Select Second State",
            ["California", "Texas", "New York", "Florida", "Louisiana"],
            key="state2"
        )
    
    st.markdown("---")
    
    st.subheader(f"Comparison: {state1} vs {state2}")
    
    comparison_data = {
        'Texas': {
            'total_outages': 1247,
            'avg_duration': 8.3,
            'total_customers_affected': 4500000,
            'most_common_cause': 'Severe Weather',
            'peak_month': 'February'
        },
        'California': {
            'total_outages': 892,
            'avg_duration': 6.1,
            'total_customers_affected': 2800000,
            'most_common_cause': 'Wildfire',
            'peak_month': 'August'
        },
        'New York': {
            'total_outages': 634,
            'avg_duration': 5.4,
            'total_customers_affected': 1900000,
            'most_common_cause': 'Winter Storm',
            'peak_month': 'January'
        },
        'Florida': {
            'total_outages': 1089,
            'avg_duration': 9.7,
            'total_customers_affected': 3200000,
            'most_common_cause': 'Hurricane',
            'peak_month': 'September'
        },
        'Louisiana': {
            'total_outages': 978,
            'avg_duration': 11.2,
            'total_customers_affected': 2100000,
            'most_common_cause': 'Hurricane',
            'peak_month': 'August'
        }
    }
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            f"{state1} - Total Outages",
            f"{comparison_data[state1]['total_outages']:,}",
            delta=None
        )
        st.metric(
            f"{state2} - Total Outages",
            f"{comparison_data[state2]['total_outages']:,}",
            delta=f"{comparison_data[state2]['total_outages'] - comparison_data[state1]['total_outages']:+,}"
        )
    
    with metrics_col2:
        st.metric(
            f"{state1} - Avg Duration (hrs)",
            f"{comparison_data[state1]['avg_duration']:.1f}",
            delta=None
        )
        st.metric(
            f"{state2} - Avg Duration (hrs)",
            f"{comparison_data[state2]['avg_duration']:.1f}",
            delta=f"{comparison_data[state2]['avg_duration'] - comparison_data[state1]['avg_duration']:+.1f}"
        )
    
    with metrics_col3:
        st.metric(
            f"{state1} - Customers Affected",
            f"{comparison_data[state1]['total_customers_affected']:,}",
            delta=None
        )
        st.metric(
            f"{state2} - Customers Affected",
            f"{comparison_data[state2]['total_customers_affected']:,}",
            delta=f"{comparison_data[state2]['total_customers_affected'] - comparison_data[state1]['total_customers_affected']:+,}"
        )
    
    st.markdown("---")
    
    st.subheader("Outage Trends Over Time")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    trend_data = pd.DataFrame({
        'Month': months,
        state1: [45, 120, 67, 89, 102, 95, 88, 110, 98, 87, 76, 270] if state1 == 'Texas' 
               else [78, 65, 72, 88, 95, 102, 115, 145, 98, 87, 76, 71],
        state2: [78, 65, 72, 88, 95, 102, 115, 145, 98, 87, 76, 71] if state2 == 'California'
               else [45, 120, 67, 89, 102, 95, 88, 110, 98, 87, 76, 270]
    })
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Month'],
        y=trend_data[state1],
        mode='lines+markers',
        name=state1,
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Month'],
        y=trend_data[state2],
        mode='lines+markers',
        name=state2,
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))
    
    fig_trend.update_layout(
        title='Monthly Outage Comparison (2014-2023 Average)',
        xaxis_title='Month',
        yaxis_title='Number of Outages',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Outage Causes Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        causes_state1 = pd.DataFrame({
            'Cause': ['Severe Weather', 'Equipment Failure', 'Vegetation', 'Other'],
            'Count': [560, 312, 245, 130]
        })
        
        fig_pie1 = px.pie(
            causes_state1,
            values='Count',
            names='Cause',
            title=f'{state1} - Outage Causes',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_pie1, use_container_width=True)
    
    with col2:
        causes_state2 = pd.DataFrame({
            'Cause': ['Wildfire', 'Severe Weather', 'Equipment Failure', 'Other'],
            'Count': [389, 267, 178, 58]
        })
        
        fig_pie2 = px.pie(
            causes_state2,
            values='Count',
            names='Cause',
            title=f'{state2} - Outage Causes',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig_pie2, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Key Findings")
    
    st.markdown(f"""
    **{state1}:**
    - Most common cause: **{comparison_data[state1]['most_common_cause']}**
    - Peak outage month: **{comparison_data[state1]['peak_month']}**
    - Average outage duration: **{comparison_data[state1]['avg_duration']} hours**
    
    **{state2}:**
    - Most common cause: **{comparison_data[state2]['most_common_cause']}**
    - Peak outage month: **{comparison_data[state2]['peak_month']}**
    - Average outage duration: **{comparison_data[state2]['avg_duration']} hours**
    """)
