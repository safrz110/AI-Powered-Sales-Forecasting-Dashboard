import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("sales_forecasting_Regression_Model.pkl")

st.write("Expected Features:")
st.write(model.feature_names_in_)
# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

[data-testid="stMetricValue"]{
    font-size:28px;
    font-weight:bold;
}

.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("📈 AI Sales Forecasting Dashboard")
st.write("Predict future sales using Machine Learning.")

st.divider()

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Input Parameters")

base_demand = st.sidebar.number_input(
    "Base Demand",
    min_value=0.0,
    value=100.0
)

promotion_active = st.sidebar.selectbox(
    "Promotion Active",
    [0, 1]
)

holiday_flag = st.sidebar.selectbox(
    "Holiday Flag",
    [0, 1]
)

weekly_seasonality_component = st.sidebar.number_input(
    "Weekly Seasonality",
    value=10.0
)

yearly_seasonality_component = st.sidebar.number_input(
    "Yearly Seasonality",
    value=20.0
)
store_id = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    value=1
)

product_id = st.sidebar.number_input(
    "Product ID",
    min_value=1,
    value=1
)

store_avg_sales = st.sidebar.number_input(
    "Store Average Sales",
    value=1000.0
)

product_avg_sales = st.sidebar.number_input(
    "Product Average Sales",
    value=500.0
)

date = st.sidebar.date_input("Date")
# ----------------------------
# Input Summary
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Input Summary")

    st.metric("Base Demand", f"{base_demand:,.0f}")
    st.metric("Promotion Active", promotion_active)
    st.metric("Holiday Flag", holiday_flag)

with col2:
    st.subheader("📊 Seasonality")

    st.metric("Weekly Seasonality", weekly_seasonality_component)
    st.metric("Yearly Seasonality", yearly_seasonality_component)

st.divider()

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🚀 Predict Sales"):

    # Feature Engineering
    log_base_demand = np.log1p(base_demand)
    sqrt_base_demand = np.sqrt(base_demand)

    promo_strength = promotion_active * base_demand
    holiday_strength = holiday_flag * base_demand

    seasonality_weighted = (
        weekly_seasonality_component * 0.7
        + yearly_seasonality_component * 1.3
    )

    seasonality_demand = seasonality_weighted * base_demand

    promo_seasonality = promotion_active * seasonality_weighted

    holiday_seasonality = holiday_flag * seasonality_weighted

    seasonality_ratio = seasonality_demand / (base_demand + 1)

    promo_ratio = promo_strength / (base_demand + 1)

    final_demand_signal = (
        base_demand
        + promo_strength
        + holiday_strength
        + seasonality_weighted
    )

    date_str = str(date)

    # Model Input
    data = pd.DataFrame({
        "store_id": [store_id],
        "product_id": [product_id],
        "promotion_active": [promotion_active],
        "holiday_flag": [holiday_flag],
        "base_demand": [base_demand],
        "log_base_demand": [log_base_demand],
        "sqrt_base_demand": [sqrt_base_demand],
        "promo_strength": [promo_strength],
        "holiday_strength": [holiday_strength],
        "seasonality_weighted": [seasonality_weighted],
        "seasonality_demand": [seasonality_demand],
        "promo_seasonality": [promo_seasonality],
        "holiday_seasonality": [holiday_seasonality],
        "store_avg_sales": [store_avg_sales],
        "product_avg_sales": [product_avg_sales],
        "seasonality_ratio": [seasonality_ratio],
        "promo_ratio": [promo_ratio],
        "final_demand_signal": [final_demand_signal],
        "date": [date_str]
    })

    st.write("Columns sent to model:")
    st.write(data.columns.tolist())

    # Prediction
    pred_log = model.predict(data)
    pred = np.expm1(pred_log)

    st.success("Prediction Completed Successfully ✅")

    st.metric(
        "📈 Predicted Sales",
        f"{pred[0]:,.2f}"
    )

    # ----------------------------
    # KPI Cards
    # ----------------------------
    st.subheader("📊 Business KPIs")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Base Demand",
            f"{base_demand:,.0f}"
        )

    with k2:
        st.metric(
            "Promotion Impact",
            f"{promo_strength:,.0f}"
        )

    with k3:
        st.metric(
            "Seasonality Impact",
            f"{seasonality_demand:,.0f}"
        )

    # ----------------------------
    # Download CSV
    # ----------------------------
    result_df = pd.DataFrame({
        "Predicted Sales": [pred[0]],
        "Base Demand": [base_demand],
        "Promotion Active": [promotion_active],
        "Holiday Flag": [holiday_flag]
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="sales_prediction.csv",
        mime="text/csv"
    )

    # ----------------------------
    # Plotly Chart
    # ----------------------------
    trend_df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Sales": [
            pred[0] * 0.75,
            pred[0] * 0.85,
            pred[0] * 0.95,
            pred[0] * 1.05,
            pred[0]
        ]
    })

    fig = px.line(
        trend_df,
        x="Month",
        y="Sales",
        markers=True,
        title="Projected Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Show Features
    # ----------------------------
    with st.expander("🔍 View Processed Features"):
        st.dataframe(data)