import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math
import time
import datetime
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SalesPulse AI — Automated Sales Forecasting Classification System"
",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at top left, #0f1923 0%, #080d13 55%, #0a1218 100%);
    color: #d4e8f5;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(120deg, #0d2137 0%, #0a3d62 50%, #1a5276 100%);
    border-radius: 20px;
    padding: 40px 52px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(52,152,219,0.3);
    box-shadow: 0 12px 60px rgba(52,152,219,0.2);
}
.hero::before {
    content: '📈';
    position: absolute;
    right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.12;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 1px;
    margin: 0 0 6px 0;
    line-height: 1;
}
.hero-title span { color: #3498db; }
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin: 0;
    letter-spacing: 1.5px;
}

/* ── Stat tiles ── */
.stat-grid { display: flex; gap: 16px; margin-bottom: 28px; }
.stat-tile {
    flex: 1;
    background: rgba(52,152,219,0.07);
    border: 1px solid rgba(52,152,219,0.2);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.stat-num  { font-size: 1.9rem; font-weight: 700; color: #3498db; line-height:1; }
.stat-lbl  { font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 5px; letter-spacing:1px; }

/* ── Section labels ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3498db;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(52,152,219,0.25);
}

/* ── Prediction result ── */
.pred-box {
    background: linear-gradient(135deg, #0d2137, #0a3d62);
    border: 2px solid #3498db;
    border-radius: 20px;
    padding: 40px 36px;
    text-align: center;
    box-shadow: 0 0 50px rgba(52,152,219,0.25);
    animation: glowpulse 3s ease-in-out infinite;
}
@keyframes glowpulse {
    0%,100% { box-shadow: 0 0 40px rgba(52,152,219,0.2); }
    50%      { box-shadow: 0 0 70px rgba(52,152,219,0.45); }
}
.pred-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.pred-value {
    font-size: 3.2rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1;
}
.pred-unit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #3498db;
    margin-top: 10px;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(52,152,219,0.15);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d14 0%, #080f18 100%);
    border-right: 1px solid rgba(52,152,219,0.12);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #1a6fa8, #2980b9, #1a6fa8);
    background-size: 200%;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 14px 0;
    width: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background-position: right center;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(52,152,219,0.45);
}

/* ── Input fields ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(52,152,219,0.3) !important;
    color: #d4e8f5 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.5);
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: rgba(52,152,219,0.2) !important;
    color: #3498db !important;
}

/* ── Progress bar override ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1a6fa8, #3498db) !important;
}

/* ── Scenario badges ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}
.badge-high   { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
.badge-medium { background: rgba(243,156,18,0.15);  color: #f39c12; border: 1px solid rgba(243,156,18,0.4); }
.badge-low    { background: rgba(231,76,60,0.15);   color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.2);
    padding: 16px 0 24px;
}
hr { border-color: rgba(52,152,219,0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("sales_forecasting_Regression_Model.pkl")

model = load_model()

# ─── Helper ───────────────────────────────────────────────────────────────────
def build_input(date, store_id, product_id, promotion_active, holiday_flag,
                base_demand, store_avg, product_avg):
    log_base   = math.log1p(base_demand)
    sqrt_base  = math.sqrt(base_demand)
    month      = date.month
    # seasonality: simple sine curve peaking in Dec
    seasonality_score = round(0.5 + 0.5 * math.sin(2 * math.pi * (month - 3) / 12), 4)
    promo_strength    = round(promotion_active * 1.35, 4)
    holiday_strength  = round(holiday_flag * 1.2, 4)
    seas_weighted     = round(seasonality_score * base_demand, 4)
    seas_demand       = round(seasonality_score * base_demand * 1.05, 4)
    promo_seas        = round(promo_strength * seasonality_score, 4)
    hol_seas          = round(holiday_strength * seasonality_score, 4)
    seas_ratio        = round(seas_demand / (base_demand + 1), 4)
    promo_ratio       = round(promo_strength / (base_demand + 1), 4)
    final_signal      = round(seas_demand * (1 + promo_strength * 0.2 + holiday_strength * 0.1), 4)

    return pd.DataFrame([{
        "date":                str(date),
        "store_id":            store_id,
        "product_id":          product_id,
        "promotion_active":    promotion_active,
        "holiday_flag":        holiday_flag,
        "base_demand":         base_demand,
        "log_base_demand":     log_base,
        "sqrt_base_demand":    sqrt_base,
        "promo_strength":      promo_strength,
        "holiday_strength":    holiday_strength,
        "seasonality_weighted":seas_weighted,
        "seasonality_demand":  seas_demand,
        "promo_seasonality":   promo_seas,
        "holiday_seasonality": hol_seas,
        "store_avg_sales":     store_avg,
        "product_avg_sales":   product_avg,
        "seasonality_ratio":   seas_ratio,
        "promo_ratio":         promo_ratio,
        "final_demand_signal": final_signal,
    }])


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">Sales<span>Pulse</span> AI</p>
  <p class="hero-sub">RETAIL SALES FORECASTING · LINEAR REGRESSION PIPELINE · REAL-TIME PREDICTION</p>
</div>
""", unsafe_allow_html=True)

# ─── Top Stats ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in zip([c1,c2,c3,c4],
    ["19", "Linear Reg", "1", "Multi-Store"],
    ["Input Features", "Algorithm", "Output: Sales Units", "Coverage"]):
    with col:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="stat-num">{num}</div>
          <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Forecast Controls")
    st.markdown("Configure your sales scenario below.")
    st.markdown("---")

    st.markdown("### 📅 Date & Identity")
    sel_date    = st.date_input("Forecast Date", value=datetime.date.today())
    store_id    = st.number_input("Store ID", min_value=1, max_value=500, value=10, step=1)
    product_id  = st.number_input("Product ID", min_value=1, max_value=1000, value=42, step=1)

    st.markdown("### 📦 Demand Inputs")
    base_demand   = st.slider("Base Demand", 10.0, 1000.0, 250.0, step=5.0,
                              help="Expected baseline units before any adjustments")
    store_avg     = st.slider("Store Avg Sales", 50.0, 2000.0, 400.0, step=10.0)
    product_avg   = st.slider("Product Avg Sales", 10.0, 1000.0, 200.0, step=5.0)

    st.markdown("### 🎯 Campaign Flags")
    promotion_active = st.toggle("🏷️ Promotion Active", value=False)
    holiday_flag     = st.toggle("🎉 Holiday Period", value=False)

    st.markdown("---")
    predict_btn = st.button("⚡ FORECAST NOW", use_container_width=True)

    # Scenario presets
    st.markdown("---")
    st.markdown("#### 🎲 Quick Scenarios")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔥 Peak Sale", use_container_width=True):
            st.session_state["preset"] = "peak"
    with col_b:
        if st.button("📉 Off Season", use_container_width=True):
            st.session_state["preset"] = "off"


# ─── Apply presets ────────────────────────────────────────────────────────────
if "preset" in st.session_state:
    if st.session_state["preset"] == "peak":
        base_demand = 800.0; store_avg = 1500.0; product_avg = 700.0
        promotion_active = True; holiday_flag = True
    elif st.session_state["preset"] == "off":
        base_demand = 80.0; store_avg = 150.0; product_avg = 60.0
        promotion_active = False; holiday_flag = False


# ─── Main Layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<div class="sec-label">Input Summary</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏪 Store & Product", "📊 Demand Signals", "🗓️ Derived Features"])

    with tab1:
        df_id = pd.DataFrame({
            "Parameter": ["Forecast Date", "Store ID", "Product ID",
                          "Promotion Active", "Holiday Flag"],
            "Value": [str(sel_date), store_id, product_id,
                      "✅ Yes" if promotion_active else "❌ No",
                      "✅ Yes" if holiday_flag else "❌ No"]
        })
        st.dataframe(df_id, use_container_width=True, hide_index=True)

    with tab2:
        df_demand = pd.DataFrame({
            "Feature": ["Base Demand", "Store Avg Sales", "Product Avg Sales"],
            "Value": [f"{base_demand:.1f}", f"{store_avg:.1f}", f"{product_avg:.1f}"],
            "Range Indicator": [
                int((base_demand-10)/(1000-10)*100),
                int((store_avg-50)/(2000-50)*100),
                int((product_avg-10)/(1000-10)*100),
            ]
        })
        for _, row in df_demand.iterrows():
            st.markdown(f"""
            <div class="card" style="padding:12px 16px; margin-bottom:10px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
                <span style="font-weight:600;">{row['Feature']}</span>
                <span style="font-family:'JetBrains Mono',monospace; color:#3498db; font-size:0.9rem;">{row['Value']}</span>
              </div>
              <div style="background:rgba(255,255,255,0.07); border-radius:5px; height:6px;">
                <div style="width:{row['Range Indicator']}%; background:linear-gradient(90deg,#1a6fa8,#3498db); height:6px; border-radius:5px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with tab3:
        month = sel_date.month
        seas = round(0.5 + 0.5 * math.sin(2 * math.pi * (month - 3) / 12), 4)
        df_derived = pd.DataFrame({
            "Derived Feature": [
                "log_base_demand", "sqrt_base_demand", "promo_strength",
                "holiday_strength", "seasonality_weighted", "final_demand_signal"
            ],
            "Computed Value": [
                f"{math.log1p(base_demand):.4f}",
                f"{math.sqrt(base_demand):.4f}",
                f"{(promotion_active * 1.35):.4f}",
                f"{(holiday_flag * 1.2):.4f}",
                f"{(seas * base_demand):.4f}",
                f"{(seas * base_demand * (1 + (promotion_active*1.35)*0.2 + (holiday_flag*1.2)*0.1)):.4f}",
            ]
        })
        st.dataframe(df_derived, use_container_width=True, hide_index=True)

    # Multi-date forecast table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">7-Day Rolling Forecast</div>', unsafe_allow_html=True)

    rows = []
    for i in range(7):
        d = sel_date + datetime.timedelta(days=i)
        df_in = build_input(d, store_id, product_id,
                            int(promotion_active), int(holiday_flag),
                            base_demand, store_avg, product_avg)
        pred = max(0, model.predict(df_in)[0])
        trend = "🔺" if pred > base_demand else "🔻"
        rows.append({"Date": str(d), "Day": d.strftime("%A"),
                     "Forecast (units)": f"{pred:.1f}", "vs Base": trend})

    df_7day = pd.DataFrame(rows)
    st.dataframe(df_7day, use_container_width=True, hide_index=True)


with right:
    st.markdown('<div class="sec-label">Live Prediction</div>', unsafe_allow_html=True)

    if predict_btn or "preset" in st.session_state:
        if "preset" in st.session_state:
            del st.session_state["preset"]

        with st.spinner("Running forecast model..."):
            time.sleep(0.7)

        df_input = build_input(
            sel_date, store_id, product_id,
            int(promotion_active), int(holiday_flag),
            base_demand, store_avg, product_avg
        )
        prediction = max(0, model.predict(df_input)[0])

        # Badge
        if prediction >= base_demand * 1.3:
            badge = '<span class="badge badge-high">HIGH DEMAND</span>'
        elif prediction >= base_demand * 0.85:
            badge = '<span class="badge badge-medium">NORMAL DEMAND</span>'
        else:
            badge = '<span class="badge badge-low">LOW DEMAND</span>'

        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-label">Forecasted Sales Units</div>
            <div class="pred-value">{prediction:,.1f}</div>
            <div class="pred-unit">units on {sel_date.strftime("%d %b %Y")}</div>
            <div style="margin-top:16px;">{badge}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Impact breakdown
        st.markdown('<div class="sec-label">Demand Drivers</div>', unsafe_allow_html=True)
        drivers = {
            "Base Demand":      base_demand,
            "Promotion Boost":  base_demand * 0.35 if promotion_active else 0,
            "Holiday Lift":     base_demand * 0.20 if holiday_flag else 0,
            "Seasonality":      round(0.5 + 0.5 * math.sin(2*math.pi*(sel_date.month-3)/12), 2) * base_demand * 0.1,
        }
        total = sum(drivers.values()) or 1
        for driver, val in drivers.items():
            pct = int(val / total * 100)
            color = "#3498db" if driver == "Base Demand" else \
                    "#2ecc71" if val > 0 else "#555"
            st.markdown(f"""
            <div class="card" style="padding:12px 16px; margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="font-weight:600; font-size:0.95rem;">{driver}</span>
                <span style="font-family:'JetBrains Mono',monospace; color:{color}; font-size:0.88rem;">+{val:.1f}</span>
              </div>
              <div style="background:rgba(255,255,255,0.07); border-radius:5px; height:5px;">
                <div style="width:{pct}%; background:{color}; height:5px; border-radius:5px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Confidence range
        st.markdown('<div class="sec-label">Confidence Range</div>', unsafe_allow_html=True)
        margin = prediction * 0.12
        lo, hi = max(0, prediction - margin), prediction + margin
        st.markdown(f"""
        <div class="card" style="display:flex; justify-content:space-around; text-align:center; padding:18px;">
          <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#e74c3c;">{lo:,.1f}</div>
            <div style="font-size:0.72rem; opacity:0.5; margin-top:4px;">LOWER BOUND</div>
          </div>
          <div style="width:1px; background:rgba(52,152,219,0.2);"></div>
          <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; color:#3498db; font-weight:700;">{prediction:,.1f}</div>
            <div style="font-size:0.72rem; opacity:0.5; margin-top:4px;">FORECAST</div>
          </div>
          <div style="width:1px; background:rgba(52,152,219,0.2);"></div>
          <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#2ecc71;">{hi:,.1f}</div>
            <div style="font-size:0.72rem; opacity:0.5; margin-top:4px;">UPPER BOUND</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Scenario comparison
        st.markdown('<div class="sec-label">Scenario Comparison</div>', unsafe_allow_html=True)
        scenarios = []
        for promo, hol, label in [(0,0,"No Promo · No Holiday"),
                                   (1,0,"Promo Only"),
                                   (0,1,"Holiday Only"),
                                   (1,1,"Promo + Holiday")]:
            df_s = build_input(sel_date, store_id, product_id,
                               promo, hol, base_demand, store_avg, product_avg)
            p = max(0, model.predict(df_s)[0])
            scenarios.append({"Scenario": label, "Forecast": round(p, 1),
                               "vs Base": f"{((p - base_demand)/base_demand*100):+.1f}%"})
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:70px 30px;">
            <div style="font-size:4rem; margin-bottom:18px;">📊</div>
            <div style="font-size:1.3rem; font-weight:700; color:#3498db; margin-bottom:8px;">
                Ready to Forecast
            </div>
            <div style="opacity:0.55; font-size:0.9rem; line-height:1.7;">
                Configure your store, product &amp; demand<br>
                parameters in the sidebar, then hit<br>
                <b>⚡ FORECAST NOW</b> to generate predictions.
            </div>
        </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
  SalesPulse AI · Linear Regression Pipeline · Retail Sales Forecasting · For demonstration purposes
</div>""", unsafe_allow_html=True)
