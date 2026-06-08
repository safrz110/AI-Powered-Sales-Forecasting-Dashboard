#  AI Sales Forecasting Dashboard

> A production-ready machine learning application that predicts retail sales using a scikit-learn regression pipeline, served through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

This project delivers an end-to-end sales forecasting solution — from feature engineering and model training to an interactive business dashboard. Users can input store-level and product-level parameters and receive an instant sales prediction, complete with KPI cards, a projected trend chart, and a downloadable CSV report.

The model is a **scikit-learn Pipeline** combining:
- `StandardScaler` for numerical feature normalization
- `OneHotEncoder` for date-based categorical encoding
- `LinearRegression` as the estimator, trained on log-transformed sales targets for better handling of skewed distributions

---

##  Features

- **Real-time Sales Prediction** — Adjust inputs and get instant ML-powered forecasts
- **19-Feature Engineering Pipeline** — Engineered features including log/sqrt transforms, promo strength, seasonality weighting, and interaction terms
- **Interactive KPI Dashboard** — Visual breakdown of base demand, promotion impact, and seasonality contribution
- **Projected Sales Trend Chart** — 5-month forward-looking Plotly visualization
- **One-click CSV Export** — Download prediction reports for offline analysis
- **Clean, Responsive UI** — Custom CSS-styled Streamlit interface with wide layout

---

##  Project Structure

```
sales-forecasting/
│
├── app.py                                  # Streamlit dashboard application
├── sales_forecasting_Regression_Model.pkl  # Trained sklearn Pipeline (scaler + encoder + LR)
├── columns.pkl                             # Feature column order used during training
├── requirements.txt                        # Pinned dependency list
└── README.md
```

---

##  How It Works

### Feature Engineering

Raw user inputs are transformed into 19 model features before prediction:

| Feature | Description |
|---|---|
| `base_demand` | Core demand signal (user input) |
| `log_base_demand` | `log1p` transform for skew reduction |
| `sqrt_base_demand` | Square root transform for variance stabilization |
| `promo_strength` | `promotion_active × base_demand` interaction |
| `holiday_strength` | `holiday_flag × base_demand` interaction |
| `seasonality_weighted` | Weighted blend of weekly (×0.7) and yearly (×1.3) seasonality |
| `seasonality_demand` | `seasonality_weighted × base_demand` |
| `promo_seasonality` | `promotion_active × seasonality_weighted` |
| `holiday_seasonality` | `holiday_flag × seasonality_weighted` |
| `seasonality_ratio` | `seasonality_demand / (base_demand + 1)` |
| `promo_ratio` | `promo_strength / (base_demand + 1)` |
| `final_demand_signal` | Sum of all demand drivers |
| `store_avg_sales` | Historical average sales for the store |
| `product_avg_sales` | Historical average sales for the product |
| `date` | One-hot encoded date for temporal patterns |

### Prediction Flow

```
User Inputs → Feature Engineering → sklearn Pipeline
                                        ├── StandardScaler (numerical)
                                        ├── OneHotEncoder (date)
                                        └── LinearRegression → log(sales)
                                                                    ↓
                                                             expm1(pred) → Actual Sales
```

The model predicts **log-transformed sales**, and `np.expm1()` is applied to convert back to the original scale.

---

##  Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

##  Dependencies

Key libraries used (see `requirements.txt` for pinned versions):

| Library | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `scikit-learn` | ML pipeline (scaler, encoder, regression) |
| `pandas` | Data manipulation and feature construction |
| `numpy` | Numerical transforms (log, sqrt, expm1) |
| `plotly` | Interactive trend charts |
| `joblib` | Model serialization and loading |

---

##  Dashboard Walkthrough

1. **Set Parameters** — Use the sidebar to enter store ID, product ID, demand values, promotion/holiday flags, and seasonality components
2. **Click Predict** — The model runs instantly and displays the forecasted sales figure
3. **Analyze KPIs** — Review how base demand, promotions, and seasonality each contribute to the final prediction
4. **View Trend** — See a 5-month projected sales trajectory based on the predicted value
5. **Export** — Download the prediction as a CSV for reporting or further analysis

---

##  Model Details

| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Preprocessing | StandardScaler + OneHotEncoder |
| Target Variable | `log1p(sales)` |
| Output Transform | `expm1(prediction)` |
| Framework | scikit-learn Pipeline |
| Serialization | joblib `.pkl` |

---

##  Roadmap

- [ ] Add multi-step forecasting (7-day, 30-day horizon)
- [ ] Integrate XGBoost / LightGBM for improved accuracy
- [ ] Add model performance metrics display (R², MAE, RMSE)
- [ ] Support bulk CSV upload for batch predictions
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces

---

##  Contributing

Contributions are welcome! Please open an issue to discuss what you'd like to change, or submit a pull request directly.

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Author

**Sarfaraz Ali**


*Built with Python, scikit-learn, and Streamlit.*
