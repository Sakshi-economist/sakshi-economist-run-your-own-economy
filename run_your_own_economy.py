# run_your_own_economy.py
#
# Run Your Own Economy - Streamlit app (auto-load dataset; no upload)
# Save this file in the same folder as your Excel dataset and run with:
#   streamlit run run_your_own_economy.py
#
# Dataset path used (your provided path):
# "C:\Users\HP\Downloads\Final_Macro_Dataset_with_Deficit_and_Trade.xlsx"
#
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(page_title="Run Your Own Economy", layout="wide")
DEFAULT_DATA_PATH = "Final_Macro_Dataset_with_Deficit_and_Trade.xlsx"

st.title("Run Your Own Economy — Policy Simulator & Forecasting")
st.write("This app automatically loads the bundled dataset; no upload required.")

# ---------------------------
# Utility helpers
# ---------------------------
def normalize_colname(s: str) -> str:
    s = str(s).lower()
    for ch in [" ", "_", "%", "(", ")", "-", ".", "/", "\\"]:
        s = s.replace(ch, "")
    return s

def map_expected_to_actual(expected_list, actual_columns):
    actual_map = {normalize_colname(c): c for c in actual_columns}
    mapping = {}
    for exp in expected_list:
        key = normalize_colname(exp)
        if key in actual_map:
            mapping[exp] = actual_map[key]
        else:
            # fuzzy partial match: pick first actual column that contains the expected key or vice-versa
            found = None
            for act_norm, act_orig in actual_map.items():
                if key in act_norm or act_norm in key:
                    found = act_orig
                    break
            mapping[exp] = found
    return mapping

# ---------------------------
# Load dataset (no uploader)
# ---------------------------
if not os.path.exists(DEFAULT_DATA_PATH):
    st.error(f"Dataset not found at the expected path:\n{DEFAULT_DATA_PATH}\n\nPlease place the Excel file there or update the path in the script.")
    st.stop()

try:
    df = pd.read_excel(DEFAULT_DATA_PATH, engine="openpyxl")
except Exception as e:
    st.error(f"Failed to read Excel file at {DEFAULT_DATA_PATH}: {e}")
    st.stop()

# show columns for debugging
st.markdown("**Dataset columns detected (preview)**")
st.write(list(df.columns))

# Ensure Year column exists
if "Year" not in df.columns:
    year_col = None
    for c in df.columns:
        if "year" in str(c).lower():
            year_col = c
            break
    if year_col:
        df = df.rename(columns={year_col: "Year"})
    else:
        st.error("No Year column found in dataset. Please ensure your dataset contains a Year column.")
        st.stop()

# Clean Year
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"]).copy()
df["Year"] = df["Year"].astype(int)
df = df.sort_values("Year").reset_index(drop=True)

# ---------------------------
# Expected variables to use for VAR (change if you like)
# ---------------------------
expected_vars = [
    "GDP Growth (%)",
    "Inflation_CPI_%",
    "Policy_Rate_%",
    "Govt_Expense_%_GDP",
    "M3_Growth_%",
    "Industry_Growth_%",
    "Exchange_Rate_INR_USD",
    "Unemployment_%",
    "Trade_Balance_USD"
]

mapping = map_expected_to_actual(expected_vars, df.columns)
st.markdown("**Column mapping (expected -> detected actual)**")
st.write(mapping)

# build list of actual columns to use (in the expected order)
var_columns = [mapping[k] for k in expected_vars if mapping[k] is not None]

if len(var_columns) < 2:
    st.error("Not enough variables found for VAR. Need at least 2 of the expected variables.")
    st.stop()

# Prepare VAR dataframe (drop rows with NA in any chosen var column)
var_df = df[["Year"] + var_columns].dropna().set_index("Year")

st.markdown("**Variables used for VAR**")
st.write(var_columns)

# ---------------------------
# App layout: tabs
# ---------------------------
tab1, tab2 = st.tabs(["🏛 Policy Simulator", "📈 Forecasting"])

# ---------------------------
# TAB 1: Policy Simulator (VAR)
# ---------------------------
with tab1:
    st.header("Policy Simulator (VAR-based)")

    st.write("Adjust policy levers. The app computes VAR impulse responses and superposes shocks.")

    col1, col2 = st.columns(2)
    with col1:
        repo_shock = st.slider("Repo rate shock (percentage points)", -5.0, 5.0, 0.0, 0.1)
        m3_shock = st.slider("M3 growth shock (pp)", -10.0, 10.0, 0.0, 0.5)
        fx_shock = st.slider("Exchange rate shock (%)", -30.0, 30.0, 0.0, 0.5)
    with col2:
        govt_shock = st.slider("Government spending shock (pp of GDP)", -10.0, 20.0, 0.0, 0.1)
        trade_shock_pct = st.slider("Trade balance shock (%)", -100.0, 100.0, 0.0, 1.0)
        tax_shock = st.slider("Tax rate shock (pp)", -10.0, 10.0, 0.0, 0.1)

    st.markdown("---")
    maxlags = st.selectbox("VAR maxlags (AIC used to pick best)", [1, 2, 3, 4], index=1)
    horizon = st.number_input("IRF horizon (steps)", min_value=1, max_value=36, value=10, step=1)

    if st.button("Run VAR Simulation"):
        try:
            model = VAR(var_df)
            results = model.fit(maxlags, ic="aic")
        except Exception as e:
            st.error(f"VAR model fit failed: {e}")
            st.stop()

        try:
            irf = results.irf(horizon)
            # irf.irfs shape: (steps, k, k) where steps = horizon+1, k = number of vars
            steps = irf.irfs.shape[0]
            k = irf.irfs.shape[1]
        except Exception as e:
            st.error(f"IRF calculation failed: {e}")
            st.stop()

        # map variable name -> column index (in var_columns)
        col_to_idx = {col: i for i, col in enumerate(var_columns)}
        shock_map = {}

        # Map slider shocks to variable indices if variable exists
        if mapping.get("Policy_Rate_%") in col_to_idx:
            shock_map[col_to_idx[mapping["Policy_Rate_%"]]] = repo_shock
        if mapping.get("M3_Growth_%") in col_to_idx:
            idx = col_to_idx[mapping["M3_Growth_%"]]
            shock_map[idx] = shock_map.get(idx, 0.0) + m3_shock
        if mapping.get("Exchange_Rate_INR_USD") in col_to_idx:
            idx = col_to_idx[mapping["Exchange_Rate_INR_USD"]]
            shock_map[idx] = shock_map.get(idx, 0.0) + fx_shock
        if mapping.get("Govt_Expense_%_GDP") in col_to_idx:
            idx = col_to_idx[mapping["Govt_Expense_%_GDP"]]
            shock_map[idx] = shock_map.get(idx, 0.0) + govt_shock
        if mapping.get("Trade_Balance_USD") in col_to_idx:
            idx = col_to_idx[mapping["Trade_Balance_USD"]]
            try:
                last_trade = float(df[mapping["Trade_Balance_USD"]].dropna().iloc[-1])
                trade_usd_change = last_trade * (trade_shock_pct / 100.0)
            except Exception:
                trade_usd_change = trade_shock_pct
            shock_map[idx] = shock_map.get(idx, 0.0) + trade_usd_change
        # Tax shock mapped approximately to GDP Growth
        if mapping.get("GDP Growth (%)") in col_to_idx:
            idx = col_to_idx[mapping["GDP Growth (%)"]]
            tax_effect_on_gdp = -0.2 * tax_shock
            shock_map[idx] = shock_map.get(idx, 0.0) + tax_effect_on_gdp

        if not shock_map:
            st.warning("No matching VAR variable was found for your selected shocks. Check the column mapping above.")
            st.stop()

        # Compute combined response via linear superposition
        combined = np.zeros((steps, k), dtype=float)
        for shock_idx, shock_mag in shock_map.items():
            try:
                # irf.irfs[:, :, shock_idx] -> shape (steps, k)
                contribution = irf.irfs[:, :, shock_idx] * shock_mag
                combined += contribution
            except Exception as e:
                st.error(f"Error computing contribution for shock index {shock_idx}: {e}")

        # Build response DF and ensure Year_Ahead length matches rows exactly
        response_df = pd.DataFrame(combined, columns=var_columns)
        response_df["Period"] = np.arange(0, response_df.shape[0])  # exact length match

        st.subheader("Combined Impulse Response (combined shocks)")
        st.write("Rows = periods ahead (0 = contemporaneous).")
        st.dataframe(response_df)

        # Plot a few key variables if present
        to_plot = [v for v in ["GDP Growth (%)", "Inflation_CPI_%", "Unemployment_%", "Industry_Growth_%"] if v in var_columns]
        for v in to_plot:
            fig = px.line(response_df, x="Period", y=v, title=f"{v} response to combined shock")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 2: Forecasting
# ---------------------------
with tab2:
    st.header("Forecasting (Hybrid ML)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Year" in numeric_cols:
        numeric_cols.remove("Year")

    if not numeric_cols:
        st.error("No numeric columns available for forecasting.")
    else:
        target = st.selectbox("Choose forecast target", numeric_cols)
        df_clean = df.dropna(subset=[target, "Year"])
        if len(df_clean) < 5:
            st.warning("Not enough data to train reliable models (need >= 5 observations).")
        else:
            X = df_clean["Year"].values.reshape(-1, 1)
            y = df_clean[target].values
            split_idx = max(1, int(0.8 * len(X)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Linear Regression
            lr = LinearRegression(); lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_pred); lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

            # Random Forest
            rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred); rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

            st.subheader("Model performance on holdout")
            perf_df = pd.DataFrame({
                "Model": ["Linear Regression", "Random Forest"],
                "R2": [lr_r2, rf_r2],
                "RMSE": [lr_rmse, rf_rmse]
            })
            st.table(perf_df)

            n_future = st.slider("Years to forecast ahead", 1, 30, 5)
            last_year = int(df_clean["Year"].iloc[-1])
            future_years = np.arange(last_year + 1, last_year + 1 + n_future).reshape(-1, 1)

            lr_fore = lr.predict(future_years)
            rf_fore = rf.predict(future_years)

            forecast_df = pd.DataFrame({
                "Year": future_years.flatten(),
                f"LR_{target}": lr_fore,
                f"RF_{target}": rf_fore
            })
            st.subheader("Forecast table")
            st.dataframe(forecast_df)

            fig = px.line(forecast_df, x="Year", y=[f"LR_{target}", f"RF_{target}"], title=f"Forecasts for {target}")
            st.plotly_chart(fig, use_container_width=True)

            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download forecast CSV", data=csv, file_name=f"forecast_{target}.csv", mime="text/csv")

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.caption("Notes: This is an educational policy simulator. VAR IRFs are linear impulse-response approximations. For production-grade simulation add diagnostics, exogenous controls, and richer data transformations.")

