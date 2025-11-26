import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ----------------------------------------
# Configuration
# ----------------------------------------
st.set_page_config(page_title="Run Your Own Economy", layout="wide")

# IMPORTANT: Relative path for Streamlit Cloud
DEFAULT_DATA_PATH = "Final_Macro_Dataset_with_Deficit_and_Trade.xlsx"

st.title("Run Your Own Economy — Policy Simulator & Forecasting")
st.write("This app automatically loads the bundled dataset; no upload required.")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel(DEFAULT_DATA_PATH)
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_data()

if error or df is None:
    st.error(
        f"Dataset not found at the expected path: {DEFAULT_DATA_PATH}\n\n"
        "Make sure the Excel file is uploaded to your GitHub repo (same folder as this script)."
    )
    st.stop()

# ----------------------------------------
# Show Preview
# ----------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------------------
# Variable Selection
# ----------------------------------------
st.subheader("Select Variables for Forecasting")

target = st.selectbox("Select Target Variable (Y)", df.columns)
features = st.multiselect("Select Independent Variables (X)", df.columns)

# ----------------------------------------
# Run Model
# ----------------------------------------
if st.button("Run Model"):
    if not features:
        st.warning("Please select at least one independent variable.")
        st.stop()

    X = df[features]
    y = df[target]

    # sklearn model
    model = LinearRegression()
    model.fit(X, y)

    st.success("Model successfully estimated!")

    # Show coefficients
    st.subheader("Model Coefficients")
    coef_table = pd.DataFrame({
        "Variable": features,
        "Coefficient": model.coef_
    })
    st.dataframe(coef_table)

    # Forecast next period
    last_row = X.iloc[-1:].copy()
    forecast = model.predict(last_row)[0]

    st.subheader("Next Period Forecast")
    st.metric(label=f"Forecasted {target}", value=round(forecast, 4))

    # Plot actual vs fitted
    fitted = model.predict(X)
    fig = px.line(title=f"Actual vs Fitted: {target}")
    fig.add_scatter(x=df.index, y=y, name="Actual")
    fig.add_scatter(x=df.index, y=fitted, name="Fitted")
    st.plotly_chart(fig, use_container_width=True)

