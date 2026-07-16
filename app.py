import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set up page styling
st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Machine Learning Sales Forecast Dashboard")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('ml_sales_forecast.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# 2. CREATE THE SLICERS (Sidebar controls)
product_list = df['Product'].unique()
selected_product = st.sidebar.selectbox("Select Product:", product_list)

min_date = df['Date'].min().to_pydatetime()
max_date = df['Date'].max().to_pydatetime()
selected_dates = st.sidebar.slider("Select Date Range:", min_date, max_date, (min_date, max_date))

# 3. Filter Data
filtered_df = df[
    (df['Product'] == selected_product) & 
    (df['Date'] >= selected_dates[0]) & 
    (df['Date'] <= selected_dates[1])
].sort_values('Date')

# 4. Create the Dashboard Visual Canvas
fig, ax = plt.subplots(figsize=(12, 5))

actuals = filtered_df[filtered_df['Type'] == 'Actual']
forecasts = filtered_df[filtered_df['Type'] == 'Forecast']

if not actuals.empty:
    ax.plot(actuals['Date'], actuals['Sales'], label='Actual Sales', color='black')
if not forecasts.empty:
    ax.plot(forecasts['Date'], forecasts['Sales'], label='ML Forecast', color='orange')

ax.fill_between(filtered_df['Date'], filtered_df['Lower_Bound'], filtered_df['Upper_Bound'], color='orange', alpha=0.15)
ax.axvline(pd.to_datetime('2026-07-01'), color='red', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.2)

# Display chart inside the web app page layout
st.pyplot(fig)
