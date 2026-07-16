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

# Calculate quick metrics based on the user's slicer filters
total_sales = int(filtered_df['Sales'].sum())
avg_daily_sales = int(filtered_df['Sales'].mean())
forecast_rows = filtered_df[filtered_df['Type'] == 'Forecast']

# Determine the average prediction uncertainty spread
if not forecast_rows.empty:
    avg_spread = int((forecast_rows['Upper_Bound'] - forecast_rows['Lower_Bound']).mean())
else:
    avg_spread = 0

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Model Overview")
st.sidebar.info(
    """
    **Hybrid ML Architecture:**
    - **Linear Regression** captures the infinite macro growth trend lines.
    - **Random Forest Regressor** predicts micro seasonal wave variations.
    - **Dynamic Intervals** scale outwards over time to accurately simulate future market uncertainty.
    """
    
# Display metrics side-by-side in 3 clean columns
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Units (Selected Range)", value=f"{total_sales:,}")
col2.metric(label="Avg Daily Volume", value=f"{avg_daily_sales:,}")
col3.metric(label="Avg Forecast Variance (±)", value=f"{avg_spread} units")

st.markdown("---") # Visual divider line

# Display chart inside the web app page layout
st.pyplot(fig)

# Collapsible data table view
with st.expander("🔍 View Raw Filtered Data Table"):
    st.dataframe(
        filtered_df[['Date', 'Product', 'Sales', 'Lower_Bound', 'Upper_Bound', 'Type']],
        use_container_width=True
    )
