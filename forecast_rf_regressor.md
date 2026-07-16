# Forecast: Random Forest Regressor

## Import Packages


```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Imports for Visuals
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

# to remove unnecessary warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

```

## Generate Base Historical Actuals (2020 to Mid-2026)


```python
np.random.seed(41)
products = ['Product A', 'Product B', 'Product C']
start_date = datetime(2020,1,1)
forecast_start = datetime(2026,7,1)
end_date = datetime(2026,12,31)
```

## Generate Past Dates


```python
history_dates = pd.date_range(start=start_date, 
                              end=forecast_start - timedelta(days=1),
                              freq='D')
all_data = []
for product in products:
    base = np.random.randint(150,300)
    for i, date in enumerate(history_dates):
        # Generate raw historical sales with trends and noise
        trend = base + (i*0.25)
        weekly = 15 * np.sin(2 * np.pi * date.weekday() / 7)
        yearly = 40 * np.sin(2 * np.pi * date.dayofyear / 365)
        noise = np.random.normal(0,10)
        sales = max(0,int(trend + weekly + yearly + noise))

        all_data.append({'Date':date,
                         'Product': product,
                         'Sales': sales,
                         'Type': 'Actual'})
df = pd.DataFrame(all_data)
                            
```

## Feature Engineering Function


```python
def create_features(data):
    df_feat = data.copy().sort_values(['Product', 'Date'])

# Time Features
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['DayofWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['DayofYear'] = df_feat['Date'].dt.dayofyear
    df_feat['Days_Since_Start'] = (df_feat['Date'] - start_date).dt.days

# Lag & Rolling features (based on History)
    df_feat['Lag_7'] = df_feat.groupby('Product')['Sales'].shift(7)
    df_feat['Rolling_Mean_7'] = df_feat.groupby('Product')['Sales'].transform(
        lambda x: x.shift(1).rolling(7).mean())

# Backfill the first few rows that get Nan from shifting
    df_feat = df_feat.bfill()
    return df_feat
```


```python
df_features = create_features(df)
```

## Train Machine Learning Model


```python
X_train = df_features[['Month', 'DayofWeek', 'DayofYear', 'Days_Since_Start',
                       'Lag_7', 'Rolling_Mean_7']]
y_train = df_features['Sales']
```


```python
model = RandomForestRegressor(n_estimators=100, random_state=41)
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(random_state=41)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=41)</pre></div></div></div></div></div>



### Calculate model error variance on training date to generate future bounds


```python
train_pred = model.predict(X_train)
residual_std = np.std(y_train - train_pred)
```

### Set actual hostorical bounds to match sales perfectly (no uncertainy in the past)


```python
df['Lower_Bound'] = df['Sales']
df['Upper_Bound'] = df['Sales']
```

## Iterative Recursive Future Forecasting (Mid-2026 to End-2026)


```python
future_dates = pd.date_range(start=forecast_start, end=end_date, freq='D')


features_list = ['Month', 'DayofWeek', 'DayofYear', 'Days_Since_Start',
                 'Lag_7', 'Rolling_Mean_7']
for date in future_dates:
    for product in products:
        # Filter dataframe up to the day before current loop date to calculate lag
        sub_df = df[df['Product'] == product].copy()

# Build features for the single forecast day
        month = date.month
        dayofweek = date.dayofweek
        dayofyear = date.dayofyear
        days_since_start = (date - start_date).days
        
# Extract features dynamically from recent past rows
        lag_7 = float(sub_df.iloc[-7]['Sales'])
        rolling_7 = float(sub_df.iloc[-7:]['Sales'].mean())

# Predict midpoint using ML Model, format explicitly as a 2D array to bypass bug
        X_pred = np.array([[month, dayofweek, dayofyear, 
                            days_since_start, lag_7, rolling_7]])
                            
        pred_sales = int(model.predict(X_pred)[0])

# Calculate expanding uncertainty bounds over time
        days_out = (date - forecast_start).days
        dynamic_spread = residual_std * (1+(days_out * 0.5)) # Spreads out further into the future

        lower_b = max(0, int(pred_sales - dynamic_spread))
        upper_b = int(pred_sales + dynamic_spread)
        
# Append rows back into main dataframe so next loop can use for lag features
        new_row = pd.DataFrame([{
            'Date': date, 'Product': product, 'Sales': pred_sales,
            'Type': 'Forecast', 'Lower_Bound': lower_b, 'Upper_Bound': upper_b
        }])
        df = pd.concat([df, new_row], ignore_index=True)
```

## Save Final Machine Learning Output


```python
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.to_csv('ml_sales_forecast.csv', index=False)
print("ML Forecast completed and saved as 'ml_sales_forecast.csv'!")
```

    ML Forecast completed and saved as 'ml_sales_forecast.csv'!


# Create Visualization


```python
df_ml = pd.read_csv('ml_sales_forecast.csv')
df_ml['Date'] = pd.to_datetime(df_ml['Date'])

def plot_ml_sales(product, date_range):
    filtered_df = df_ml[
        (df_ml['Product'] == product) &
        (df_ml['Date'] >= pd.to_datetime(date_range[0])) &
        (df_ml['Date'] <= pd.to_datetime(date_range[1]))
    ].sort_values('Date')

    plt.figure(figsize=(12,6))

# Plot real history vs predicted forecast line
    actuals = filtered_df[filtered_df['Type'] == 'Actual']
    forecasts = filtered_df[filtered_df['Type'] == 'Forecast']

    plt.plot(actuals['Date'], actuals['Sales'], label='Actual Sales',
        color='black', linewidth=1.5)
    plt.plot(forecasts['Date'], forecasts['Sales'], label='ML Predicted Midpoint',
        color='darkorange', linestyle='-')
# Fill dynamic ML intervals
    plt.fill_between(
        filtered_df['Date'],
        filtered_df['Lower_Bound'],
        filtered_df['Upper_Bound'],
        color='orange',
        alpha=0.15,
        label='ML Uncertainty Range'
    )
    plt.axvline(x=pd.to_datetime('2026-07-01'), color='red',
        linestyle='-', label='Forecast Boundary')
    plt.title(f'Machine Learning Sales Forecast Dashboard - {product}')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

product_slicer = widgets.Dropdown(options=df_ml['Product'].unique(),
                                  description='Product:')
date_slicer = widgets.SelectionRangeSlider(
    options=[(d.strftime('%Y-%m-%d'), d) for d in pd.date_range(df_ml['Date'].min(),
        df_ml['Date'].max(), freq='MS')],
    index = (0,23),
    description = 'Dates',
    layout = {'width':'500px'}
)
interact(plot_ml_sales, product = product_slicer, date_range = date_slicer);
                                                                
                                                                
```


    interactive(children=(Dropdown(description='Product:', options=('Product A', 'Product B', 'Product C'), value=…



```python

```
