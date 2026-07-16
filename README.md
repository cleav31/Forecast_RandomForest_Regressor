# Time Series Forecasting with Random Forest Regressor

This repository contains a machine learning project designed to forecast continuous sequential data using a **Random Forest Regressor**. Despite extensive feature engineering, the model produced a flat and generic baseline forecast. This repository serves as a documented benchmark, highlighting the structural limitations of non-linear ensemble tree methods when applied to certain time series structures.

---

## 📌 Project Overview
The objective of this project was to leverage a tabular machine learning approach for sequential forecasting. However, the evaluation phase revealed that the model struggled to capture dynamic peaks and valleys, reverting instead to a highly smoothed, mean-focused prediction line.

## ⚙️ Data Generation & Tech Stack
To test the forecasting pipeline safely and effectively, all inputs used in this project were simulated:
* **Synthetic Data Generation**: The underlying time series dataset was programmatically generated using **Python** to replicate realistic seasonal cycles, trends, and random noise.
* **Core Libraries**: Built using the standard Python data science ecosystem, including **NumPy** and **Pandas** for data manipulation.
* **Machine Learning**: Frameworks like **Scikit-Learn** were utilized for dataset splitting, feature scaling, model training, and evaluation.

## 🛠️ Methodology & Feature Engineering
To help a tree-based model recognize temporal patterns, the synthetic sequential data was transformed using several domain-specific feature engineering techniques:

* **Calendar Features**: Extracted year, month, day, and day of the week to capture cyclical seasonality.
* **Lag Features**: Integrated historical values (e.g., $t-1$, $t-7$) to provide immediate baseline context.
* **Rolling Statistics**: Computed rolling means and standard deviations over varying windows to smooth noise and indicate momentum.

---

## ⚠️ Performance Analysis & Model Limitations
The resulting forecast flattened significantly, failing to capture the true volatility of the validation dataset. The flat prediction curve is a direct result of specific architectural limitations of tree-based ensembles:

### 1. Inability to Extrapolate
Random Forest models make predictions by averaging the target values of training samples within specific leaf nodes. Consequently, they **cannot** predict values outside the minimum and maximum boundaries found in the training data. If the test data contains a broader upward or downward trend, the model caps its predictions, resulting in a flat line.

### 2. Mean-Reversion Bias
When decision trees encounter high noise or weak feature correlations, they default to splits that minimize variance. This causes the ensemble to average out extreme fluctuations. The final output heavily favors the global or localized mean, washing out the distinct variance required for an accurate forecast.

### 3. Lack of Sequential Ordering
Random Forest treats each row of data as an independent observation. It fundamentally lacks an internal mechanism to understand time-based sequence or memory, making it highly dependent on the quality of engineered lag features.

---

## 🚀 Next Steps & Alternative Approaches
To improve upon this baseline, future iterations of this project could pivot toward architectures natively designed for sequential data:

* **Statistical Baselines**: Implementing **ARIMA** or **SARIMA** models to handle underlying trends and seasonality directly.
* **Gradient Boosting with Linear Base Learners**: Utilizing **XGBoost** or **LightGBM** with linear trees to allow for better boundary extrapolation.
* **Deep Learning**: Deploying **Long Short-Term Memory (LSTM)** networks or **Gated Recurrent Units (GRUs)** to capture deep sequential dependencies.

---
