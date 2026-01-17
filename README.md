# 📈 Gold Price Prediction using LSTM

## 📌 Project Overview
This project applies Deep Learning techniques to forecast the daily log returns of **Gold Comex Futures (GC=F)**. By utilizing a **Long Short-Term Memory (LSTM)** neural network, the model analyzes historical price data to predict future market movements.

The project emphasizes statistical rigor, ensuring data stationarity before model training, and uses a robust evaluation framework comparing the model against a zero-return baseline.

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Data Source:** Yahoo Finance (`yfinance`)
* **Data Analysis:** Pandas, NumPy, Statsmodels
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** TensorFlow/Keras (LSTM), Scikit-Learn

## 📊 Dataset & Preprocessing
* **Source:** Historical daily data for Gold Futures (`GC=F`) fetched via `yfinance`.
* **Stationarity Check:**
    * **Augmented Dickey-Fuller (ADF) Test** was conducted to ensure the time series is stationary.
    * *Raw Prices:* p-value = `1.0` (Non-Stationary)
    * *Log Returns:* p-value = `0.0` (Stationary)
    * **Decision:** The model is trained on **Log Returns** rather than raw prices to ensure stability and convergence.
* **Feature Engineering:**
    * Calculated Logarithmic Returns.
    * Data scaling using `StandardScaler`.
    * Created rolling window sequences of **60 days** for time-series forecasting.

## 🧠 Model Architecture
The model is a Stacked LSTM network designed to capture temporal dependencies in financial data:

1.  **LSTM Layer 1:** 50 Units, `return_sequences=True`
2.  **Dropout:** 20% (to prevent overfitting)
3.  **LSTM Layer 2:** 50 Units, `return_sequences=False`
4.  **Dropout:** 20%
5.  **Dense Output Layer:** 1 Unit (Regression)

**Training Configuration:**
* **Optimizer:** Adam
* **Loss Function:** Huber Loss (Robust to outliers/spikes in financial data)
* **Callbacks:** EarlyStopping (Patience=10) to prevent overfitting.

## 📉 Evaluation
The model is evaluated on the remaining 20% of the dataset (Test Set).

* **Metric 1:** Root Mean Squared Error (RMSE) comparing Model vs. Zero-Return Baseline.
* **Metric 2:** Directional Accuracy (Percentage of time the model correctly predicted the Up/Down movement).

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/gold-price-lstm.git](https://github.com/yourusername/gold-price-lstm.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn yfinance statsmodels scikit-learn tensorflow
    ```
3.  Run the script:
    ```bash
    python project2.py
    ```

 

## 🔮 Future Improvements
* **Multivariate Analysis:** Incorporating correlated assets like the US Dollar Index (DXY) or Treasury Yields.
* **Hyperparameter Tuning:** Using Keras Tuner to optimize LSTM units and learning rates.
* **Price Reconstruction:** Converting log return predictions back to dollar price levels for clearer visualization.

---
*Created by [Your Name]*
