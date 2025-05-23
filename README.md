# Stock Market Analysis

A machine-learning pipeline to analyze and forecast Apple Inc. (AAPL) daily closing prices using lagged features and a Random Forest regressor.


## Quick Start

1. **Clone the repo**  
   ``bash
   git clone https://github.com/<your-username>/aapl-stock-predictor.git
   cd aapl-stock-predictor
``

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   Ensure the file `data/AAPL.csv` exists (contains columns: `Date, Open, High, Low, Close, Adj Close, Volume`).

4. **Run the notebook**
   Open `notebooks/StockMarketAnalysis.ipynb` and run all cells in order.

   * **Section 1 (EDA):** summary stats, price/volume plots, returns, volatility, moving averages, decomposition, ACF/PACF
   * **Section 2 (ML):** create lag & MA features, train/test split (train through 2021, test since 2022), train Random Forest, evaluate MAE/RMSE, plot actual vs. predicted
   * **Section 3 (Artifacts):** saves model (`.joblib`), predictions (`.csv`), and evaluation plots (`.png`)

5. **Inspect outputs**

   * Trained model: `models/aapl_rf_model.joblib`
   * Predictions:  `outputs/aapl_rf_predictions.csv`
   * Backtest plot: `outputs/aapl_backtest.png`

---

## Usage Example

Load the trained model and forecast next-day close prices in a Python script:

```python
import joblib
import pandas as pd

model = joblib.load("models/aapl_rf_model.joblib")

df = pd.read_csv("data/AAPL.csv", parse_dates=["Date"]).set_index("Date")
# prepare lag and MA features exactly as in notebook...
X_new = df[["lag_1","lag_2","lag_3","lag_4","lag_5","ma_5","ma_20"]].dropna()

predictions = model.predict(X_new)
print(predictions[:5])
```

## License

This project is released under the **MIT License**.
Data source: “Big Tech Stock Prices” Kaggle dataset.

