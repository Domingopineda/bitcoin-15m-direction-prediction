# Bitcoin 15-Minute Direction Prediction (BTCUSDT)

## Project Overview

End-to-end machine learning pipeline that predicts whether the **next 15-minute BTCUSDT candle will close higher than the current one**, using public Binance market data and technical indicators.

The project covers the full workflow a data team runs in production: automated data collection with validation, data cleaning, feature engineering, model comparison with a proper time-series evaluation, and a prediction script that scores the latest closed candle.

> This project is for educational and portfolio purposes. It is not financial advice, and model probabilities are estimates, not guarantees.

## How It Works

```
Binance API ──► data_collection ──► data_cleaning ──► feature_engineering
                                                            │
                                                     model_training
                                                            │
                                          best model (models/best_btc_model.pkl)
                                                            │
                                                       prediction
```

1. **`src/data_collection.py`** — Downloads BTCUSDT 15-minute klines from the Binance public API (no API key required), starting 2025-10-11. Updates are incremental: it loads the existing CSV, downloads only new **closed** candles, removes duplicates, and validates gaps, chronological order and OHLCV integrity before saving `data/bitcoin_15m.csv`.
2. **`src/data_cleaning.py`** — Converts timestamps and numeric types, removes duplicates and invalid records, validates price/volume and candle structure (high ≥ open/close ≥ low), and saves `data/bitcoin_15m_cleaned.csv`.
3. **`src/feature_engineering.py`** — Builds **22 technical features** per candle and the target variable: `Target = 1` if the next candle's return is positive. Saves `data/bitcoin_15m_features.csv`. The last row keeps its features with an unknown target, so it can be scored live.
4. **`src/model_training.py`** — Trains and compares models on a **chronological 80/20 split** (no shuffling, so no look-ahead leakage), evaluates them against a dummy baseline, selects the best model by **Brier score** (probability quality, not just accuracy), and saves `data/model_comparison.csv` and `models/best_btc_model.pkl`.
5. **`src/prediction.py`** — Loads the trained model, verifies feature compatibility, takes the latest **fully closed** candle, and prints the probability of an up move with a trading-style signal: `UP` (≥ 55%), `DOWN / NOT UP` (≤ 45%), or `NO CLEAR EDGE` in between. Times are shown in UTC and New York time.

## Features (22)

- **Returns & candle structure:** Returns, change %, range %, candle body, upper/lower wicks
- **Momentum:** RSI (7, 21), MACD, MACD signal, MACD diff, Momentum (3, 6, 12)
- **Trend:** EMA (9, 21), EMA difference
- **Volatility:** ATR (14), rolling volatility (12, 24)
- **Volume:** volume change %, volume moving average (12)

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | StandardScaler pipeline, balanced class weights |
| Random Forest | 300 trees, max depth 8, balanced class weights |
| XGBoost | 300 estimators, max depth 4, learning rate 0.03 |
| Dummy (baseline) | Prior-probability reference — a model only counts if it beats this |

**Evaluation metrics:** Accuracy, Precision, Recall, F1, ROC AUC, Brier Score, Log Loss, and the confusion matrix for every model.

## Project Structure

```
bitcoin-15m-direction-prediction/
├── src/
│   ├── data_collection.py      # Binance API download + validation
│   ├── data_cleaning.py        # Cleaning and candle validation
│   ├── feature_engineering.py  # 22 technical indicators + target
│   ├── model_training.py       # Model comparison and selection
│   └── prediction.py           # Live prediction on the latest closed candle
├── data/                       # Generated CSVs (not tracked in Git)
├── models/                     # Trained model .pkl (not tracked in Git)
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

python src/data_collection.py      # 1. Download / update BTCUSDT 15m data
python src/data_cleaning.py        # 2. Clean and validate
python src/feature_engineering.py  # 3. Build features
python src/model_training.py       # 4. Train, compare and save the best model
python src/prediction.py           # 5. Predict the next candle
```

Run the scripts from the repository root, in order.

## Data Source

- Binance public klines API (`/api/v3/klines`), symbol BTCUSDT, interval 15m
- Historical series starts 2025-10-11; only closed candles are used

## Skills Demonstrated

- API data collection and incremental ETL with validation
- Data cleaning and integrity checks for time-series data
- Feature engineering with technical indicators
- Machine learning classification (scikit-learn, XGBoost)
- Time-series-aware evaluation (chronological split, probability scoring, baseline comparison)
- Python (Pandas, NumPy)

## Author

Domingo Pineda Sabino — Data Analyst | SQL · Power BI · Tableau · Python
LinkedIn: https://www.linkedin.com/in/domingo-pineda-sabino-data
GitHub: https://github.com/Domingopineda
