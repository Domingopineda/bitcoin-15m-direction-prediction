import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ==================================================
# 1. CONFIGURATION
# ==================================================

# Main project directory
ROOT = Path(__file__).resolve().parent.parent

# File paths
MODEL_PATH = ROOT / "models" / "best_btc_model.pkl"

DATA_PATH = ROOT / "data" / "bitcoin_15m_features.csv"


# Features used during model training
FEATURES = [
    "Returns",
    "change_pct",
    "range_pct",
    "RSI_7",
    "RSI_21",
    "EMA_9",
    "EMA_21",
    "MACD",
    "MACD_Signal",
    "MACD_Diff",
    "ATR",
    "Momentum_3",
    "Momentum_6",
    "Momentum_12",
    "Volatility_12",
    "Volatility_24",
    "Volume_Change",
    "Volume_MA_12",
    "Candle_Body",
    "Upper_Wick",
    "Lower_Wick",
    "EMA_Difference"
]


# ==================================================
# 2. LOAD MODEL
# ==================================================

def load_model():

    print("\nLoading model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found: {MODEL_PATH}\n"
            "Run model_training.py first."
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    print("Model path:", MODEL_PATH)

    return model


# ==================================================
# 3. LOAD LATEST FEATURES
# ==================================================

def load_latest_features():

    print("\nLoading latest BTC data...")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nFeature dataset not found: {DATA_PATH}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:

        raise ValueError("Feature dataset is empty.")

    # Check required columns

    required_columns = FEATURES + [
        "time_utc",
        "close"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in dataset: {missing}"
        )

    # Convert timestamp

    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        utc=True,
        errors="coerce"
    )

    # Sort chronologically

    df = df.sort_values(
        "time_utc"
    ).reset_index(drop=True)

    # Convert features to numeric

    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    # Remove invalid values

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Target is NOT required here.
    # The latest candle has Target = NaN
    # because its future result is unknown.

    valid = df.dropna(
        subset=FEATURES + [
            "time_utc",
            "close"
        ]
    )

    if valid.empty:

        raise ValueError(
            "No valid feature records available."
        )

    # Latest valid candle

    latest = valid.iloc[-1:]

    timestamp = latest["time_utc"].iloc[0]

    # Check whether candle is closed

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    candle_close_time = timestamp + pd.Timedelta(
        minutes=15
    )

    if candle_close_time > now_utc:

        raise ValueError(
            "\nLatest candle is not closed yet.\n"
            "Update your data and try again."
        )

    # Prepare model input

    X = latest[FEATURES].copy()

    print("Latest data loaded successfully!")

    print("Available records:", len(valid))

    print("Latest candle UTC:", timestamp)

    return latest, X


# ==================================================
# 4. VERIFY MODEL FEATURES
# ==================================================

def verify_model(model):

    print("\nChecking model compatibility...")

    if hasattr(model, "n_features_in_"):

        if model.n_features_in_ != len(FEATURES):

            raise ValueError(
                "\nModel feature count does not match.\n"
                f"Model expects: {model.n_features_in_}\n"
                f"Prediction provides: {len(FEATURES)}\n"
                "Retrain the model."
            )

    # Check feature names when available

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

        if model_features != FEATURES:

            raise ValueError(
                "\nModel feature names or order do not match.\n"
                "Retrain the model using the current feature list."
            )

    print("Model compatibility verified!")


# ==================================================
# 5. GENERATE PREDICTION
# ==================================================

def predict():

    # Load model

    model = load_model()

    # Load latest data

    latest, X = load_latest_features()

    # Verify model

    verify_model(model)

    # Generate probabilities

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if 1 not in classes or 0 not in classes:

        raise ValueError(
            "Model must contain classes 0 and 1."
        )

    probability_up = probabilities[
        classes.index(1)
    ]

    probability_not_up = probabilities[
        classes.index(0)
    ]

    # Get candle information

    timestamp = latest["time_utc"].iloc[0]

    current_price = latest["close"].iloc[0]

    # Forecast next 15-minute candle

    next_candle_start = timestamp + pd.Timedelta(
        minutes=15
    )

    next_candle_end = next_candle_start + pd.Timedelta(
        minutes=15
    )

    # Convert timestamps to New York time

    timestamp_ny = timestamp.tz_convert(
        "America/New_York"
    )

    next_start_ny = next_candle_start.tz_convert(
        "America/New_York"
    )

    next_end_ny = next_candle_end.tz_convert(
        "America/New_York"
    )

    # ==================================================
    # 6. DISPLAY RESULTS
    # ==================================================

    print("\n")
    print("=" * 55)

    print("       BTCUSDT 15 MINUTE PREDICTION")

    print("=" * 55)

    print("\nLAST CLOSED CANDLE")

    print(
        "UTC:",
        timestamp.strftime("%Y-%m-%d %H:%M")
    )

    print(
        "NEW YORK:",
        timestamp_ny.strftime("%Y-%m-%d %I:%M %p")
    )

    print(
        f"\nCURRENT BTC PRICE: ${current_price:,.2f}"
    )

    print("\nFORECAST CANDLE")

    print(
        next_start_ny.strftime("%I:%M %p"),
        "-",
        next_end_ny.strftime("%I:%M %p")
    )

    print("\n" + "=" * 55)

    print("\nMODEL PROBABILITIES")

    print(
        f"PROBABILITY UP: {probability_up:.2%}"
    )

    print(
        f"PROBABILITY NOT UP: {probability_not_up:.2%}"
    )

    print("\n" + "=" * 55)

    # ==================================================
    # 7. MODEL SIGNAL
    # ==================================================

    if probability_up >= 0.55:

        signal = "UP"

    elif probability_up <= 0.45:

        signal = "DOWN / NOT UP"

    else:

        signal = "NO CLEAR EDGE"

    print("\nMODEL SIGNAL:", signal)

    print("\n" + "=" * 55)

    print(
        "\nRemember: probabilities are estimates, "
        "not guarantees."
    )

    print(
        "\nThis model predicts whether the next "
        "15-minute candle closes above the current close."
    )

    print("\n")


# ==================================================
# 8. RUN PROGRAM
# ==================================================

if __name__ == "__main__":

    predict()