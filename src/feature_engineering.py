import pandas as pd
import numpy as np

from pathlib import Path

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange


# ==========================================
# PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "bitcoin_15m.csv"

FEATURES_PATH = ROOT / "data" / "bitcoin_15m_features.csv"


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

    df = df.copy()

    df["time_utc"] = pd.to_datetime(
        df["time_utc"],
        utc=True
    )

    df = df.sort_values("time_utc").reset_index(drop=True)

    # Returns
    df["Returns"] = df["close"].pct_change()

    df["change_pct"] = df["Returns"]

    # Candle range
    df["range_pct"] = (
        (df["high"] - df["low"]) / df["open"]
    )

    # RSI
    df["RSI_7"] = RSIIndicator(
        close=df["close"],
        window=7
    ).rsi()

    df["RSI_21"] = RSIIndicator(
        close=df["close"],
        window=21
    ).rsi()

    # EMA
    df["EMA_9"] = EMAIndicator(
        close=df["close"],
        window=9
    ).ema_indicator()

    df["EMA_21"] = EMAIndicator(
        close=df["close"],
        window=21
    ).ema_indicator()

    # MACD
    macd = MACD(close=df["close"])

    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()

    # ATR
    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["ATR"] = atr.average_true_range()

    # Momentum
    df["Momentum_3"] = df["close"].pct_change(3)
    df["Momentum_6"] = df["close"].pct_change(6)
    df["Momentum_12"] = df["close"].pct_change(12)

    # Volatility
    df["Volatility_12"] = df["Returns"].rolling(12).std()
    df["Volatility_24"] = df["Returns"].rolling(24).std()

    # Volume
    df["Volume_Change"] = df["volume"].pct_change()

    df["Volume_MA_12"] = (
        df["volume"].rolling(12).mean()
    )

    # Candle structure
    df["Candle_Body"] = (
        (df["close"] - df["open"]) / df["open"]
    )

    df["Upper_Wick"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / df["open"]

    df["Lower_Wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["open"]

    # EMA relationship
    df["EMA_Difference"] = (
        (df["EMA_9"] - df["EMA_21"]) / df["close"]
    )

    # Future return
    df["Future_Return"] = (
        df["close"].shift(-1) / df["close"] - 1
    )

    # Target
    df["Target"] = np.where(
        df["Future_Return"].notna(),
        (df["Future_Return"] > 0).astype(float),
        np.nan
    )

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Drop rows with missing FEATURES only.
    # Preserve the last row even when Target is unknown.

    feature_columns = [
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

    df = df.dropna(
        subset=feature_columns
    )

    return df


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\nLoading BTC historical data...")

    df = pd.read_csv(DATA_PATH)

    print("Raw records:", len(df))

    print(
        "Last raw candle:",
        df["time_utc"].iloc[-1]
    )

    df = create_features(df)

    FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURES_PATH,
        index=False
    )

    print("\nFEATURE ENGINEERING COMPLETED")

    print("Dataset shape:", df.shape)

    print("\nLAST 5 CANDLES:")

    print(
        df[
            ["time_utc", "close", "RSI_7", "RSI_21", "Target"]
        ].tail()
    )

    print("\nLAST AVAILABLE CANDLE:")

    print(df["time_utc"].iloc[-1])

    print("\nTARGET DISTRIBUTION:")

    print(
        df["Target"].value_counts(normalize=True) * 100
    )

    print("\nSaved to:", FEATURES_PATH)