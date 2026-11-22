import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("data/bitcoin_15m.csv")

print("INITIAL DATASET:", df.shape)

# 1. Convert time_utc to datetime
df['time_utc'] = pd.to_datetime(
    df["time_utc"],
   errors="coerce",
    utc=True 
    )

# 2. Remove duplicates records
df = df.drop_duplicates()

# 3. Convert numerical columns
numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "change_pct",
    "range_pct"
]


for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# 5. Remove invalid numerical records
df = df.dropna(subset=numeric_columns)

# 6. Validate prices and volume
df = df[
    (df["open"] > 0) &
    (df["high"] > 0) &
    (df["low"] > 0) &
    (df["close"] > 0) &
    (df["volume"] >= 0)
]

# 7. Validate candle structure
df = df[
    (df["high"] >= df["low"]) &
    (df["high"] >= df["open"]) &
    (df["high"] >= df["close"]) &
    (df["low"] <= df["open"]) &
    (df["low"] <= df["close"])
]

# 8. Sort chronologically
df = df.sort_values("time_utc")


# 9. Reset index
df = df.reset_index(drop=True)

# 10. Check 15- minute intervals
df["time_difference"] = df["time_utc"].diff()

print("\nFINAL DATASET:", df.shape)

print("\nDATA TYPES:")
print(df.dtypes)

print("\nDUPLICATES:")
print(df.duplicated(subset=["time_utc"]).sum())

print("\nTIME INTERVALS:")
print(df["time_difference"].value_counts().head())

print("\nDATE RANGE:")
print(df["time_utc"].min())
print(df["time_utc"].max())

#Save Cleaned dataset
df.drop(columns=["time_difference"]).to_csv(
    "data/bitcoin_15m_cleaned.csv",
    index=False
)

print("\nCLEANING COMPLETED!")





