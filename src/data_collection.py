import pandas as pd
import requests
import time

from pathlib import Path


# =====================================================
# 1. CONFIGURATION
# =====================================================

SYMBOL = "BTCUSDT"
INTERVAL = "15m"

LIMIT = 1000
INTERVAL_MS = 15 * 60 * 1000

# Project directory
ROOT = Path(__file__).resolve().parent

if ROOT.name == "src":
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data" / "bitcoin_15m.csv"

# Alternative Binance public endpoints
BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com"
]

COLUMNS = [
    "time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore"
]


# =====================================================
# 2. BINANCE API CONNECTION
# =====================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "BTC-15m-Data-Collection/1.0"
})


def binance_request(endpoint, params=None):

    last_error = None

    for base_url in BASE_URLS:

        url = base_url + endpoint

        try:

            print(f"Connecting to: {base_url}")

            response = session.get(
                url,
                params=params,
                timeout=30
            )

            # Binance restriction
            if response.status_code == 451:

                print("Endpoint restricted. Trying alternative...")
                continue

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as error:

            last_error = error

            print(f"Connection failed: {error}")

    raise ConnectionError(
        f"Could not connect to Binance endpoints. Last error: {last_error}"
    )


# =====================================================
# 3. DOWNLOAD BTCUSDT CANDLES
# =====================================================

def get_binance_data(start_time):

    params = {

        "symbol": SYMBOL,

        "interval": INTERVAL,

        "limit": LIMIT,

        "startTime": start_time

    }

    data = binance_request(
        "/api/v3/klines",
        params
    )

    if not data:

        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(
        data,
        columns=KLINE_COLUMNS
    )

    # Convert timestamps to UTC
    df["time_utc"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    # Convert OHLCV to numeric
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# =====================================================
# 4. DOWNLOAD ALL NEW CLOSED CANDLES
# =====================================================

def download_new_candles(start_time):

    all_data = []

    while True:

        current_time_ms = int(
            time.time() * 1000
        )

        if start_time >= current_time_ms:

            break

        batch = get_binance_data(
            start_time
        )

        if batch.empty:

            break

        # Exclude the candle that is still open.
        # A candle is closed only when close_time is
        # earlier than the current UTC time.

        close_times = pd.to_numeric(
            batch["close_time"],
            errors="coerce"
        )

        current_batch = batch.loc[
            close_times < current_time_ms
        ].copy()

        if not current_batch.empty:

            all_data.append(
                current_batch[COLUMNS]
            )

            last_timestamp = current_batch[
                "time_utc"
            ].max()

            print(
                "Downloaded until:",
                last_timestamp
            )

        last_open_time = int(
            batch["open_time"].iloc[-1]
        )

        next_start_time = (
            last_open_time + INTERVAL_MS
        )

        if next_start_time <= start_time:

            raise RuntimeError(
                "Pagination stopped: timestamp did not advance."
            )

        start_time = next_start_time

        if len(batch) < LIMIT:

            break

        time.sleep(0.25)

    if not all_data:

        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(
        all_data,
        ignore_index=True
    )


# =====================================================
# 5. LOAD EXISTING DATA
# =====================================================

def load_existing_data():

    if DATA_PATH.exists():

        print("\nLoading existing historical data...")

        df = pd.read_csv(DATA_PATH)

        if "time_utc" not in df.columns:

            raise ValueError(
                "Existing CSV does not contain time_utc."
            )

        df["time_utc"] = pd.to_datetime(
            df["time_utc"],
            errors="coerce",
            utc=True
        )

        if df["time_utc"].isna().any():

            raise ValueError(
                "Invalid timestamps found in existing CSV. "
                "The file was not modified."
            )

        missing_columns = [
            column for column in COLUMNS
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing columns in existing CSV: {missing_columns}"
            )

        print("Existing records:", len(df))

        print(
            "Last saved candle:",
            df["time_utc"].max()
        )

        return df

    print("\nNo historical CSV found.")

    print("Starting initial historical download...")

    return pd.DataFrame(columns=COLUMNS)


# =====================================================
# 6. MAIN UPDATE PROCESS
# =====================================================

def main():

    print("\n======================================")

    print("BTCUSDT 15 MINUTE DATA COLLECTION")

    print("======================================")

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load previous data
    existing = load_existing_data()

    # Determine download starting point
    if not existing.empty:

        last_timestamp = existing[
            "time_utc"
        ].max()

        start_time = int(
            last_timestamp.timestamp() * 1000
        ) + INTERVAL_MS

    else:

        # Initial historical date
        initial_date = pd.Timestamp(
            "2025-10-11",
            tz="UTC"
        )

        start_time = int(
            initial_date.timestamp() * 1000
        )

    # Download new candles
    print("\nDownloading new BTCUSDT candles...")

    new_data = download_new_candles(
        start_time
    )

    print(
        "\nNew candles downloaded:",
        len(new_data)
    )

    # =================================================
    # 7. MERGE DATA
    # =================================================

    combined = pd.concat(
        [existing, new_data],
        ignore_index=True
    )

    if combined.empty:

        print("\nNo data available.")

        return

    combined["time_utc"] = pd.to_datetime(
        combined["time_utc"],
        errors="coerce",
        utc=True
    )

    # Sort chronologically
    combined = combined.sort_values(
        "time_utc"
    )

    # Remove duplicate timestamps
    duplicates_before = combined[
        "time_utc"
    ].duplicated().sum()

    combined = combined.drop_duplicates(
        subset="time_utc",
        keep="last"
    )

    combined = combined.reset_index(
        drop=True
    )

    # =================================================
    # 8. DATA VALIDATION
    # =================================================

    print("\n========== DATA VALIDATION ==========")

    duplicate_count = combined[
        "time_utc"
    ].duplicated().sum()

    missing_values = combined[
        COLUMNS
    ].isnull().sum().sum()

    time_differences = combined[
        "time_utc"
    ].diff().dropna()

    expected_interval = pd.Timedelta(
        minutes=15
    )

    gaps = time_differences[
        time_differences > expected_interval
    ]

    irregular = time_differences[
        time_differences < expected_interval
    ]

    print(
        "Total records:",
        len(combined)
    )

    print(
        "Duplicates removed:",
        duplicates_before
    )

    print(
        "Remaining duplicates:",
        duplicate_count
    )

    print(
        "Missing values:",
        missing_values
    )

    print(
        "Gaps greater than 15 minutes:",
        len(gaps)
    )

    print(
        "Irregular intervals:",
        len(irregular)
    )

    print(
        "Chronological order:",
        combined["time_utc"].is_monotonic_increasing
    )

    # Validate OHLCV values
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce"
        )

    invalid_ohlcv = (
        combined[
            ["open", "high", "low", "close", "volume"]
        ].isna().any(axis=1)
        |
        (combined["high"] < combined["low"])
        |
        (combined["high"] < combined[["open", "close"]].max(axis=1))
        |
        (combined["low"] > combined[["open", "close"]].min(axis=1))
        |
        (combined["volume"] < 0)
    )

    print(
        "Invalid OHLCV records:",
        int(invalid_ohlcv.sum())
    )

    # =================================================
    # 9. SAVE UPDATED DATA
    # =================================================

    # Preserve any extra columns in the existing CSV.
    extra_columns = [
        column for column in combined.columns
        if column not in COLUMNS
    ]

    combined = combined[
        COLUMNS + extra_columns
    ]

    temp_path = DATA_PATH.with_suffix(".tmp")

    combined["time_utc"] = combined[
        "time_utc"
    ].dt.strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )

    combined.to_csv(
        temp_path,
        index=False
    )

    temp_path.replace(DATA_PATH)

    # =================================================
    # 10. FINAL REPORT
    # =================================================

    print("\n======================================")

    print("DATA COLLECTION COMPLETED")

    print("======================================")

    print(
        "New candles:",
        len(new_data)
    )

    print(
        "Total records:",
        len(combined)
    )

    print(
        "Last timestamp UTC:",
        combined["time_utc"].iloc[-1]
    )

    print(
        "CSV location:",
        DATA_PATH
    )

    if len(gaps) > 0:

        print(
            "\nWARNING: Historical time gaps detected."
        )

    if invalid_ohlcv.sum() > 0:

        print(
            "\nWARNING: Invalid OHLCV records detected."
        )

    print("\n======================================")


# =====================================================
# 11. RUN SCRIPT
# =====================================================

if __name__ == "__main__":

    main()
