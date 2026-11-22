import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==================================================
# 1. PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "bitcoin_15m_features.csv"
RESULTS_PATH = ROOT / "data" / "model_comparison.csv"
MODEL_PATH = ROOT / "models" / "best_btc_model.pkl"

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("\nLOADING DATA...")

df = pd.read_csv(DATA_PATH)

df["time_utc"] = pd.to_datetime(
    df["time_utc"],
    utc=True
)

df = df.sort_values("time_utc").reset_index(drop=True)

print("Raw records:", len(df))


# ==================================================
# 3. SELECT FEATURES
# ==================================================

features = [
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


# Check required columns

required_columns = features + ["Target", "time_utc"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ==================================================
# 4. CLEAN TRAINING DATA
# ==================================================

print("\nCLEANING TRAINING DATA...")

# Convert features and target to numeric

for col in features + ["Target"]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Replace infinite values

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Count rows before cleaning

records_before = len(df)

# Remove rows without features or known Target

df = df.dropna(
    subset=features + ["Target"]
).copy()

# Target must be integer: 0 or 1

df["Target"] = df["Target"].astype(int)

# Validate target values

if not df["Target"].isin([0, 1]).all():
    raise ValueError("Target contains values different from 0 and 1.")

records_after = len(df)

print("Records before cleaning:", records_before)
print("Records after cleaning:", records_after)
print("Records removed:", records_before - records_after)

print("\nTARGET DISTRIBUTION:")
print(
    df["Target"].value_counts(normalize=True) * 100
)


# ==================================================
# 5. PREPARE X AND Y
# ==================================================

X = df[features]
y = df["Target"]

if y.nunique() < 2:
    raise ValueError(
        "Training data must contain both classes: 0 and 1."
    )


# ==================================================
# 6. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTRAINING RECORDS:", len(X_train))
print("TEST RECORDS:", len(X_test))

print("\nTRAINING PERIOD:")
print(df["time_utc"].iloc[0])
print(df["time_utc"].iloc[split_index - 1])

print("\nTEST PERIOD:")
print(df["time_utc"].iloc[split_index])
print(df["time_utc"].iloc[-1])

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both training and test periods must contain Target classes 0 and 1."
    )


# ==================================================
# 7. DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ==================================================
# 8. EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, y_true, probabilities, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    loss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1]
    )

    print("\nMODEL:", name)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))
    print("Brier Score:", round(brier, 4))
    print("Log Loss:", round(loss, 4))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_true, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
        "Brier_Score": brier,
        "Log_Loss": loss
    }


# ==================================================
# 9. TRAIN AND EVALUATE MODELS
# ==================================================

results = []

trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)
    print("=" * 55)

    model.fit(X_train, y_train)

    probabilities_all = model.predict_proba(X_test)

    # Find the probability corresponding to class 1 (UP)

    up_index = list(model.classes_).index(1)

    probabilities = probabilities_all[:, up_index]

    predictions = (probabilities >= 0.50).astype(int)

    metrics = evaluate_model(
        name,
        y_test,
        probabilities,
        predictions
    )

    results.append(metrics)

    trained_models[name] = model


# ==================================================
# 10. BASELINE MODEL
# ==================================================

print("\n" + "=" * 55)
print("TRAINING BASELINE MODEL")
print("=" * 55)

baseline = DummyClassifier(
    strategy="prior"
)

baseline.fit(X_train, y_train)

baseline_probabilities_all = baseline.predict_proba(X_test)

baseline_up_index = list(baseline.classes_).index(1)

baseline_probabilities = baseline_probabilities_all[:, baseline_up_index]

baseline_predictions = (
    baseline_probabilities >= 0.50
).astype(int)

baseline_metrics = evaluate_model(
    "Baseline (Prior)",
    y_test,
    baseline_probabilities,
    baseline_predictions
)

results.append(baseline_metrics)


# ==================================================
# 11. COMPARE MODELS
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Brier_Score"
).reset_index(drop=True)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(
    results_df.to_string(index=False)
)


# ==================================================
# 12. SELECT BEST TRAINED MODEL
# ==================================================

# Baseline is a reference, not a production model.

trained_results = results_df[
    results_df["Model"].isin(trained_models.keys())
]

best_model_name = trained_results.iloc[0]["Model"]

best_model = trained_models[best_model_name]


# ==================================================
# 13. SAVE RESULTS AND MODEL
# ==================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print("BEST MODEL:", best_model_name)

print("Results saved to:")
print(RESULTS_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nAVAILABLE FEATURES:")
print(features)

print("\nFINAL TRAINING RECORDS:", len(df))
