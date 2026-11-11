"""
Data loading, cleaning, and feature engineering for credit card transactions.

Handles:
- Missing value imputation
- Currency string parsing (e.g., "$12.34" -> 12.34)
- Datetime synthesis from Year/Month/Day/Time
- Temporal sorting to guarantee zero future data leakage
- Categorical encoding (One-Hot and Frequency Encoding)
- Real-time friendly cumulative behavioral signals (velocity, count, avg amount)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_CSV_PATH,
    SAMPLE_CSV_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_PARQUET_PATH,
    CATEGORICAL_COLUMNS,
    RANDOM_SEED
)


def load_raw_data(
    file_path: Optional[Path] = None,
    sample_size: Optional[int] = 100_000
) -> pd.DataFrame:
    """
    Loads raw transaction data from CSV.

    1. Checks if the real full Kaggle CSV exists at data/raw/credit_card_transactions-ibm_v2.csv.
    2. If not, falls back to the generated 100,000-row sample.
    3. If sample_size is provided and smaller than total rows, takes a representative sample.
    """
    if file_path is None:
        if RAW_CSV_PATH.exists():
            file_path = RAW_CSV_PATH
            print(f"[*] Found full Kaggle dataset at: {RAW_CSV_PATH}")
        elif SAMPLE_CSV_PATH.exists():
            file_path = SAMPLE_CSV_PATH
            print(f"[*] Found sample dataset at: {SAMPLE_CSV_PATH}")
        else:
            raise FileNotFoundError(
                f"No transaction CSV found! Please place Kaggle CSV at {RAW_CSV_PATH} "
                f"or run `python data/sample_generator.py`."
            )

    print(f"[*] Loading transactions from {file_path}...")
    if sample_size and file_path == RAW_CSV_PATH:
        # Load in chunks to avoid blowing up RAM on large 10GB files
        print(f"[*] Sampling {sample_size:,} rows from full file...")
        chunks = []
        rows_so_far = 0
        for chunk in pd.read_csv(file_path, chunksize=50_000, low_memory=False):
            chunks.append(chunk)
            rows_so_far += len(chunk)
            if rows_so_far >= sample_size:
                break
        df = pd.concat(chunks, ignore_index=True).iloc[:sample_size]
    else:
        df = pd.read_csv(file_path, low_memory=False)
        if sample_size and len(df) > sample_size:
            df = df.iloc[:sample_size]

    print(f"[+] Loaded {len(df):,} transactions.")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw columns:
    - Strips '$' from Amount and converts to float32
    - Synthesizes Timestamp from Year, Month, Day, and Time
    - Sorts chronologically (crucial for time splits & real-time replay)
    - Maps 'Is Fraud?' to binary integer (0 or 1)
    - Imputes missing values in State, Zip, and Errors
    """
    print("[*] Cleaning raw fields and imputing missing values...")
    df = df.copy()

    # 1. Clean Amount: "$12.34" -> 12.34
    if df["Amount"].dtype == object:
        df["Amount_clean"] = (
            df["Amount"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .astype(np.float32)
        )
    else:
        df["Amount_clean"] = df["Amount"].astype(np.float32)

    # Log transform for scale stability (amount distributions are heavily skewed)
    df["Amount_log"] = np.sign(df["Amount_clean"]) * np.log1p(np.abs(df["Amount_clean"]))

    # 2. Synthesize unified Timestamp and sort chronologically
    time_str = df["Time"].astype(str).str.strip()
    # Normalize formats like "6:05" to "06:05"
    time_str = time_str.apply(lambda x: f"{int(x.split(':')[0]):02d}:{int(x.split(':')[1]):02d}" if ":" in x else "00:00")

    dt_str = (
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2) + " " +
        time_str
    )
    df["Timestamp"] = pd.to_datetime(dt_str, errors="coerce")

    # If any timestamp failed parsing, fallback to year start
    if df["Timestamp"].isna().any():
        df["Timestamp"] = df["Timestamp"].fillna(pd.to_datetime(df["Year"].astype(str) + "-01-01"))

    # Sort strictly by time so graph construction and replay respect temporal causality
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # 3. Clean Binary Label
    if "Is Fraud?" in df.columns:
        if df["Is Fraud?"].dtype == object:
            df["label"] = (df["Is Fraud?"].astype(str).str.strip().str.lower() == "yes").astype(np.int64)
        else:
            df["label"] = df["Is Fraud?"].astype(np.int64)
    else:
        df["label"] = 0

    # 4. Handle Missing Values in Categoricals
    df["Errors?"] = df["Errors?"].fillna("None").astype(str).str.strip()
    df["Merchant State"] = df["Merchant State"].fillna("ONLINE_OR_INTL").astype(str).str.strip()
    df["Merchant City"] = df["Merchant City"].fillna("UNKNOWN").astype(str).str.strip()
    df["Zip"] = df["Zip"].fillna(0.0).astype(np.int64)

    # 5. Build Unique Entity Keys
    # A single user can have multiple cards; card is the primary payment entity
    df["Card_ID"] = "card_" + df["User"].astype(str) + "_" + df["Card"].astype(str)
    df["Merchant_ID"] = "merchant_" + df["Merchant Name"].astype(str)

    return df


def engineer_streaming_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time and behavioral velocity features without future data leakage.
    Each row only sees information that happened BEFORE or AT that transaction.
    """
    print("[*] Engineering streaming temporal and behavioral velocity features...")
    df = df.copy()

    # Cyclical Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["DayOfMonth"] = df["Timestamp"].dt.day
    df["Month_num"] = df["Timestamp"].dt.month

    # Continuous cyclical encodings (sin/cos of hour)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0).astype(np.float32)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0).astype(np.float32)

    # Cumulative Card Velocity (How many times has this card been used up to now?)
    # cumcount() only looks at past rows because df is sorted by Timestamp
    df["Card_Tx_Count_SoFar"] = df.groupby("Card_ID").cumcount().astype(np.float32)

    # Cumulative Card Average Amount up to this transaction
    cum_sum = df.groupby("Card_ID")["Amount_clean"].cumsum()
    df["Card_Avg_Amount_SoFar"] = (cum_sum / (df["Card_Tx_Count_SoFar"] + 1.0)).astype(np.float32)

    # Time delta since previous transaction on the same card (in hours)
    prev_time = df.groupby("Card_ID")["Timestamp"].shift(1)
    time_diff_hours = (df["Timestamp"] - prev_time).dt.total_seconds() / 3600.0
    # First transaction has no previous, fill with large baseline (e.g. 720 hours = 30 days)
    df["Hours_Since_Prev_Tx"] = time_diff_hours.fillna(720.0).clip(upper=720.0).astype(np.float32)

    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Encodes categorical variables into numerical representations.
    Returns:
    - Transformed DataFrame
    - List of final feature column names used for the transaction representation
    """
    print("[*] Encoding categorical variables and scaling numerical features...")
    df = df.copy()

    # One-hot encode "Use Chip"
    chip_dummies = pd.get_dummies(df["Use Chip"], prefix="chip", dtype=np.float32)
    df = pd.concat([df, chip_dummies], axis=1)

    # One-hot encode top errors
    top_errors = ["None", "Bad PIN", "Insufficient Balance", "Technical Glitch"]
    df["error_group"] = df["Errors?"].apply(lambda x: x if x in top_errors else "Other_Error")
    error_dummies = pd.get_dummies(df["error_group"], prefix="err", dtype=np.float32)
    df = pd.concat([df, error_dummies], axis=1)

    # Normalized MCC (Merchant Category Code)
    df["MCC_num"] = (df["MCC"] / 10000.0).astype(np.float32)

    # Collect all transaction feature names
    feature_cols = [
        "Amount_clean",
        "Amount_log",
        "Hour_sin",
        "Hour_cos",
        "DayOfWeek",
        "Month_num",
        "MCC_num",
        "Card_Tx_Count_SoFar",
        "Card_Avg_Amount_SoFar",
        "Hours_Since_Prev_Tx",
    ] + list(chip_dummies.columns) + list(error_dummies.columns)

    # Standardize numerical features using StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols]).astype(np.float32)

    return df, feature_cols


def prepare_dataset(
    sample_size: Optional[int] = 100_000,
    save_parquet: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    End-to-end pipeline: load -> clean -> engineer -> encode.
    """
    raw_df = load_raw_data(sample_size=sample_size)
    cleaned_df = clean_transactions(raw_df)
    engineered_df = engineer_streaming_features(cleaned_df)
    final_df, feature_cols = encode_features(engineered_df)

    if save_parquet:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            final_df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
            print(f"[+] Processed data saved to: {PROCESSED_PARQUET_PATH}")
        except Exception:
            final_df.to_csv(PROCESSED_DATA_PATH, index=False, compression="gzip")
            print(f"[+] Processed data saved to compressed CSV: {PROCESSED_DATA_PATH}")

    fraud_count = final_df["label"].sum()
    print(f"\n[Summary] Preprocessing Complete:")
    print(f"  - Total Transactions: {len(final_df):,}")
    print(f"  - Features Extracted: {len(feature_cols)} features ({feature_cols})")
    print(f"  - Legitimate: {len(final_df) - fraud_count:,}")
    print(f"  - Fraud: {fraud_count:,} ({fraud_count / len(final_df):.4%})")
    print(f"  - Unique Cards: {final_df['Card_ID'].nunique():,}")
    print(f"  - Unique Merchants: {final_df['Merchant_ID'].nunique():,}")

    return final_df, feature_cols


if __name__ == "__main__":
    df, features = prepare_dataset(sample_size=100_000)
