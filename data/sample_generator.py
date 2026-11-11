"""
Realistic Sample Dataset Generator for GraphShield.

Generates a ~100,000-row synthetic dataset matching the exact 15-column schema,
data types, value formats, and ~0.12% fraud rate of the IBM Kaggle Credit Card
Transactions dataset (1991-2020).

Use this for rapid experimentation and testing without needing to download
the entire ~10GB Kaggle file.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import SAMPLE_CSV_PATH, RANDOM_SEED, FRAUD_RATIO_TARGET


def generate_kaggle_sample(
    num_rows: int = 100_000,
    fraud_ratio: float = FRAUD_RATIO_TARGET,
    output_path: Path = SAMPLE_CSV_PATH,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates a realistic credit card transactions dataset mimicking the Kaggle dataset.

    Columns produced (identical to Kaggle):
    1. User (int)
    2. Card (int)
    3. Year (int, 1991 - 2020)
    4. Month (int, 1 - 12)
    5. Day (int, 1 - 31)
    6. Time (str, "HH:MM")
    7. Amount (str, "$XX.XX")
    8. Use Chip (str, "Swipe Transaction", "Chip Transaction", "Online Transaction")
    9. Merchant Name (int, synthetic merchant ID)
    10. Merchant City (str, e.g. "ONLINE", "New York", "Dallas")
    11. Merchant State (str or NaN, 2-letter state code)
    12. Zip (float or NaN)
    13. MCC (int, 4-digit Merchant Category Code)
    14. Errors? (str or NaN, e.g. "Bad PIN", "Insufficient Balance")
    15. Is Fraud? (str, "No" or "Yes")
    """
    print(f"[*] Generating {num_rows:,} realistic transactions (fraud ratio: {fraud_ratio:.4%})...")
    np.random.seed(seed)

    # 1. User and Card entities
    # A few hundred users, each having 1 to 4 cards
    num_users = max(500, num_rows // 200)
    users = np.random.randint(0, num_users, size=num_rows)
    # Card index per user (0, 1, 2, or 3) with 0 being most common
    cards = np.random.choice([0, 1, 2, 3], size=num_rows, p=[0.70, 0.20, 0.08, 0.02])

    # 2. Merchants & MCCs
    # In real life, merchant popularity follows a Power Law / Pareto distribution
    # (a few grocery chains/online merchants get most transactions)
    num_merchants = max(2000, num_rows // 25)
    merchant_ids = np.random.randint(1000000000, 9999999999, size=num_merchants, dtype=np.int64)
    # Pareto distribution to simulate frequent merchants
    pareto_weights = np.random.pareto(a=1.5, size=num_merchants)
    pareto_weights /= pareto_weights.sum()
    merchants = np.random.choice(merchant_ids, size=num_rows, p=pareto_weights)

    # Standard Merchant Category Codes (MCC)
    common_mccs = [
        5411,  # Grocery Stores, Supermarkets
        5541,  # Service Stations (Fuel)
        5812,  # Eating Places, Restaurants
        5311,  # Department Stores
        5912,  # Drug Stores and Pharmacies
        4829,  # Money Orders / Wire Transfers (High Risk)
        7011,  # Hotels, Motels
        5732,  # Electronic Sales (High Risk)
        5944,  # Jewelry Stores (High Risk)
        4121,  # Taxicabs and Limousines
    ]
    mcc_probs = [0.25, 0.20, 0.20, 0.10, 0.08, 0.03, 0.04, 0.04, 0.02, 0.04]
    mccs = np.random.choice(common_mccs, size=num_rows, p=mcc_probs)

    # 3. Timestamps spanning 1991 to 2020
    # Slightly more transactions in recent years to reflect industry volume growth
    years = np.random.choice(
        np.arange(1991, 2021),
        size=num_rows,
        p=np.linspace(0.01, 0.056, 30) / np.sum(np.linspace(0.01, 0.056, 30))
    )
    months = np.random.randint(1, 13, size=num_rows)
    days = np.random.randint(1, 29, size=num_rows)  # Safe day range for all months
    hours = np.random.randint(0, 24, size=num_rows)
    minutes = np.random.randint(0, 60, size=num_rows)
    times = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]

    # 4. Chip vs Swipe vs Online
    # Chip transactions were introduced mostly after 2014
    chip_types = []
    for yr in years:
        if yr < 2014:
            p = [0.85, 0.00, 0.15]  # Swipe, Chip, Online
        else:
            p = [0.35, 0.45, 0.20]
        chip_types.append(np.random.choice(["Swipe Transaction", "Chip Transaction", "Online Transaction"], p=p))

    # 5. Geo Location (City, State, Zip)
    cities = ["ONLINE", "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Dallas", "Miami", "Seattle"]
    states = [None, "NY", "CA", "IL", "TX", "AZ", "TX", "FL", "WA"]
    zips = [np.nan, 10001.0, 90001.0, 60601.0, 77001.0, 85001.0, 75201.0, 33101.0, 98101.0]

    geo_idx = np.random.choice(len(cities), size=num_rows, p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.09, 0.10, 0.08])
    merchant_cities = [cities[i] for i in geo_idx]
    merchant_states = [states[i] if cities[i] != "ONLINE" else np.nan for i in geo_idx]
    merchant_zips = [zips[i] if cities[i] != "ONLINE" else np.nan for i in geo_idx]

    # 6. Amounts ($XX.XX)
    # Log-normal distribution: most transactions are $10 - $80, with occasional large ones
    raw_amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=num_rows)
    raw_amounts = np.clip(raw_amounts, 0.50, 4500.0)

    # 7. Errors
    error_choices = [np.nan, "Bad PIN", "Insufficient Balance", "Technical Glitch", "Bad Card Number"]
    error_probs = [0.97, 0.012, 0.010, 0.005, 0.003]
    errors = np.random.choice(error_choices, size=num_rows, p=error_probs)

    # 8. Fraud labels (~0.12% target)
    num_frauds = int(num_rows * fraud_ratio)
    fraud_flags = np.array(["No"] * num_rows, dtype=object)

    # In real fraud, fraudsters often target specific high-risk MCCs, online channels, or unusual amounts
    # Pick fraud indices with higher likelihood on online transactions, high amounts, or high-risk MCCs
    fraud_weight_score = (
        (np.array(chip_types) == "Online Transaction").astype(float) * 2.0 +
        (raw_amounts > 200).astype(float) * 2.5 +
        np.isin(mccs, [4829, 5732, 5944]).astype(float) * 3.0 +
        0.5
    )
    fraud_weight_score /= fraud_weight_score.sum()
    fraud_indices = np.random.choice(num_rows, size=num_frauds, replace=False, p=fraud_weight_score)
    fraud_flags[fraud_indices] = "Yes"

    # Fraud amounts tend to be higher or round numbers
    for idx in fraud_indices:
        if np.random.rand() > 0.4:
            raw_amounts[idx] = np.random.choice([199.99, 499.00, 850.00, 1200.50, 2499.00])

    amount_strings = [f"${a:.2f}" for a in raw_amounts]

    # Assemble DataFrame matching Kaggle exact schema
    df = pd.DataFrame({
        "User": users,
        "Card": cards,
        "Year": years,
        "Month": months,
        "Day": days,
        "Time": times,
        "Amount": amount_strings,
        "Use Chip": chip_types,
        "Merchant Name": merchants,
        "Merchant City": merchant_cities,
        "Merchant State": merchant_states,
        "Zip": merchant_zips,
        "MCC": mccs,
        "Errors?": errors,
        "Is Fraud?": fraud_flags
    })

    # Sort chronologically by timestamp (mimics realistic historical order)
    df["dt_sort"] = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str).str.zfill(2) + "-" +
        df["Day"].astype(str).str.zfill(2) + " " +
        df["Time"]
    )
    df = df.sort_values("dt_sort").reset_index(drop=True)
    df = df.drop(columns=["dt_sort"])

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    actual_frauds = (df["Is Fraud?"] == "Yes").sum()
    print(f"[+] Sample successfully saved to: {output_path}")
    print(f"    Rows: {len(df):,}")
    print(f"    Legitimate: {len(df) - actual_frauds:,} ({1 - actual_frauds/len(df):.2%})")
    print(f"    Fraudulent: {actual_frauds:,} ({actual_frauds/len(df):.4%})")
    print(f"    Year range: {df['Year'].min()} - {df['Year'].max()}")

    return df


if __name__ == "__main__":
    generate_kaggle_sample(num_rows=100_000)

