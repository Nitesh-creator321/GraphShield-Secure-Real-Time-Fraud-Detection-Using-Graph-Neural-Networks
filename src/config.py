"""
Central configuration file for GraphShield.
Contains directory paths, Kaggle dataset schema definitions,
feature engineering constants, and reproducibility seeds.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"

# Ensure runtime directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default data file paths
RAW_CSV_FILENAME = "credit_card_transactions-ibm_v2.csv"
RAW_CSV_PATH = DATA_RAW_DIR / RAW_CSV_FILENAME
SAMPLE_CSV_PATH = DATA_RAW_DIR / "transactions_sample_100k.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_transactions.csv.gz"
PROCESSED_PARQUET_PATH = DATA_PROCESSED_DIR / "cleaned_transactions.parquet"
GRAPH_DATA_PATH = DATA_PROCESSED_DIR / "graph_data.pt"

# Kaggle IBM Credit Card Dataset Schema (15 columns)
RAW_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?"
]

# Sensitive identification columns to protect / mask
SENSITIVE_COLUMNS = ["User", "Card", "Merchant Name"]

# Categorical columns to encode
CATEGORICAL_COLUMNS = ["Use Chip", "Merchant State", "Errors?"]

# Numerical columns to scale
NUMERICAL_COLUMNS = [
    "Amount_clean",
    "Amount_log",
    "Hour",
    "DayOfWeek",
    "DayOfMonth",
    "Month_num",
    "MCC_num",
    "Card_Tx_Count_SoFar",
    "Card_Avg_Amount_SoFar"
]

# Time Split Configuration (Strict temporal ordering to prevent data leakage)
# Training: Historical baseline (e.g. 1991-2018)
# Validation: Model tuning & threshold selection (e.g. 2019)
# Streaming / Live test: Real-time simulation (e.g. 2020)
TRAIN_END_YEAR = 2018
VAL_YEAR = 2019
STREAM_START_YEAR = 2020

# Model & Graph Hyperparameters
RANDOM_SEED = 42
FRAUD_RATIO_TARGET = 0.0012  # ~0.12% natural fraud prevalence in Kaggle dataset
DEFAULT_ALERT_THRESHOLD = 0.50
NEIGHBORHOOD_HOPS = 2
MAX_NEIGHBORS_PER_HOP = 10
