"""
Single-command launcher for GraphShield Live Dashboard.

Usage:
    python run_dashboard.py

Starts the FastAPI server and automatically opens your web browser to http://localhost:8000.
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_PATH, SAMPLE_CSV_PATH


def open_browser():
    """Waits 1.5 seconds for the server to start, then opens the browser."""
    time.sleep(1.5)
    url = "http://localhost:8000"
    print(f"\n[+] Opening web browser at: {url}")
    webbrowser.open(url)


def main():
    print("=" * 65)
    print("     GraphShield: Secure Real-Time GNN Fraud Defense")
    print("=" * 65)

    # 1. Verify that data exists; if not, generate sample
    if not PROCESSED_DATA_PATH.exists() and not SAMPLE_CSV_PATH.exists():
        print("[*] No dataset found. Generating 100k sample dataset...")
        from data.sample_generator import generate_kaggle_sample
        from src.data.loader import prepare_dataset
        generate_kaggle_sample(100_000)
        prepare_dataset(100_000)
    elif not PROCESSED_DATA_PATH.exists() and SAMPLE_CSV_PATH.exists():
        print("[*] Preprocessing raw sample dataset...")
        from src.data.loader import prepare_dataset
        prepare_dataset(100_000)

    # 2. Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Start Uvicorn FastAPI Server
    import uvicorn
    print("\n[+] Starting FastAPI Web Server at http://127.0.0.1:8000 ...")
    print("[+] Press Ctrl+C in this terminal at any time to stop the server.\n")

    uvicorn.run("src.serving.app:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

