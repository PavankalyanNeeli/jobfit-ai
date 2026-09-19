"""
Script to download Kaggle data for JobFit AI.

This script fetches real HR and job posting datasets from Kaggle to use 
for evaluating the JobFit AI models.

To set up Kaggle API credentials:
1. Go to https://www.kaggle.com/settings
2. Scroll to 'API' and click 'Create New Token'
3. Download 'kaggle.json'
4. Place it in ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
5. Ensure the file has restricted permissions (e.g., chmod 600 ~/.kaggle/kaggle.json)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR

DATASETS = [
    "arashnic/hr-analytics-job-change-of-data-scientists",
    "shivamb/real-or-fake-fake-jobposting-prediction"
]

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for JobFit AI.")
    parser.add_argument("--dry-run", action="store_true", help="Check API connection without downloading")
    args = parser.parse_args()

    try:
        import kaggle
    except ImportError:
        print("Error: The 'kaggle' package is not installed.")
        print("Install it via: pip install kaggle")
        sys.exit(1)
    except OSError as e:
        print(f"Error initializing Kaggle API: {e}")
        print("Please ensure your kaggle.json is correctly placed and has proper permissions.")
        sys.exit(1)

    # Initialize API
    kaggle.api.authenticate()

    if args.dry_run:
        print("Dry run successful! Kaggle API is connected and authenticated.")
        print(f"Would download the following datasets to {RAW_DIR}:")
        for ds in DATASETS:
            print(f" - {ds}")
        sys.exit(0)

    # Ensure raw directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download datasets
    for dataset in DATASETS:
        print(f"Downloading {dataset}...")
        try:
            kaggle.api.dataset_download_files(dataset, path=str(RAW_DIR), unzip=True)
            print(f"Successfully downloaded and extracted {dataset} to {RAW_DIR}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")

if __name__ == "__main__":
    main()
