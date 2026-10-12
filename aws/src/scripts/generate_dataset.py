"""Genera la serie cruda sintética en data/local/demand.csv (el mismo
generador que usa la Lambda de ingesta) para trabajar sin AWS.

    python3 -m src.scripts.generate_dataset
    python3 -m src.scripts.generate_dataset --end-date 2025-06-30 --drift
"""
import argparse
import csv
from datetime import datetime

from src.datasets.synthetic_demand import COLUMNS, generate
from src.scripts.paths import LOCAL_DATA_DIR, RAW_DATASET_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    rows = generate(
        datetime.strptime(args.start_date, "%Y-%m-%d"),
        datetime.strptime(args.end_date, "%Y-%m-%d").replace(hour=23),
        args.seed,
        args.drift,
    )
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_DATASET_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows → {RAW_DATASET_PATH}")


if __name__ == "__main__":
    main()
