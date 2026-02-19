"""Command line utilities for syncing and exporting ULSI outputs."""

from __future__ import annotations

import argparse
from datetime import date

from .dashboard import build_dashboard_table
from .data import fetch_all_series
from .metrics import compute_features, compute_ulsi


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync USD liquidity inputs and export ULSI outputs")
    parser.add_argument("--start", type=_parse_date, required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, required=True, help="End date in YYYY-MM-DD")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    raw_df, statuses = fetch_all_series(start=args.start, end=args.end)
    features_df = compute_features(raw_df)
    ulsi_df = compute_ulsi(features_df)
    unified_df = build_dashboard_table(raw_df, features_df, ulsi_df)
    unified_df.to_csv(args.output, index=False)

    print(f"Exported unified dataset to: {args.output}")
    print("Sync summary:")
    for name, status in statuses.items():
        print(
            f"- {name}: success={status.success}, source={status.source}, rows={status.row_count}, "
            f"latest={status.latest_date}, message={status.message}"
        )


if __name__ == "__main__":
    main()
