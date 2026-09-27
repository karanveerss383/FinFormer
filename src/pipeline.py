import os
from pathlib import Path

import numpy as np
import pandas as pd
import simfin as sf
from dotenv import load_dotenv

BASE = Path(__file__).parent.parent
data_dir = BASE / "data"
join_on = ["SimFinId", "Fiscal Year"]


def init_api() -> str:
    """Read the SimFin API key from the project environment."""
    load_dotenv(BASE / ".env")
    api_key = os.getenv("SIM_FIN")
    if not api_key:
        raise RuntimeError("SIM_FIN API key is missing; set it in .env")
    return api_key


def filter_missing_dict(
    data_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Filter statement columns with the existing 15% missingness rule."""

    filtered_data = {}

    for name, df in data_dict.items():

        per_missing = np.array(df.isna().sum() / df.count() < 0.15)
        inm_cols = df.columns[np.where(per_missing)[0].tolist()]
        new_df = df[inm_cols]
        filtered_data[name] = new_df

    return filtered_data


def main() -> None:
    api_key = init_api()
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create data directory: {data_dir}") from exc

    try:
        sf.set_api_key(api_key)
        sf.set_data_dir(str(data_dir))
    except Exception as exc:
        raise RuntimeError("Could not configure SimFin") from exc

    try:
        balance = sf.load_balance(variant="annual", market="us").reset_index()

        income = sf.load_income(variant="annual", market="us").reset_index()

        cashflow = sf.load_cashflow(variant="annual", market="us").reset_index()
    except Exception as exc:
        raise RuntimeError("Could not load annual US statements from SimFin") from exc

    try:
        companies = sf.load_companies(market="us").reset_index()
    except Exception as exc:
        raise RuntimeError("Could not load US companies from SimFin") from exc

    data_dict = {
        "balance": balance,
        "income": income,
        "cashflow": cashflow,
        "companies": companies,
    }

    filtered_data = filter_missing_dict(data_dict)

    drop_before_merge = [
        "Fiscal Period",
        "Currency",
        "Publish Date",
        "Restated Date",
        "Shares (Basic)",
        "Shares (Diluted)",
        "Ticker",
        "Report Date",
    ]

    for name, df in filtered_data.items():
        df = df.drop(columns=drop_before_merge, errors="ignore")
        filtered_data[name] = df
        try:
            df.to_csv(f"{data_dir}/{name}.csv", index=False)
        except OSError as exc:
            raise OSError(f"Could not write {name}.csv to {data_dir}") from exc

    combined_df = pd.DataFrame()

    for name, df in filtered_data.items():
        if name == "companies":
            continue
        missing_keys = [key for key in join_on if key not in df.columns]
        if missing_keys:
            raise ValueError(f"{name} is missing merge keys: {missing_keys}")
        if combined_df.empty:
            combined_df = df.copy()
            continue
        combined_df = pd.merge(combined_df, df, on=join_on, how="inner")
    if combined_df.empty:
        raise ValueError("Merged financial statements are empty")
    try:
        combined_df.to_csv(f"{data_dir}/combined_data.csv", index=False)
    except OSError as exc:
        raise OSError(f"Could not write combined_data.csv to {data_dir}") from exc


if __name__ == "__main__":
    main()
