from pathlib import Path

import polars as pl


def load_dataset(df_path: str, name: str):
    """
    Load  datasets.
    Returns:
        df: as Polars DataFrames
    """

    # Verify files exist
    if not Path(df_path).exists():
        raise FileNotFoundError(
            f"{name} dataset not found at {df_path}. Run 'dvc pull' to download data."
        )

    # Load datasets with Polars
    df = pl.read_csv(df_path)

    print("Dataset loaded successfully")
    print(f"  - {name} dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df
