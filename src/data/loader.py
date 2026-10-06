"""
Data Loader Module for Credit Card Fraud Detection Project.
Handles loading raw datasets from `data/raw/` and basic inspections.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd


# Determine project root directory (two levels up from src/data)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(
    dataset: str = "creditcard",
    raw_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load a raw dataset by name or filename from `data/raw/`.

    Args:
        dataset: Dataset identifier ('creditcard', 'banksim') or filename ('creditcard.csv').
        raw_dir: Directory containing raw data files. Defaults to PROJECT_ROOT/data/raw.
        verbose: Whether to print summary info after loading.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    base_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR

    # Map dataset names to filenames
    filename_map = {
        "creditcard": "creditcard.csv",
        "banksim": "banksim.csv",
    }
    filename = filename_map.get(dataset.lower(), dataset)
    if not filename.endswith(".csv"):
        filename += ".csv"

    file_path = base_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. "
            f"Available files in {base_dir}: {[f.name for f in base_dir.glob('*.csv')]}"
        )

    if verbose:
        print(f"[DataLoader] Đang đọc dữ liệu từ: {file_path.relative_to(PROJECT_ROOT)}")

    df = pd.read_csv(file_path)

    if verbose:
        target_col = "Class" if "Class" in df.columns else ("fraud" if "fraud" in df.columns else None)
        print(f"[DataLoader] Đã tải thành công: {len(df):,} dòng, {df.shape[1]} cột.")
        if target_col:
            fraud_count = (df[target_col] == 1).sum()
            legit_count = (df[target_col] == 0).sum()
            fraud_pct = (fraud_count / len(df)) * 100
            print(
                f"[DataLoader] Phân bố nhãn '{target_col}': "
                f"Hợp lệ={legit_count:,} ({(100 - fraud_pct):.2f}%), "
                f"Gian lận={fraud_count:,} ({fraud_pct:.3f}%)"
            )

    return df


def get_data_summary(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute a statistical and structural summary of the dataframe.

    Args:
        df: Input DataFrame.
        target_col: Target column name for fraud class distribution.

    Returns:
        Dictionary containing summary statistics.
    """
    missing_count = int(df.isnull().sum().sum())
    duplicates_count = int(df.duplicated().sum())

    summary: Dict[str, Any] = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "missing_values": missing_count,
        "duplicate_rows": duplicates_count,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts().to_dict()
        total = len(df)
        summary["class_distribution"] = {
            cls: {"count": count, "ratio": round(count / total, 5)}
            for cls, count in class_counts.items()
        }

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM TRA MODULE LOADER VỚI CÁC BỘ DỮ LIỆU RAW")
    print("=" * 60)

    # Test loading creditcard dataset
    try:
        df_cc = load_raw_data("creditcard", verbose=True)
        summary_cc = get_data_summary(df_cc, target_col="Class")
        print("Tổng quan CreditCard:", summary_cc)
    except Exception as e:
        print("Lỗi đọc creditcard:", e)

    print("-" * 60)

    # Test loading banksim dataset
    try:
        df_bs = load_raw_data("banksim", verbose=True)
        summary_bs = get_data_summary(df_bs, target_col="fraud")
        print("Tổng quan BankSim:", summary_bs)
    except Exception as e:
        print("Lỗi đọc banksim:", e)
