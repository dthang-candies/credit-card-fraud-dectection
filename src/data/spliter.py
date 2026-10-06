"""
Data Splitter Module for Credit Card Fraud Detection Project.
Handles stratified or chronological splitting of datasets into Train, Validation, and Test sets.

CRITICAL ML BEST PRACTICE:
Always split the dataset into training, validation, and test sets BEFORE
fitting any preprocessing pipelines (e.g. scaling, imputation) to strictly
prevent data leakage.
"""

from typing import Tuple, Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target variable (y).

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y).
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into Train and Test sets.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        test_size: Proportion of test set (e.g., 0.2).
        stratify: Whether to use stratified split based on target.
        random_state: Seed for reproducibility.
        shuffle: Whether to shuffle data before splitting. Set False for chronological split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y = split_features_target(df, target_col=target_col)

    stratify_target = y if (stratify and shuffle) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state if shuffle else None,
        shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split dataset into 3 partitions: Train, Validation, and Test.

    Calculates proper internal proportions:
    e.g., test_size=0.2, val_size=0.1 means:
    - Test: 20% of original
    - Val: 10% of original
    - Train: 70% of original

    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        test_size: Ratio of test set relative to full dataset (default 0.2).
        val_size: Ratio of validation set relative to full dataset (default 0.1).
        stratify: Whether to stratify by target class.
        random_state: Random state for reproducibility.
        shuffle: Whether to shuffle data.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"Tổng test_size ({test_size}) + val_size ({val_size}) phải nhỏ hơn 1.0"
        )

    X, y = split_features_target(df, target_col=target_col)

    # 1. Tách Test set trước
    stratify_target = y if (stratify and shuffle) else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state if shuffle else None,
        shuffle=shuffle
    )

    # 2. Tách Validation set từ Train_Val
    # Tỉ lệ val trên phần còn lại (1.0 - test_size)
    relative_val_size = val_size / (1.0 - test_size)
    stratify_val = y_train_val if (stratify and shuffle) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=stratify_val,
        random_state=random_state if shuffle else None,
        shuffle=shuffle
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_summary(
    y_train: pd.Series,
    y_test: pd.Series,
    y_val: Optional[pd.Series] = None,
    target_col: str = "Class"
) -> None:
    """
    Print distribution and fraud ratio across splits.
    """
    def _stats(y_s, name):
        total = len(y_s)
        frauds = int((y_s == 1).sum())
        ratio = (frauds / total) * 100 if total > 0 else 0
        return f"{name:12}: {total:>7,} mẫu | Gian lận={frauds:>5,} ({ratio:>6.3f}%)"

    print("=" * 60)
    print(f"THỐNG KÊ PHÂN CHIA TẬP DỮ LIỆU (Target='{target_col}')")
    print("=" * 60)
    print(_stats(y_train, "Train Set"))
    if y_val is not None:
        print(_stats(y_val, "Val Set"))
    print(_stats(y_test, "Test Set"))
    print("=" * 60)


if __name__ == "__main__":
    from src.data.loader import load_raw_data

    print("Kiểm tra phân chia tập dữ liệu...")
    df = load_raw_data("banksim", verbose=False)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col="fraud", test_size=0.2, val_size=0.1, stratify=True
    )
    print_split_summary(y_train, y_test, y_val=y_val, target_col="fraud")
