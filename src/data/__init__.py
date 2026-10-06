"""
Data processing package for credit card fraud detection.
Provides dataset loading, stratified/chronological splitting, and preprocessing pipelines.
"""

from src.data.loader import (
    load_raw_data,
    get_data_summary,
    PROJECT_ROOT,
    DEFAULT_RAW_DIR,
    DEFAULT_PROCESSED_DIR,
)

from src.data.spliter import (
    split_features_target,
    train_test_split_data,
    train_val_test_split,
    print_split_summary,
)

from src.data.preprocessing import (
    CreditCardPreprocessor,
    BankSimPreprocessor,
    get_preprocessor,
    handle_imbalance,
    save_processed_data,
)

__all__ = [
    "load_raw_data",
    "get_data_summary",
    "split_features_target",
    "train_test_split_data",
    "train_val_test_split",
    "print_split_summary",
    "CreditCardPreprocessor",
    "BankSimPreprocessor",
    "get_preprocessor",
    "handle_imbalance",
    "save_processed_data",
    "PROJECT_ROOT",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
]
