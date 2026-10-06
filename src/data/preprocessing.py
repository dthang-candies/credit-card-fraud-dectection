"""
Data Preprocessing Module for Credit Card Fraud Detection.
Handles feature scaling, encoding, cleaning, and imbalance treatment.

CRITICAL ML BEST PRACTICES:
1. Featurization Ordering: Fit scalers/transformers ONLY on the training split,
   then transform validation and test splits independently.
2. Imbalance Resampling: Apply SMOTE/Under-sampling ONLY on the training set.
   Never resample validation or test sets to ensure realistic evaluation.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


# Determine project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class CreditCardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor tailored for the Credit Card Fraud dataset (Kaggle).
    - V1 to V28: Preserved as they are already PCA components.
    - Amount & Time: Scaled using RobustScaler (default) or StandardScaler.
    - Optional: Adds cyclical hour-of-day features from Time.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        scale_time: bool = True,
        add_cyclical_time: bool = False
    ):
        self.scaler_type = scaler_type
        self.scale_time = scale_time
        self.add_cyclical_time = add_cyclical_time
        self.scaler_amount = None
        self.scaler_time = None
        self.feature_names_: List[str] = []

    def _create_scaler(self):
        if self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "standard":
            return StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{self.scaler_type}'. Choose 'robust' or 'standard'.")

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()

        # Fit amount scaler
        if "Amount" in X_df.columns:
            self.scaler_amount = self._create_scaler()
            self.scaler_amount.fit(X_df[["Amount"]])

        # Fit time scaler if requested
        if self.scale_time and "Time" in X_df.columns:
            self.scaler_time = self._create_scaler()
            self.scaler_time.fit(X_df[["Time"]])

        # Record output feature names
        sample_transformed = self._transform_df(X_df.head(2))
        self.feature_names_ = list(sample_transformed.columns)

        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()

        if self.add_cyclical_time and "Time" in X_df.columns:
            # 3600 seconds per hour, 24 hours per day
            hours = (X_df["Time"] // 3600) % 24
            X_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            X_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        if self.scaler_amount is not None and "Amount" in X_df.columns:
            X_df["Amount"] = self.scaler_amount.transform(X_df[["Amount"]])

        if self.scaler_time is not None and "Time" in X_df.columns:
            X_df["Time"] = self.scaler_time.transform(X_df[["Time"]])

        return X_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_amount is None and "Amount" in X.columns:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
        return self._transform_df(X)


class BankSimPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor tailored for the BankSim dataset.
    - Drops 'Unnamed: 0' index column if present.
    - Scales 'amount' feature.
    - Retains binary category indicators and customer demographics.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.scaler_amount = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        if "Unnamed: 0" in X_df.columns:
            X_df = X_df.drop(columns=["Unnamed: 0"])

        if "amount" in X_df.columns:
            if self.scaler_type == "robust":
                self.scaler_amount = RobustScaler()
            else:
                self.scaler_amount = StandardScaler()
            self.scaler_amount.fit(X_df[["amount"]])

        sample_transformed = self._transform_df(X_df.head(2))
        self.feature_names_ = list(sample_transformed.columns)
        return self

    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        if "Unnamed: 0" in X_df.columns:
            X_df = X_df.drop(columns=["Unnamed: 0"])

        if self.scaler_amount is not None and "amount" in X_df.columns:
            X_df["amount"] = self.scaler_amount.transform(X_df[["amount"]])

        return X_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_amount is None and "amount" in X.columns:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")
        return self._transform_df(X)


def get_preprocessor(dataset_name: str, scaler_type: str = "robust") -> BaseEstimator:
    """
    Factory function to retrieve the appropriate preprocessor for a dataset.
    """
    ds = dataset_name.lower()
    if "credit" in ds:
        return CreditCardPreprocessor(scaler_type=scaler_type)
    elif "bank" in ds:
        return BankSimPreprocessor(scaler_type=scaler_type)
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Supported: 'creditcard', 'banksim'")


def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "none",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply resampling techniques to training data ONLY.

    Args:
        X_train: Training features.
        y_train: Training labels.
        method: Resampling method ('none', 'smote', 'undersample').
        random_state: Random state for reproducibility.

    Returns:
        Resampled (X_resampled, y_resampled).
    """
    if method == "none" or method is None:
        return X_train, y_train

    if not HAS_IMBLEARN:
        print("[Cảnh báo] Thư viện imbalanced-learn chưa được cài đặt. Bỏ qua bước cân bằng.")
        return X_train, y_train

    method_lower = method.lower()
    if method_lower == "smote":
        print(f"[ImbalanceHandler] Đang áp dụng SMOTE trên tập Train (trước: {dict(y_train.value_counts())})...")
        sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"[ImbalanceHandler] Hoàn tất SMOTE (sau: {dict(pd.Series(y_res).value_counts())}).")
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

    elif method_lower in ("undersample", "under"):
        print(f"[ImbalanceHandler] Đang áp dụng RandomUnderSampler trên tập Train (trước: {dict(y_train.value_counts())})...")
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(f"[ImbalanceHandler] Hoàn tất Under-sampling (sau: {dict(pd.Series(y_res).value_counts())}).")
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

    else:
        raise ValueError(f"Phương pháp không hỗ trợ '{method}'. Chọn 'none', 'smote', hoặc 'undersample'.")


def save_processed_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    preprocessor: Optional[BaseEstimator] = None,
    output_dir: Optional[Union[str, Path]] = None,
    dataset_name: str = "creditcard",
    save_format: str = "csv"
) -> Path:
    """
    Save processed splits and fitted preprocessor artifact to `data/processed/{dataset_name}/`.

    Args:
        X_train, y_train: Training partition.
        X_test, y_test: Test partition.
        X_val, y_val: Optional Validation partition.
        preprocessor: Fitted preprocessor object to save for inference.
        output_dir: Destination base folder. Defaults to `data/processed/`.
        dataset_name: Name of dataset subfolder.
        save_format: Format to save datasets ('csv' or 'npz').

    Returns:
        Path: The directory where artifacts are saved.
    """
    base_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR
    target_dir = base_dir / dataset_name.lower()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SaveProcessed] Đang lưu dữ liệu đã xử lý vào: {target_dir.relative_to(PROJECT_ROOT)}")

    target_name = y_train.name if y_train.name else "Class"

    if save_format == "csv":
        # Save feature sets
        X_train.to_csv(target_dir / "X_train.csv", index=False)
        y_train.to_csv(target_dir / "y_train.csv", index=False)
        X_test.to_csv(target_dir / "X_test.csv", index=False)
        y_test.to_csv(target_dir / "y_test.csv", index=False)

        # Also save full combined sets for convenience
        train_full = X_train.copy()
        train_full[target_name] = y_train.values
        train_full.to_csv(target_dir / "train.csv", index=False)

        test_full = X_test.copy()
        test_full[target_name] = y_test.values
        test_full.to_csv(target_dir / "test.csv", index=False)

        if X_val is not None and y_val is not None:
            X_val.to_csv(target_dir / "X_val.csv", index=False)
            y_val.to_csv(target_dir / "y_val.csv", index=False)
            val_full = X_val.copy()
            val_full[target_name] = y_val.values
            val_full.to_csv(target_dir / "val.csv", index=False)

    elif save_format == "npz":
        save_dict = {
            "X_train": X_train.values,
            "y_train": y_train.values,
            "X_test": X_test.values,
            "y_test": y_test.values,
        }
        if X_val is not None and y_val is not None:
            save_dict["X_val"] = X_val.values
            save_dict["y_val"] = y_val.values
        np.savez_compressed(target_dir / "data.npz", **save_dict)

    # Save preprocessor artifact for inference
    if preprocessor is not None:
        model_path = target_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, model_path)
        print(f"[SaveProcessed] Đã lưu pipeline preprocessor vào: {model_path.name}")

    print(f"[SaveProcessed] Hoàn tất lưu dữ liệu thành công tại: {target_dir}")
    return target_dir


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.spliter import train_val_test_split, print_split_summary

    print("Kiểm tra toàn diện quy trình Preprocessing trên BankSim:")
    df_raw = load_raw_data("banksim", verbose=False)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        df_raw, target_col="fraud", test_size=0.2, val_size=0.1
    )

    preproc = get_preprocessor("banksim", scaler_type="robust")
    X_tr_scaled = preproc.fit_transform(X_tr)
    X_va_scaled = preproc.transform(X_va)
    X_te_scaled = preproc.transform(X_te)

    print("Kích thước sau tiền xử lý:", X_tr_scaled.shape, X_va_scaled.shape, X_te_scaled.shape)
    save_dir = save_processed_data(
        X_tr_scaled, y_tr, X_te_scaled, y_te, X_va_scaled, y_va,
        preprocessor=preproc, dataset_name="banksim"
    )
