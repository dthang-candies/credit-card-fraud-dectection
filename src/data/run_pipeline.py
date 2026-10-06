"""
Command-Line Runner for End-to-End Data Processing Pipeline.
Loads data from data/raw/, splits into Train/Val/Test, preprocesses features,
and exports results to data/processed/.

Usage examples:
    python -m src.data.run_pipeline --dataset creditcard
    python -m src.data.run_pipeline --dataset banksim
    python -m src.data.run_pipeline --dataset all --resample smote
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data, get_data_summary
from src.data.spliter import train_val_test_split, print_split_summary
from src.data.preprocessing import (
    get_preprocessor,
    handle_imbalance,
    save_processed_data,
)


def process_dataset(
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaler_type: str = "robust",
    resample_method: str = "none",
    drop_duplicates: bool = True,
    random_state: int = 42,
):
    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU XỬ LÝ DỮ LIỆU CHO BỘ DỮ LIỆU: {dataset_name.upper()}")
    print("=" * 70)

    # 1. Đọc dữ liệu thô
    df = load_raw_data(dataset_name, verbose=True)

    target_col = "Class" if "Class" in df.columns else ("fraud" if "fraud" in df.columns else None)
    if not target_col:
        raise ValueError(f"Không tìm thấy cột nhãn fraud/Class trong dữ liệu {dataset_name}")

    # 2. Xử lý trùng lặp nếu được yêu cầu
    if drop_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            print(f"[Tiền xử lý] Đã loại bỏ {dropped_count:,} dòng trùng lặp. Còn lại {len(df):,} dòng.")

    # 3. Phân chia Train / Val / Test (tuân thủ ML Best Practices: Tách trước khi chuẩn hóa)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df=df,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        stratify=True,
        random_state=random_state,
        shuffle=True,
    )
    print_split_summary(y_train, y_test, y_val=y_val, target_col=target_col)

    # 4. Fit Preprocessor chỉ trên Train, sau đó transform Val và Test
    preprocessor = get_preprocessor(dataset_name, scaler_type=scaler_type)
    print(f"[Tiền xử lý] Đang chuẩn hóa thuộc tính với {type(preprocessor).__name__} ({scaler_type})...")

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # 5. Cân bằng mẫu (chỉ áp dụng trên tập Train nếu có yêu cầu)
    if resample_method != "none":
        X_train_scaled, y_train = handle_imbalance(
            X_train_scaled, y_train, method=resample_method, random_state=random_state
        )

    # 6. Lưu dữ liệu đã xử lý vào data/processed/{dataset_name}/
    output_dir = save_processed_data(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        X_val=X_val_scaled,
        y_val=y_val,
        preprocessor=preprocessor,
        dataset_name=dataset_name,
        save_format="csv",
    )

    print(f"[Hoàn tất] Bộ dữ liệu {dataset_name} đã sẵn sàng tại {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu phát hiện gian lận thẻ tín dụng.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["creditcard", "banksim", "all"],
        help="Tên bộ dữ liệu cần xử lý ('creditcard', 'banksim', hoặc 'all')",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ tập Test (mặc định 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1, help="Tỉ lệ tập Validation (mặc định 0.1)")
    parser.add_argument(
        "--scaler",
        type=str,
        default="robust",
        choices=["robust", "standard"],
        help="Phương pháp chuẩn hóa số liệu ('robust' hoặc 'standard')",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="none",
        choices=["none", "smote", "undersample"],
        help="Kỹ thuật cân bằng dữ liệu trên tập Train ('none', 'smote', 'undersample')",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Giữ nguyên các bản ghi trùng lặp (mặc định sẽ loại bỏ)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho tái lập kết quả")

    args = parser.parse_args()

    datasets_to_process = (
        ["creditcard", "banksim"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets_to_process:
        process_dataset(
            dataset_name=ds,
            test_size=args.test_size,
            val_size=args.val_size,
            scaler_type=args.scaler,
            resample_method=args.resample,
            drop_duplicates=not args.keep_duplicates,
            random_state=args.seed,
        )


if __name__ == "__main__":
    main()
