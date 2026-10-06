"""
Comprehensive Model Benchmark and Evaluation Script.
Replicates and compares all Machine Learning and Deep Learning models from the paper:
'Phát hiện gian lận thẻ tín dụng sử dụng mô hình học sâu Autoencoder kết hợp thuật toán Isolation forest'
(Ngo Thuy Linh & Nguyen Duong Hung, 2024).

Models compared:
- Logistic Regression (Scratch)
- Support Vector Machine (Linear SVM Scratch)
- K-Nearest Neighbors (KNN Scratch)
- Gaussian Naive Bayes (Scratch)
- Decision Tree (Scratch)
- Random Forest (Scratch)
- XGBoost (Scratch)
- Autoencoder (Scratch)
- Isolation Forest (Scratch - IF in Table 5)
- Autoencoder + Isolation Forest (Hybrid proposed model - AE+IF in Table 4 & Table 5)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml import (
    LogisticRegressionScratch,
    LinearSVMScratch,
    KNNScratch,
    GaussianNaiveBayesScratch,
    DecisionTreeClassifierScratch,
    RandomForestClassifierScratch,
    XGBoostClassifierScratch,
)
from src.dl import (
    AutoencoderScratch,
    IsolationForestScratch,
    AutoencoderIsolationForest,
)
from src.evaluation.metrics import (
    compute_metrics,
    compute_paper_style_metrics,
    print_metrics_report,
    compare_models,
)


def load_processed_splits(dataset_name: str = "banksim"):
    data_dir = PROJECT_ROOT / "data" / "processed" / dataset_name.lower()
    if not (data_dir / "X_train.csv").exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy dữ liệu đã xử lý tại {data_dir}. "
            f"Vui lòng chạy 'python -m src.data.run_pipeline --dataset {dataset_name}' trước."
        )

    X_train = pd.read_csv(data_dir / "X_train.csv").values
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze().values
    X_test = pd.read_csv(data_dir / "X_test.csv").values
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze().values

    return X_train, y_train, X_test, y_test


def evaluate_single_model(name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    print(f"\n---> Đang huấn luyện mô hình: {name}...")
    start_time = time.time()

    # Fit model
    if isinstance(model, (AutoencoderScratch, AutoencoderIsolationForest)):
        model.fit(X_train, y=y_train)
    else:
        model.fit(X_train, y_train)

    train_time = round(time.time() - start_time, 2)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None

    metrics = compute_metrics(y_test, y_pred, y_prob=y_prob)
    metrics["Train Time (s)"] = train_time

    print(f"[Xong: {train_time}s] Acc: {metrics['Accuracy']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1- Score']:.4f} | AROC: {metrics['AROC Score']:.4f}")
    return metrics


def run_all_benchmarks(dataset_name: str = "banksim", sample_limit: int = 50000):
    print("=" * 80)
    print(f"THỰC NGHIỆM ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN BỘ DỮ LIỆU: {dataset_name.upper()}")
    print("=" * 80)

    X_train, y_train, X_test, y_test = load_processed_splits(dataset_name)

    # Subsample training data if dataset is huge for fast benchmarking
    if len(X_train) > sample_limit:
        print(f"[Thông báo] Lấy mẫu đại diện {sample_limit:,} trên tổng số {len(X_train):,} mẫu train để tăng tốc...")
        rng = np.random.RandomState(42)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos = len(pos_idx)
        n_neg = sample_limit - n_pos
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        sub_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(sub_idx)
        X_train_sub, y_train_sub = X_train[sub_idx], y_train[sub_idx]
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # Contamination estimate
    fraud_rate = float(np.mean(y_train == 1))
    contamination = max(0.001, min(0.05, fraud_rate))

    # Initialize all models from scratch
    models = {
        "Logistic Regression": LogisticRegressionScratch(learning_rate=0.05, n_iterations=300),
        "Linear SVM": LinearSVMScratch(learning_rate=0.01, lambda_param=0.01, n_iterations=200),
        "KNN": KNNScratch(n_neighbors=5, max_train_samples=5000),
        "Naive Bayes": GaussianNaiveBayesScratch(),
        "Decision Tree": DecisionTreeClassifierScratch(max_depth=6, min_samples_split=10),
        "Random Forest": RandomForestClassifierScratch(n_estimators=20, max_depth=6),
        "XGBoost": XGBoostClassifierScratch(n_estimators=25, learning_rate=0.1, max_depth=4),
        "Autoencoder": AutoencoderScratch(hidden_dim=16, latent_dim=8, epochs=10, contamination=contamination),
        "Isolation Forest (IF)": IsolationForestScratch(n_estimators=50, max_samples=256, contamination=contamination),
        "Autoencoder + IF (Đề xuất)": AutoencoderIsolationForest(
            ae_hidden_dim=16,
            ae_latent_dim=8,
            ae_epochs=10,
            if_n_estimators=50,
            if_max_samples=256,
            contamination=contamination,
        ),
    }

    results = {}
    for name, model in models.items():
        try:
            metrics = evaluate_single_model(name, model, X_train_sub, y_train_sub, X_test, y_test)
            results[name] = metrics
        except Exception as e:
            print(f"[Lỗi] Mô hình {name} gặp lỗi: {e}")

    # Output comparison table
    title = f"BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH (DỮ LIỆU: {dataset_name.upper()})"
    df_results = compare_models(results, dataset_title=title)

    # Save results to results/
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_out = results_dir / f"benchmark_results_{dataset_name.lower()}.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"[Đã lưu kết quả thực nghiệm vào]: {csv_out}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm so sánh tất cả mô hình trong nghiên cứu.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="banksim",
        choices=["banksim", "creditcard", "all"],
        help="Chọn bộ dữ liệu thực nghiệm ('banksim', 'creditcard', hoặc 'all')",
    )
    args = parser.parse_args()

    datasets = ["banksim", "creditcard"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_all_benchmarks(ds)


if __name__ == "__main__":
    main()
