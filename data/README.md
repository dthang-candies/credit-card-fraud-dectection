# Dữ Liệu Phát Hiện Gian Lận Thẻ Tín Dụng

Thư mục này chứa dữ liệu thô (`raw/`) và dữ liệu đã qua tiền xử lý (`processed/`).

## 1. Cấu trúc thư mục

```
data/
├── raw/
│   ├── creditcard.csv       # Bộ dữ liệu Kaggle Credit Card Fraud (284,807 giao dịch)
│   └── banksim.csv          # Bộ dữ liệu mô phỏng BankSim (7,189 giao dịch)
└── processed/
    ├── creditcard/          # Đã chuẩn hóa & phân chia Train/Val/Test
    │   ├── train.csv, val.csv, test.csv
    │   ├── X_train.csv, y_train.csv
    │   ├── X_val.csv, y_val.csv
    │   ├── X_test.csv, y_test.csv
    │   └── preprocessor.joblib
    └── banksim/
        ├── train.csv, val.csv, test.csv
        ├── X_train.csv, y_train.csv
        ├── X_val.csv, y_val.csv
        ├── X_test.csv, y_test.csv
        └── preprocessor.joblib
```

## 2. Đặc điểm dữ liệu & quy trình xử lý

### Bộ dữ liệu `creditcard.csv`
- **Số lượng**: 284,807 dòng, 31 cột.
- **Nhãn mục tiêu**: `Class` (0 = Hợp lệ: 99.83%, 1 = Gian lận: 0.17%).
- **Xử lý**:
  - Loại bỏ các dòng trùng lặp (1,081 dòng).
  - Phân tầng (Stratified split) tỷ lệ 70% Train, 10% Val, 20% Test để bảo toàn tỷ lệ gian lận hiếm.
  - Chuẩn hóa `Amount` và `Time` bằng `RobustScaler` (fit trên tập Train để chống data leakage).
  - Giữ nguyên các đặc trưng PCA `V1` đến `V28`.

### Bộ dữ liệu `banksim.csv`
- **Số lượng**: 7,189 dòng, 19 cột.
- **Nhãn mục tiêu**: `fraud` (0 = Hợp lệ: 97.22%, 1 = Gian lận: 2.78%).
- **Xử lý**:
  - Loại bỏ cột index không cần thiết (`Unnamed: 0`).
  - Phân tầng (Stratified split) tỷ lệ 70% Train, 10% Val, 20% Test.
  - Chuẩn hóa cột `amount` bằng `RobustScaler`.
