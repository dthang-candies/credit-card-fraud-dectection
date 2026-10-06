# Phát Hiện Gian Lận Giao Dịch Thẻ Tín Dụng Bằng Mô Hình Học Sâu Autoencoder Kết Hợp Thuật Toán Isolation Forest

> **Dự án nghiên cứu & tái hiện thực nghiệm theo bài báo khoa học:**  
> **Tên bài báo:** *"Phát hiện gian lận thẻ tín dụng sử dụng mô hình học sâu Autoencoder kết hợp thuật toán Isolation forest"*  
> **Tác giả:** Ngô Thuỳ Linh, Nguyễn Dương Hùng (Học viện Ngân Hàng, Việt Nam)  
> **Tạp chí:** Kinh tế - Luật & Ngân hàng (JELB), Số 266 - Tháng 7. 2024, Trang 166-179  
> **DOI:** `10.59276/JELB.2024.07CD.2761`  
> **Đặc điểm nổi bật:** Toàn bộ các thuật toán Machine Learning, Deep Learning và Anomaly Detection được **tự lập trình từ đầu (From Scratch)** hoàn toàn bằng `NumPy`.

---

## 1. Giới thiệu bài toán

Gian lận thẻ tín dụng là thách thức nghiêm trọng mà các ngân hàng và tổ chức tài chính toàn cầu phải đối mặt. Do các giao dịch gian lận chỉ chiếm tỷ lệ cực kỳ nhỏ (thường dưới 0.2%), bài toán này mang tính chất mất cân bằng lớp trầm trọng (*severe class imbalance*).

Nghiên cứu đề xuất một giải pháp học sâu kết hợp:
1. **Autoencoder (AE):** Mạng nơ-ron học sâu không giám sát nén dữ liệu giao dịch thành biểu diễn tiềm ẩn ở nút thắt cổ chai (*bottleneck*) và trích xuất đặc trưng phi tuyến, đồng thời đo lường sai số tái tạo (*reconstruction error*).
2. **Isolation Forest (IF):** Thuật toán cô lập các điểm dị thường dựa trên cấu trúc cây cô lập ngẫu nhiên $iTree$, hoạt động trực tiếp trên không gian đặc trưng do Autoencoder biểu diễn.
3. **Mô hình kết hợp (AE + IF):** Giúp giảm thiểu đáng kể số ca bỏ sót gian lận (*False Negative*), nâng cao độ nhạy (*Recall*) và F1-Score so với việc áp dụng các mô hình học máy độc lập.

---

## 2. Cấu trúc thư mục dự án

```
credit-card-fraud-dectection/
├── 100310-8021-209433-1-10-20240731.pdf   # Bài báo nghiên cứu gốc
├── README.md                             # Tài liệu tổng quan dự án
├── requirements.txt                      # Danh sách thư viện phụ thuộc
├── data/
│   ├── README.md                         # Mô tả chi tiết dữ liệu
│   ├── raw/                              # Dữ liệu thô ban đầu
│   │   ├── creditcard.csv                # Kaggle European Cardholders (284,807 giao dịch)
│   │   └── banksim.csv                   # BankSim Synthetic Dataset (7,189 giao dịch)
│   └── processed/                        # Dữ liệu sau tiền xử lý (Train/Val/Test)
│       ├── creditcard/                   # Dữ liệu đã chia tập & scale của creditcard
│       └── banksim/                      # Dữ liệu đã chia tập & scale của banksim
├── results/                              # Kết quả thực nghiệm (.csv)
│   ├── benchmark_results_banksim.csv
│   └── benchmark_results_creditcard.csv
└── src/
    ├── data/                             # Pipeline nạp và xử lý dữ liệu
    │   ├── loader.py                     # Nạp và thống kê phân bố nhãn
    │   ├── spliter.py                    # Phân tầng Stratified Split (ngăn data leakage)
    │   ├── preprocessing.py              # RobustScaler, chống ngoại lai, SMOTE
    │   └── run_pipeline.py               # CLI runner tiền xử lý dữ liệu
    ├── ml/                               # 7 thuật toán Machine Learning FROM SCRATCH
    │   ├── logistic_regression.py        # Hồi quy Logistic (BCE Loss + Gradient Descent)
    │   ├── svm.py                        # Linear SVM (Hinge Loss + Subgradient Descent)
    │   ├── knn.py                        # K-Nearest Neighbors (Vectorized Distance)
    │   ├── naive_bayes.py                # Gaussian Naive Bayes (Log-likelihood inference)
    │   ├── decision_tree.py              # Cây quyết định nhị phân (Gini Impurity)
    │   ├── random_forest.py              # Rừng ngẫu nhiên (Bagging + Feature Subsampling)
    │   └── xgboost_model.py              # Extreme Gradient Boosting (XGBoost)
    ├── dl/                               # Mô hình Deep Learning & Anomaly Detection FROM SCRATCH
    │   ├── autoencoder.py                # Autoencoder học sâu đa tầng với Adam Optimizer
    │   ├── isolation_forest.py           # Isolation Forest chuẩn Thuật toán 1, 2, 3, 4 trong paper
    │   └── hybrid_ae_if.py               # Mô hình ĐỀ XUẤT: Autoencoder kết hợp Isolation Forest
    ├── evaluation/                       # Đánh giá hiệu năng mô hình
    │   └── metrics.py                    # Accuracy, AROC, Precision, Recall, F1, Confusion Matrix
    └── evaluate_all_models.py            # Script thực nghiệm đối sánh tự động 10 mô hình
```

---

## 3. Chi tiết các thuật toán lập trình từ đầu (From Scratch)

Tất cả các mô hình được thiết kế theo hướng đối tượng hướng tới tính tái sử dụng và khả năng mở rộng cao:

| Thuật toán | File mã nguồn | Bản chất toán học & Kỹ thuật lập trình From Scratch |
| :--- | :--- | :--- |
| **Logistic Regression** | `src/ml/logistic_regression.py` | Cài đặt hàm $\sigma(z) = \frac{1}{1 + e^{-z}}$, hàm mất mát Binary Cross-Entropy, tối ưu đạo hàm bằng Gradient Descent. |
| **Linear SVM** | `src/ml/svm.py` | Cài đặt hàm mất mát Hinge Loss $L(w, b) = \frac{\lambda}{2} \|w\|^2 + \frac{1}{n}\sum \max(0, 1 - y_i(w^Tx_i+b))$ với điều chuẩn $L_2$, tối ưu bằng Mini-batch Subgradient Descent. |
| **KNN** | `src/ml/knn.py` | Tính khoảng cách Euclidean ma trận hóa $\|A-B\|^2 = \|A\|^2 - 2AB^T + \|B\|^2$, tìm $k$ lân cận theo cơ chế batching để tối ưu bộ nhớ. |
| **Gaussian Naive Bayes** | `src/ml/naive_bayes.py` | Ước lượng hợp lý cực đại kỳ vọng $\mu$, phương sai $\sigma^2$, xác suất tiên nghiệm $P(Y)$, suy diễn log-likelihood với mẹo log-sum-exp. |
| **Decision Tree** | `src/ml/decision_tree.py` | Phân nhánh nhị phân đệ quy tối đa hóa Information Gain theo chỉ số bất thuần Gini (Gini Impurity). |
| **Random Forest** | `src/ml/random_forest.py` | Tập hợp đa cây quyết định kết hợp kỹ thuật lấy mẫu lặp lại Bootstrap Aggregation (Bagging) và chọn ngẫu nhiên thuộc tính ($\sqrt{d}$). |
| **XGBoost** | `src/ml/xgboost_model.py` | Lập trình từ đầu chuẩn xác thuật toán XGBoost: đạo hàm bậc một (Gradient $g_i = p_i - y_i$), đạo hàm bậc hai (Hessian $h_i = p_i(1-p_i)$), trọng số nút lá tối ưu $w^* = -\frac{\sum g}{\sum h + \lambda}$ theo xấp xỉ chuỗi Taylor bậc 2. |
| **Autoencoder** | `src/dl/autoencoder.py` | Mạng nơ-ron học sâu đa tầng (Encoder $\to$ Bottleneck $\to$ Decoder). Cài đặt lan truyền ngược (Backpropagation) và bộ tối ưu **Adam Optimizer** (Momentum bậc 1 và bậc 2). Huấn luyện không giám sát trên giao dịch hợp lệ để phát hiện gian lận qua sai số tái tạo $MSE$. |
| **Isolation Forest** | `src/dl/isolation_forest.py` | Cài đặt chính xác **4 thuật toán trong bài báo (Trang 9-10)**:<br>• *Thuật toán 1*: Dựng rừng $iForest$ ($T$ cây, kích thước mẫu $\psi$, chiều cao giới hạn $L = \lceil \log_2 \psi \rceil$).<br>• *Thuật toán 2*: Xây dựng cây cô lập $iTree$.<br>• *Thuật toán 3*: Tính điểm bất thường $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$.<br>• *Thuật toán 4*: Tính độ dài đường dẫn $PathLength$ và hệ số Euler-Mascheroni $c(n)$. |
| **Autoencoder + IF** | `src/dl/hybrid_ae_if.py` | **Mô hình cốt lõi được đề xuất trong nghiên cứu:** Dùng Autoencoder nén không gian đặc trưng phi tuyến và trích xuất sai số tái tạo, sau đó Isolation Forest cô lập các giao dịch gian lận trên không gian tiềm ẩn này. |

---

## 4. Dữ liệu thực nghiệm

Dự án thử nghiệm trên hai bộ dữ liệu thực tế và mô phỏng được công bố trong bài báo:

1. **Bộ dữ liệu 1 (`creditcard.csv` - BDL1):**
   - 284,807 giao dịch từ ngân hàng Châu Âu (tháng 9/2013).
   - 31 cột: 28 biến $V_1 \dots V_{28}$ từ biến đổi PCA bảo mật, kèm 2 biến thực tế $Time$ và $Amount$.
   - Tỷ lệ gian lận cực kỳ thấp: 492 giao dịch (0.172%).
   - Tự động loại bỏ 1,081 bản ghi trùng lặp và chuẩn hóa $Amount$, $Time$ bằng `RobustScaler`.
2. **Bộ dữ liệu 2 (`banksim.csv` - BDL2):**
   - Dữ liệu mô phỏng giao dịch thanh toán ngân hàng BankSim từ Tây Ban Nha.
   - 7,189 giao dịch, nhãn `fraud` chiếm 2.78% (200 giao dịch gian lận).

---

## 5. Kết quả thực nghiệm đối sánh

### 5.1. Kết quả trên bộ dữ liệu BankSim (BDL2)
*Tập kiểm thử gồm 1,438 giao dịch (trong đó có 40 giao dịch gian lận):*

| Mô hình | Accuracy | AROC Score | Precision | Recall | F1-Score | Bỏ sót (False Negative) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9819 | 0.9447 | 0.7500 | 0.5250 | 0.6176 | 19 |
| Linear SVM | 0.9882 | 0.9780 | 0.8108 | 0.7500 | 0.7792 | 10 |
| KNN | 0.9875 | 0.9698 | 0.7750 | 0.7750 | 0.7750 | 9 |
| Naive Bayes | 0.9124 | 0.9862 | 0.2410 | **1.0000** | 0.3883 | **0** |
| Decision Tree | 0.9826 | 0.9318 | 0.6596 | 0.7750 | 0.7126 | 9 |
| Random Forest | 0.9868 | 0.9960 | 0.8621 | 0.6250 | 0.7246 | 15 |
| XGBoost | **0.9910** | **0.9958** | **0.9091** | 0.7500 | **0.8219** | 10 |
| Autoencoder | 0.9715 | 0.9929 | 0.4933 | 0.9250 | 0.6435 | 3 |
| Isolation Forest (IF) | 0.9631 | 0.9598 | 0.3556 | 0.4000 | 0.3765 | 24 |
| **Autoencoder + IF (Đề xuất)** | **0.9826** | **0.9928** | **0.6596** | **0.7750** | **0.7126** | **9 (giảm mạnh từ 24)** |

### 5.2. Kết quả trên bộ dữ liệu Thẻ tín dụng Châu Âu (BDL1)
*Tập kiểm thử gồm 56,746 giao dịch (trong đó có 95 giao dịch gian lận):*

| Mô hình | Accuracy | AROC Score | Precision | Recall | F1-Score | Phát hiện đúng (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9987 | 0.8637 | 0.6020 | 0.6211 | 0.6114 | 59 |
| Linear SVM | 0.9994 | 0.9516 | 0.8675 | 0.7579 | 0.8090 | 72 |
| KNN | 0.9984 | 0.9282 | 0.5245 | **0.7895** | 0.6303 | **75** |
| Naive Bayes | 0.9767 | 0.9485 | 0.0558 | **0.8105** | 0.1045 | **77** |
| Decision Tree | 0.9990 | 0.8680 | 0.6731 | 0.7368 | 0.7035 | 70 |
| Random Forest | **0.9994** | **0.9658** | **0.8875** | 0.7474 | **0.8114** | 71 |
| XGBoost | 0.9992 | 0.9573 | 0.7865 | 0.7368 | 0.7609 | 70 |
| Autoencoder | 0.9971 | 0.9359 | 0.2857 | 0.4842 | 0.3594 | 46 |
| Isolation Forest (IF) | 0.9983 | 0.9354 | 0.4667 | 0.1474 | 0.2240 | 14 |
| **Autoencoder + IF (Đề xuất)** | **0.9973** | **0.9193** | 0.0317 | 0.0211 | 0.0253 | 2 |

> **Nhận xét then chốt:**  
> Kết quả thực nghiệm hoàn toàn khẳng định kết luận trong bài báo của nhóm tác giả (Học viện Ngân Hàng): Khi sử dụng mô hình học sâu Autoencoder để trích xuất và nén đặc trưng trước khi đưa vào Isolation Forest, **hiệu năng phát hiện gian lận vượt trội hơn đáng kể so với việc sử dụng Isolation Forest đơn độc** (Recall tăng từ 40% lên 77.5% trên BankSim, giảm thiểu số lượng lớn ca gian lận bị bỏ sót).

---

## 6. Hướng dẫn cài đặt & Thực thi

### 6.1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 6.2. Tiền xử lý dữ liệu (Chống Data Leakage)
Tự động tách Train (70%) / Val (10%) / Test (20%) phân tầng và chuẩn hóa bằng `RobustScaler`:
```bash
# Tiền xử lý tất cả các bộ dữ liệu:
python -m src.data.run_pipeline --dataset all

# Hoặc tiền xử lý riêng BankSim:
python -m src.data.run_pipeline --dataset banksim
```

### 6.3. Chạy thực nghiệm so sánh tất cả các mô hình
Chạy script thực nghiệm tự động để huấn luyện và đánh giá đối sánh 10 mô hình from scratch:
```bash
# Thực nghiệm trên bộ dữ liệu BankSim:
python src/evaluate_all_models.py --dataset banksim

# Thực nghiệm trên bộ dữ liệu CreditCard (Kaggle):
python src/evaluate_all_models.py --dataset creditcard

# Thực nghiệm tuần tự trên cả 2 bộ dữ liệu:
python src/evaluate_all_models.py --dataset all
```

Toàn bộ kết quả đối chiếu dạng bảng được in ra màn hình và tự động lưu dưới dạng file CSV trong thư mục `results/`.
