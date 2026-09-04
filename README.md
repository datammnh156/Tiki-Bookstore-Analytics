# Phân Tích Dữ Liệu và Xây Dựng Mô Hình Dự Đoán Sách Bán Chạy của Nhà Sách Tiki Sử Dụng Machine Learning

**Đồ án ngành Hệ thống Thông tin Quản lý**

---

## Giới Thiệu

Đồ án này phân tích dữ liệu sách từ Tiki.vn để xây dựng mô hình dự đoán khả năng một đầu sách thuộc nhóm bán chạy (Bestseller). Project cung cấp các chức năng hỗ trợ quyết định như mô phỏng mức giảm giá và hệ thống gợi ý sách.

---

## Mục Tiêu

1. Dự đoán Bestseller bằng Machine Learning (Random Forest, Logistic Regression)
2. Giải thích mô hình với Feature Importance và SHAP
3. Mô phỏng mức giảm giá dựa trên xác suất dự đoán
4. Xây dựng hệ thống gợi ý sách (Content-Based Filtering)
5. Trực quan hóa insights với Power BI

---

## Dataset

- **Nguồn**: Tiki.vn (thu thập thông qua Tiki API bằng Python)
- **Số lượng**: 4.005 đầu sách
- **Danh mục**: 9 thể loại sách
- **Trạng thái**: Đã làm sạch, loại bỏ trùng lặp và thiếu dữ liệu

### Định Nghĩa Nhãn Bestseller

Nhãn `is_bestseller` được xây dựng trong đồ án:
- Nhóm sách theo `category_name`
- Tính phân vị 70% của `quantity_sold` trong từng danh mục
- Sách có `quantity_sold` ≥ ngưỡng được gán `is_bestseller = 1`
- Khoảng 30% sách bán tốt nhất mỗi danh mục

**Phân bố**: 1.206 Bestseller (30,1%) | 2.799 Non-Bestseller (69,9%)

---

## Machine Learning

### Đặc Trưng

| Đặc Trưng | Mô Tả |
|-----------|-------|
| `price` | Giá sách (VND) |
| `discount_rate` | Mức giảm giá (%) |
| `rating_average` | Điểm đánh giá trung bình (0-5) |
| `has_rating` | Có đánh giá (0/1) |
| `category_name` | Danh mục sách (One-Hot Encoded) |

**Lưu ý**: `quantity_sold` KHÔNG được dùng làm feature (dùng để xây dựng target)

### Mô Hình

**Logistic Regression**: Accuracy 70,3%, Precision 51,6%, Recall 19,9%, F1 28,7%

**Random Forest (Mô Hình Chính)**:

| Chỉ Số | Giá Trị |
|--------|--------|
| Train Accuracy | 82,0% |
| Test Accuracy | 75,4% |
| Precision (Bestseller) | 65,3% |
| Recall (Bestseller) | 39,0% |
| F1-Score (Bestseller) | 48,8% |
| 5-Fold CV | 75,1% ± 1,4% |

---

## Giải Thích Mô Hình

- **Feature Importance**: Từ Random Forest, dựa trên tầm quan trọng trong phân chia cây
- **SHAP Analysis**: TreeExplainer, Summary Plot, Waterfall Plot để giải thích dự đoán cụ thể

**Lưu ý**: SHAP giải thích những gì mô hình học được, không chứng minh quan hệ nhân quả

---

## Mô Phỏng Mức Giảm Giá

### Phương Pháp

- Quét `discount_rate` từ 0% đến 50% (bước 5%)
- Giữ nguyên các feature khác
- Sử dụng `model.predict_proba()` để tính xác suất

### Tiêu Chí Đề Xuất

```
Đề xuất thay đổi nếu: candidate_probability - current_probability >= 0.05
```

### Kết Quả

- **Tổng sách mô phỏng**: 4.005
- **Được đề xuất thay đổi**: 1.648 (41,1%)
- **Giữ nguyên**: 2.357 (58,9%)
- **Cải thiện xác suất dự đoán trung bình**: 13,6 điểm phần trăm
- **Cải thiện cao nhất**: 44,1 điểm phần trăm

**Lưu ý quan trọng**: Tương quan, KHÔNG phải nhân quả. Chỉ mang tính tham khảo, cần A/B test để xác minh.

---

## Hệ Thống Gợi Ý Sách

### Phương Pháp (Content-Based Filtering)

**1. TF-IDF Similarity (70%)**
- Dùng tên sách làm dữ liệu văn bản
- TF-IDF vectorization
- Cosine Similarity

**2. Tương Đồng Đặc Trưng Số (30% trọng số)**
- Chuẩn hóa đặc trưng: price, rating_average, discount_rate, quantity_sold, review_count, has_rating, is_bestseller
- Tính Euclidean Distance
- Chuyển đổi thành similarity: `1 / (1 + distance)`

**3. Kết hợp điểm tương đồng**
```
final_similarity = 0.7 * tfidf_similarity + 0.3 * numeric_similarity
```

**4. Lọc**: Chỉ gợi ý trong cùng danh mục, Top 5 mỗi sách

### Kết Quả

- **Tổng gợi ý**: 20.025
- **High (≥0.7)**: 3.702 (18,5%)
- **Medium (0.5-0.7)**: 5.171 (25,8%)
- **Low (<0.5)**: 11.152 (55,7%)

**Lưu ý**: High/Medium/Low là mức độ tương đồng dựa trên similarity score, không phải accuracy hay xác suất người dùng thích sách.

---

## Power BI

Dashboard: `DoAnNganhPBI.pbix`

Trình bày: Tổng quan, phân tích giá, đánh giá, insights Bestseller

---

## Yêu Cầu Hệ Thống

- **Python**: 3.9 trở lên
- **Package Manager**: pip hoặc Anaconda
- **SQL Server** (tùy chọn): Chỉ cần nếu muốn load dữ liệu vào SQL và dùng Power BI
- **Power BI Desktop** (tùy chọn): Để xem dashboard

---

## Hướng Dẫn Cài Đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/datammnh156/Tiki-Bookstore-Analytics.git
cd DoAnNganh
```

### Bước 2: Tạo Virtual Environment (Khuyến Nghị)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Xác Minh Cài Đặt

```bash
python -c "import pandas, numpy, sklearn, shap; print('✓ Cài đặt thành công!')"
```

---

## Cấu Trúc Project

```
DoAnNganh/
├── README.md
├── requirements.txt
├── DoAnNganhPBI.pbix
├── data/clean/
│   ├── tiki_books_cleaned.csv
│   ├── discount_recommendations.csv
│   ├── book_recommendations.csv
│   └── shap_values.csv
├── models/
│   ├── bestseller_model.pkl
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   └── logistic_scaler.pkl
├── scripts/
│   ├── cleaning/
│   ├── crawl/
│   ├── eda/
│   ├── ml/
│   ├── analysis/
│   └── etl/
└── outputs/
    ├── charts/
    └── reports/
```

---

## Công Nghệ Sử Dụng

**Xử lý dữ liệu**: Python, pandas, NumPy

**Machine Learning**: scikit-learn, SHAP

**Visualization**: Matplotlib, Seaborn, Power BI

**Data Collection**: Python (urllib, json) - Thu thập dữ liệu qua Tiki API

**Storage**: SQL Server, pickle

**Version Control**: Git/GitHub

---

## Hướng Dẫn Chạy

### Chạy Nhanh (KHÔNG cần SQL Server)

Project có sẵn dữ liệu đã làm sạch và model đã train. Có thể chạy ngay:

```bash
# SHAP Analysis
python scripts/ml/07.shap_analysis.py

# Discount Optimization
python scripts/ml/06_optimize_discount_all.py

# Recommendation System
python scripts/ml/08.recommendation.py
```

### 🔄 Chạy Đầy Đủ (Từ Đầu)

Nếu muốn chạy toàn bộ pipeline từ crawl → train → analysis:

```bash
# 1. Thu thập dữ liệu (có thể bỏ qua vì đã có data)
python scripts/crawl/crawl_data_tiki.py

# 2. Làm sạch dữ liệu
python scripts/cleaning/cleaned_data.py

# 3. Phân tích khám phá
python scripts/eda/descriptive_analysis.py

# 4. Huấn luyện mô hình
python scripts/ml/01_train_random_forest.py
python scripts/ml/02_train_logistic_regression.py

# 5. Đánh giá và so sánh
python scripts/ml/03_model_comparison.py
python scripts/ml/04_random_forest_evaluation.py

# 6. Giải thích mô hình
python scripts/ml/07.shap_analysis.py

# 7. Các chức năng hỗ trợ
python scripts/ml/06_optimize_discount_all.py
python scripts/ml/08.recommendation.py
```

### 📊 SQL Server và Power BI (Tùy chọn)

Các chức năng Machine Learning, SHAP, mô phỏng mức giảm giá và gợi ý sách có thể sử dụng dữ liệu đã có trong repository mà không cần SQL Server.

Nếu muốn chạy các script ETL để load dữ liệu vào SQL Server, cần:
- SQL Server đã được cài đặt và đang hoạt động
- Microsoft ODBC Driver for SQL Server
- Cấu hình thông tin kết nối trong script ETL phù hợp với máy đang sử dụng

Sau đó có thể chạy:
```bash
python scripts/etl/load_all_tables.py
```

Dashboard Power BI được lưu trong: `DoAnNganhPBI.pbix`

---

## Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'xxx'

**Nguyên nhân**: Thiếu thư viện

**Giải pháp**:
```bash
pip install -r requirements.txt
```

### Lỗi: FileNotFoundError: [Errno 2] No such file or directory

**Nguyên nhân**: Đang chạy script ở sai thư mục

**Giải pháp**: Đảm bảo đang ở thư mục gốc `DoAnNganh/`
```bash
cd d:\DoAnNganh
python scripts/ml/07.shap_analysis.py
```

### Lỗi: Permission denied

**Nguyên nhân**: Không có quyền ghi file

**Giải pháp**: Chạy terminal/PowerShell với quyền Administrator

### Lỗi: ImportError: DLL load failed (Windows)

**Nguyên nhân**: Thiếu Visual C++ Redistributable

**Giải pháp**: Tải và cài đặt [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

## Kết Quả Chính

- **Random Forest**: 75,4% Test Accuracy, 48,8% F1-Score
- **Discount Optimization**: 41,1% sách được đề xuất, cải thiện 13,6 điểm phần trăm
- **Recommendation**: 20.025 gợi ý, 18,5% lượt gợi ý thuộc mức High
- **Top Features**: rating_average, price, discount_rate, has_rating

---

## Hạn Chế

### 1. Nhãn Bestseller
- Xây dựng từ phân vị 70%, KHÔNG phải nhãn chính thức Tiki

### 2. Discount Simulation
- Tương quan, KHÔNG phải nhân quả
- Chỉ tham khảo, cần A/B test

### 3. Recommendation
- Dựa trên nội dung, KHÔNG sử dụng hành vi người dùng

### 4. Dữ Liệu
- Snapshot tại thời điểm crawl, chỉ từ Tiki.vn

---

**Cập nhật lần cuối**: 04/09/2026
