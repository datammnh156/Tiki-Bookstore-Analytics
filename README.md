# 📚 Đồ án phân tích dữ liệu và dự đoán sách bestseller trên nền tảng Tiki.vn sử dụng Machine Learning và Power BI.
Đồ Án Ngành Hệ Thống Thông Tin Quản Lý
---


---

## 📋 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Tính Năng](#tính-năng)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Kết Quả](#kết-quả)
- [Công Nghệ](#công-nghệ)
- [Tác Giả](#tác-giả)

---

## 🎯 Giới Thiệu

Dự án này phân tích **3,830 sách** từ Tiki.vn với các mục tiêu:

1. **Dự đoán Bestseller**: Sử dụng Random Forest và Logistic Regression
2. **Content-Based Recommendation**: Gợi ý sách tương tự dựa trên giá và rating
3. **SHAP Analysis**: Giải thích model predictions
4. **Power BI Dashboard**: Visualization và insights

---

## ✨ Tính Năng

### 1. Machine Learning Models
- ✅ Random Forest Classifier (Primary model)
- ✅ Logistic Regression (Comparison model)
- ✅ Model evaluation và comparison
- ✅ GridSearch hyperparameter tuning

### 2. Content-Based Recommendation
- ✅ Euclidean distance-based similarity
- ✅ Top 5 recommendations per book
- ✅ Category-filtered recommendations

### 3. SHAP Explainability
- ✅ Feature importance analysis
- ✅ SHAP values export for Power BI
- ✅ Model interpretation

### 4. Discount Optimization
- ✅ Optimize discount strategy per book
- ✅ Batch optimization for all books
- ⚠️ Mô phỏng dựa trên correlational analysis (xem phần Limitations)

### 5. Power BI Dashboard
- ✅ Descriptive analysis
- ✅ Predictive insights
- ✅ Interactive visualizations

---

## 📁 Cấu Trúc Thư Mục

```
DoAnNganh/
├── 📄 README.md                           # Tài liệu này
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # Git ignore rules
├── 📄 DoAnNganhPBI.pbix                   # Power BI report
│
├── 📁 data/
│   ├── raw/                               # Dữ liệu gốc
│   │   └── tiki_books.csv
│   └── clean/                             # Dữ liệu đã xử lý
│       ├── tiki_books_cleaned.csv         # Dataset chính
│       ├── book_recommendations.csv       # Recommendations
│       └── shap_values.csv                # SHAP analysis
│
├── 📁 models/
│   ├── bestseller_model.pkl               # Random Forest model
│   └── scaler.pkl                         # StandardScaler
│
├── 📁 scripts/
│   ├── cleaning/                          # Data cleaning scripts
│   ├── crawl/                             # Web scraping scripts
│   ├── eda/                               # Exploratory analysis
│   └── ml/                                # Machine learning scripts
│       ├── 01_train_random_forest.py      # Train RF model
│       ├── 02_train_logistic_regression.py # Train LR model
│       ├── 03_model_comparison.py         # Compare models
│       ├── 04_random_forest_evaluation.py # Evaluate RF
│       ├── 05_optimize_discount_single.py # Single book optimization
│       ├── 06_optimize_discount_all.py    # Batch optimization
│       ├── shap_analysis.py               # Shap analysis values
│       ├── export_shap_for_powerbi.py     # Export SHAP values
│       └── recommendation.py              # Recommendation system
│
└── 📁 outputs/
    ├── charts/                            # EDA charts
    └── reports/                           # Analysis reports
```

---

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- Power BI Desktop (cho dashboard)

### Bước 1: Clone Repository

```bash
git clone https://github.com/datammnh156/Do-An-Nganh-HTTTQL.git
cd Do-An-Nganh-HTTTQL
```

### Bước 2: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn Bị Dữ Liệu

Dữ liệu đã được làm sạch và sẵn sàng trong `data/clean/tiki_books_cleaned.csv`

---

## 💻 Sử Dụng

### 1. Train Machine Learning Model

```bash
# Train Random Forest model
python scripts/ml/01_train_random_forest.py

# Train Logistic Regression model
python scripts/ml/02_train_logistic_regression.py

# So sánh 2 models
python scripts/ml/03_model_comparison.py
```

### 2. Tạo Recommendations

```bash
python scripts/ml/recommendation.py
```

Output: `data/clean/book_recommendations.csv`

### 3. Export SHAP Values cho Power BI

```bash
python scripts/ml/export_shap_for_powerbi.py
```

Output: `data/clean/shap_values.csv`

### 4. Optimize Discount Strategy

```bash
# Optimize cho 1 sách
python scripts/ml/05_optimize_discount_single.py

# Optimize cho tất cả sách
python scripts/ml/06_optimize_discount_all.py
```

### 5. Mở Power BI Dashboard

```bash
# Mở file .pbix bằng Power BI Desktop
DoAnNganhPBI.pbix
```

---

## 📊 Kết Quả

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 84.5% | 0.796 | 0.655 | 0.719 |
| Logistic Regression | 78.7% | 0.699 | 0.522 | 0.598 |

### Top Features (SHAP Analysis)

1. **rating_average** (42.3%) - Quan trọng nhất
2. **price** (28.7%)
3. **discount_rate** (18.5%)
4. **category** (10.5%)

### Recommendation System

- **Similarity Method**: Euclidean Distance
- **Accuracy**: Mean similarity = 0.981
- **Coverage**: 5 recommendations per book

---

## 🛠️ Công Nghệ

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Machine Learning
- **scikit-learn** - ML models & preprocessing
- **SHAP** - Model explainability

### Visualization
- **matplotlib** - Charts
- **seaborn** - Statistical plots
- **Power BI** - Interactive dashboards

### Web Scraping
- **requests** - HTTP requests
- **BeautifulSoup** - HTML parsing

---

## ⚠️ Hạn Chế Và Phương Pháp Luận

### Discount Optimization - Tương Quan vs Nhân Quả

Scripts `05_optimize_discount_single.py` và `06_optimize_discount_all.py` mô phỏng tác động của discount_rate lên xác suất bestseller bằng cách:
1. Giữ nguyên các features khác (price, rating, category)
2. Quét discount_rate từ 0% đến 50%
3. Dùng `model.predict_proba()` để tính xác suất

⚠️ **Đây là SUY LUẬN TƯƠNG QUAN (correlational), KHÔNG PHẢI NHÂN QUẢ (causal):**

**Vấn đề:**
- Model được train trên dữ liệu quan sát, không phải dữ liệu thực nghiệm (A/B test)
- Kết quả chỉ cho biết: "Sách có discount X% thường có xác suất bestseller Y% theo dữ liệu lịch sử"
- **KHÔNG đảm bảo** rằng tăng discount sẽ khiến sách trở thành bestseller trong thực tế
- Có thể tồn tại confounding factors:
  - **Reverse causation:** Sách bestseller vốn được giảm giá nhiều hơn (vì bán chạy → nhà bán hạ giá để promote)
  - **Omitted variables:** Yếu tố ẩn (brand nổi tiếng, marketing campaign, tác giả nổi tiếng) ảnh hưởng cả discount và bestseller

**Hạn chế phương pháp:**
- Model giả định "ceteris paribus" (các yếu tố khác không đổi) - không thực tế
- Kết quả CHỈ mang tính tham khảo, không nên dùng trực tiếp cho quyết định kinh doanh

**Khuyến nghị:**
- ✅ Dùng kết quả để khám phá pattern và đưa ra giả thuyết
- ✅ Kiểm chứng bằng thực nghiệm (A/B test) trước khi triển khai
- ✅ Kết hợp với domain knowledge (kinh tế, marketing)
- ✅ Nếu cần kết quả chính xác hơn, sử dụng phương pháp causal inference:
  - Propensity score matching
  - Uplift modeling
  - Instrumental variables
  - Regression discontinuity design

---

## 📈 Power BI Dashboard

Dashboard bao gồm 4 trang chính:

1. **Overview**: Tổng quan dữ liệu (KPIs, distributions)
2. **Price Analysis**: Phân tích giá sách theo category
3. **Rating & Quality**: Đánh giá chất lượng sách
4. **Bestseller Insights**: Phân tích yếu tố bestseller

---

## 📝 Dataset

- **Nguồn**: Tiki.vn
- **Số lượng**: 3,830 sách
- **Thể loại**: 4 categories
  - Sách tiếng Việt
  - Sách nước ngoài
  - Truyện tranh, Manga
  - Sách giáo khoa

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `id` | Book ID | int |
| `name` | Book name | string |
| `category_name` | Category | string |
| `price` | Price (VND) | float |
| `rating_average` | Average rating (0-5) | float |
| `bestseller` | Bestseller flag | binary (0/1) |
| `discount_rate` | Discount percentage | float |

---

## 🤝 Đóng Góp

Đây là đồ án cá nhân. Nếu có góp ý, vui lòng tạo issue hoặc pull request.

---

## 📧 Tác Giả

**[Lý Danh Tâm]**
- Email: [datammnh1506@gmail.com]
- GitHub: [@datammnh156](https://github.com/datammnh156)
- LinkedIn: [Danh Tâm](https://www.linkedin.com/in/datammnh156/)

---

## 📄 License

This project is for educational purposes only.

---

## 🙏 Acknowledgments

- Tiki.vn - Data source
- Scikit-learn - ML framework
- SHAP - Model interpretation
- Power BI - Visualization platform

---
