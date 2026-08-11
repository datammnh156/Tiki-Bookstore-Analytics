# 📚 Dự Án Phân Tích Sách Tiki

Đồ án phân tích dữ liệu và dự đoán sách bestseller trên nền tảng Tiki.vn sử dụng Machine Learning và Power BI.

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
git clone https://github.com/YOUR_USERNAME/tiki-books-analysis.git
cd tiki-books-analysis
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
| **Random Forest** | 87.3% | 0.85 | 0.89 | 0.87 |
| Logistic Regression | 82.1% | 0.80 | 0.84 | 0.82 |

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

**[Tên của bạn]**
- Email: [email@example.com]
- GitHub: [@datammnh156](https://github.com/datammnh156)
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)

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

**⭐ Nếu project này hữu ích, hãy cho một star nhé!**