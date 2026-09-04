# So Sánh Mô Hình: Logistic Regression vs Random Forest

## Bảng So Sánh Hiệu Suất

| Model | Accuracy | Precision (Bestseller) | Recall (Bestseller) | F1-Score (Bestseller) | Train-Test Gap |
|-------|----------|----------------------|---------------------|---------------------|----------------|
| Logistic Regression | 0.703 | 0.516 | 0.199 | 0.287 | 0.011 |
| Random Forest | 0.754 | 0.653 | 0.390 | 0.488 | 0.066 |

## Phân Tích Chi Tiết

### 1. So Sánh Các Chỉ Số

- **Accuracy:** Random Forest cao hơn 5.1% (75.4% vs 70.3%)
- **Precision:** Random Forest cao hơn 13.7% (65.3% vs 51.6%)
- **Recall:** Random Forest cao hơn 19.1% (39.0% vs 19.9%)
- **F1-Score:** Random Forest cao hơn 20.1% (48.8% vs 28.7%)

### 2. Kiểm Tra Overfitting

- **Logistic Regression:** Train-Test Gap = 0.011
- **Random Forest:** Train-Test Gap = 0.066

Random Forest có gap thấp hơn, chứng tỏ generalization tốt hơn.

## Phân Tích Chi Tiết Hơn

## Biểu Đồ So Sánh

![Model Comparison](../charts/model_comparison.png)

---

*Báo cáo được tạo tự động vào 2026-09-03 19:16:50*
