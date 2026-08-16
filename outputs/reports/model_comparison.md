# So Sánh Mô Hình: Logistic Regression vs Random Forest

## Bảng So Sánh Hiệu Suất

| Model | Accuracy | Precision (Bestseller) | Recall (Bestseller) | F1-Score (Bestseller) | Train-Test Gap |
|-------|----------|----------------------|---------------------|---------------------|----------------|
| Logistic Regression | 0.787 | 0.699 | 0.522 | 0.598 | 0.001 |
| Random Forest | 0.845 | 0.796 | 0.655 | 0.719 | 0.018 |

## Phân Tích Chi Tiết

### 1. So Sánh Các Chỉ Số

- **Accuracy:** Random Forest cao hơn 5.7% (84.5% vs 78.7%)
- **Precision:** Random Forest cao hơn 9.6% (79.6% vs 69.9%)
- **Recall:** Random Forest cao hơn 13.4% (65.5% vs 52.2%)
- **F1-Score:** Random Forest cao hơn 12.1% (71.9% vs 59.8%)

### 2. Kiểm Tra Overfitting

- **Logistic Regression:** Train-Test Gap = 0.001
- **Random Forest:** Train-Test Gap = 0.018

Random Forest có gap thấp hơn, chứng tỏ generalization tốt hơn.

## Phân Tích Chi Tiết Hơn

## Biểu Đồ So Sánh

![Model Comparison](../charts/model_comparison.png)

---

*Báo cáo được tạo tự động vào 2026-08-15 15:43:01*
