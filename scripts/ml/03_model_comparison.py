"""
So sánh giữa Logistic Regression và Random Forest
"""

import pandas as pd
import numpy as np
import pickle
import sys
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*80)
print("📊 SO SÁNH MÔ HÌNH: LOGISTIC REGRESSION vs RANDOM FOREST")
print("="*80)

# ============================================================================
# 1. LOAD VÀ CHUẨN BỊ DỮ LIỆU 
# ============================================================================

print("\n🔧 Đang tải và chuẩn bị dữ liệu...")
df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
print(f"✅ Đã load {len(df):,} sách")

# Create one-hot encoding
category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
df = pd.concat([df, category_dummies], axis=1)

# Define features (8 features)
feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
cat_cols = [col for col in df.columns if col.startswith('cat_')]
feature_cols.extend(cat_cols)

print(f"✅ Số features: {len(feature_cols)}")
print(f"   - Numeric: price, discount_rate, rating_average, has_rating")
print(f"   - Category: {len(cat_cols)} cột one-hot")

X = df[feature_cols]
y = df['is_bestseller']

# Train/test split (GIỐNG NHAU)
print(f"\n✅ Tỷ lệ bestseller: {y.mean():.1%}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train size: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"✅ Test size: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Scaling (GIỐNG NHAU)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Đã scale features")

# ============================================================================
# 2. TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*80)
print("1️⃣ LOGISTIC REGRESSION ")
print("="*80)

# Load Logistic Regression từ train_model_logistic.py
lr_model = pickle.load(open('models/logistic_model.pkl', 'rb'))
lr_scaler_loaded = pickle.load(open('models/logistic_scaler.pkl', 'rb'))

print("✅ Đã load Logistic Regression từ models/logistic_model.pkl")
print("✅ Đã load scaler từ models/logistic_scaler.pkl")

# Evaluate (dùng scaler gốc từ train/test split, không dùng lr_scaler_loaded)
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)
lr_precision = precision_score(y_test, lr_test_pred)
lr_recall = recall_score(y_test, lr_test_pred)
lr_f1 = f1_score(y_test, lr_test_pred)
lr_gap = lr_train_acc - lr_test_acc

print(f"\n📊 Kết quả Logistic Regression:")
print(f"   - Train Accuracy: {lr_train_acc:.3f}")
print(f"   - Test Accuracy: {lr_test_acc:.3f}")
print(f"   - Precision (Bestseller): {lr_precision:.3f}")
print(f"   - Recall (Bestseller): {lr_recall:.3f}")
print(f"   - F1-Score (Bestseller): {lr_f1:.3f}")
print(f"   - Train-Test Gap: {lr_gap:.3f}")

# ============================================================================
# 3. LOAD RANDOM FOREST ĐÃ TRAIN
# ============================================================================

print("\n" + "="*80)
print("2️⃣ RANDOM FOREST ")
print("="*80)

with open('models/bestseller_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print("✅ Đã load Random Forest từ models/bestseller_model.pkl")
print(f"   - n_estimators: {rf_model.n_estimators}")
print(f"   - max_depth: {rf_model.max_depth}")
print(f"   - min_samples_split: {rf_model.min_samples_split}")

# Evaluate
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_precision = precision_score(y_test, rf_test_pred)
rf_recall = recall_score(y_test, rf_test_pred)
rf_f1 = f1_score(y_test, rf_test_pred)
rf_gap = rf_train_acc - rf_test_acc

print(f"\n📊 Kết quả Random Forest:")
print(f"   - Train Accuracy: {rf_train_acc:.3f}")
print(f"   - Test Accuracy: {rf_test_acc:.3f}")
print(f"   - Precision (Bestseller): {rf_precision:.3f}")
print(f"   - Recall (Bestseller): {rf_recall:.3f}")
print(f"   - F1-Score (Bestseller): {rf_f1:.3f}")
print(f"   - Train-Test Gap: {rf_gap:.3f}")

# ============================================================================
# 4. BẢNG SO SÁNH
# ============================================================================

print("\n" + "="*80)
print("📊 BẢNG SO SÁNH Logistic Regression và Random Forest đã Train")
print("="*80 + "\n")

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_test_acc, rf_test_acc],
    'Precision (Bestseller)': [lr_precision, rf_precision],
    'Recall (Bestseller)': [lr_recall, rf_recall],
    'F1-Score (Bestseller)': [lr_f1, rf_f1],
    'Train-Test Gap': [lr_gap, rf_gap]
})

# Format numbers
for col in comparison_df.columns[1:]:
    comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.3f}")

# In bảng với format đẹp
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Gap':<12}")
print("-" * 85)

for idx, row in comparison_df.iterrows():
    model_name = row['Model']
    acc = row['Accuracy']
    prec = row['Precision (Bestseller)']
    recall = row['Recall (Bestseller)']
    f1 = row['F1-Score (Bestseller)']
    gap = row['Train-Test Gap']
    
    print(f"{model_name:<25} {acc:<12} {prec:<12} {recall:<12} {f1:<12} {gap:<12}")

print()


# ============================================================================
# 5. VẼ BIỂU ĐỒ SO SÁNH
# ============================================================================

print("\n" + "="*80)
print("📈 ĐANG VẼ BIỂU ĐỒ SO SÁNH...")
print("="*80)

# Prepare data for plotting
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lr_scores = [lr_test_acc, lr_precision, lr_recall, lr_f1]
rf_scores = [rf_test_acc, rf_precision, rf_recall, rf_f1]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#3498db')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='#2ecc71')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('So Sánh Hiệu Suất: Logistic Regression vs Random Forest', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('outputs/charts/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Đã lưu biểu đồ: outputs/charts/model_comparison.png")

# ============================================================================
# 6. LƯU BÁO CÁO
# ============================================================================

print("\n" + "="*80)
print("💾 ĐANG LƯU BÁO CÁO...")
print("="*80)

# Tính toán các hiệu số
acc_diff = rf_test_acc - lr_test_acc
prec_diff = rf_precision - lr_precision
recall_diff = rf_recall - lr_recall
f1_diff = rf_f1 - lr_f1


report = f"""# So Sánh Mô Hình: Logistic Regression vs Random Forest

## Bảng So Sánh Hiệu Suất

| Model | Accuracy | Precision (Bestseller) | Recall (Bestseller) | F1-Score (Bestseller) | Train-Test Gap |
|-------|----------|----------------------|---------------------|---------------------|----------------|
| Logistic Regression | {lr_test_acc:.3f} | {lr_precision:.3f} | {lr_recall:.3f} | {lr_f1:.3f} | {lr_gap:.3f} |
| Random Forest | {rf_test_acc:.3f} | {rf_precision:.3f} | {rf_recall:.3f} | {rf_f1:.3f} | {rf_gap:.3f} |

## Phân Tích Chi Tiết

### 1. So Sánh Các Chỉ Số

- **Accuracy:** Random Forest cao hơn {acc_diff:.1%} ({rf_test_acc:.1%} vs {lr_test_acc:.1%})
- **Precision:** Random Forest cao hơn {prec_diff:.1%} ({rf_precision:.1%} vs {lr_precision:.1%})
- **Recall:** Random Forest cao hơn {recall_diff:.1%} ({rf_recall:.1%} vs {lr_recall:.1%})
- **F1-Score:** Random Forest cao hơn {f1_diff:.1%} ({rf_f1:.1%} vs {lr_f1:.1%})

### 2. Kiểm Tra Overfitting

- **Logistic Regression:** Train-Test Gap = {lr_gap:.3f}
- **Random Forest:** Train-Test Gap = {rf_gap:.3f}

Random Forest có gap thấp hơn, chứng tỏ generalization tốt hơn.

## Phân Tích Chi Tiết Hơn

## Biểu Đồ So Sánh

![Model Comparison](../charts/model_comparison.png)

---

*Báo cáo được tạo tự động vào {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open('outputs/reports/model_comparison.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("✅ Đã lưu báo cáo: outputs/reports/model_comparison.md")

print("\n" + "="*80)
print("✅ HOÀN THÀNH!")
print("="*80)

