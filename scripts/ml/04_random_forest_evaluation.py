"""
Đánh giá tổng thể model bestseller prediction
Kiểm tra: Cross-validation, Overfitting, Calibration, Stability, Baseline 
của Random Forest model đã train

"""

import pandas as pd
import numpy as np
import pickle
import sys
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*70)
print("📊 ĐÁNH GIÁ TỔNG THỂ MODEL BESTSELLER PREDICTION")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n🔧 Đang tải dữ liệu...")
df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
print(f"✅ Đã load data: {len(df):,} sách")

# Create one-hot encoding
category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
df = pd.concat([df, category_dummies], axis=1)

# Define features
feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
cat_cols = [col for col in df.columns if col.startswith('cat_')]
feature_cols.extend(cat_cols)

X = df[feature_cols]
y = df['is_bestseller']

print(f"✅ Số features: {len(feature_cols)}")
print(f"✅ Tỷ lệ bestseller: {y.mean():.1%} ({y.sum():,}/{len(y):,})")

# ============================================================================
# 1. CROSS-VALIDATION (K-FOLD)
# ============================================================================

print("\n" + "="*70)
print("1️⃣ CROSS-VALIDATION (5-FOLD)")
print("="*70)

# Chuẩn bị model và scaler
model_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}

# K-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    score = model.score(X_test_scaled, y_test)
    cv_scores.append(score)
    print(f"   Fold {fold}: {score:.1%}")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"\n📊 Kết quả Cross-Validation:")
print(f"   - Mean Accuracy: {cv_mean:.1%}")
print(f"   - Std Deviation: {cv_std:.3f}")
print(f"   - Min: {min(cv_scores):.1%}")
print(f"   - Max: {max(cv_scores):.1%}")

if cv_std < 0.02:
    print(f"   ✅ Model ổn định (std < 2%)")
else:
    print(f"   ⚠️ Model có biến động (std ≥ 2%)")

# ============================================================================
# 2. OVERFITTING CHECK
# ============================================================================

print("\n" + "="*70)
print("2️⃣ OVERFITTING CHECK")
print("="*70)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(**model_params)
model.fit(X_train_scaled, y_train)

# Evaluate on both sets
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
gap = train_acc - test_acc

print(f"   - Train Accuracy: {train_acc:.1%}")
print(f"   - Test Accuracy: {test_acc:.1%}")
print(f"   - Gap (Train - Test): {gap:.1%}")

if gap > 0.10:
    print(f"   ⚠️ CẢNH BÁO: Overfitting nghiêm trọng (gap > 10%)")
elif gap > 0.05:
    print(f"   ⚠️ Có dấu hiệu overfitting nhẹ (gap 5-10%)")
else:
    print(f"   ✅ Model không bị overfitting đáng kể (gap < 5%)")

# ============================================================================
# 3. CALIBRATION CHECK
# ============================================================================

print("\n" + "="*70)
print("3️⃣ CALIBRATION CHECK")
print("="*70)

# Predict probabilities
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Method 1: Simple binning
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

print("\n📊 Phân tích theo bin xác suất:")
print(f"{'Bin Xác Suất':<15} {'Số Mẫu':<10} {'Tỷ Lệ Thực':<15} {'Chênh Lệch':<15}")
print("-" * 60)

for i in range(len(bins)-1):
    mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
    n_samples = mask.sum()
    
    if n_samples > 0:
        actual_rate = y_test[mask].mean()
        expected_rate = (bins[i] + bins[i+1]) / 2
        diff = abs(actual_rate - expected_rate)
        
        print(f"{bin_labels[i]:<15} {n_samples:<10} {actual_rate:>7.1%}        {diff:>7.1%}")

# Method 2: Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

print("\n📈 Calibration Curve (10 bins):")
for i, (true_p, pred_p) in enumerate(zip(prob_true, prob_pred), 1):
    diff = abs(true_p - pred_p)
    status = "✅" if diff < 0.1 else "⚠️"
    print(f"   Bin {i:2d}: Dự đoán {pred_p:.1%} → Thực tế {true_p:.1%} (chênh {diff:.1%}) {status}")

avg_diff = np.mean(np.abs(prob_true - prob_pred))
print(f"\n📊 Chênh lệch trung bình: {avg_diff:.1%}")

if avg_diff < 0.05:
    print("   ✅ Model đã calibrated tốt (chênh < 5%)")
elif avg_diff < 0.10:
    print("   ⚠️ Calibration chấp nhận được (chênh 5-10%)")
else:
    print("   ⚠️ Model cần calibration (chênh > 10%)")

# Save calibration plot
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, 'o-', label='Random Forest')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve - Bestseller Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/charts/calibration_curve.png', dpi=150)
print("\n💾 Đã lưu calibration plot: outputs/charts/calibration_curve.png")

# ============================================================================
# 4. STABILITY CHECK
# ============================================================================

print("\n" + "="*70)
print("4️⃣ STABILITY CHECK (Multiple Random Seeds)")
print("="*70)

seeds = [0, 1, 42, 100, 123]
stability_results = []

for seed in seeds:
    # Train/test split với seed khác nhau
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Scale
    sc = StandardScaler()
    X_tr_scaled = sc.fit_transform(X_tr)
    X_te_scaled = sc.transform(X_te)
    
    # Train model với seed khác nhau
    params = model_params.copy()
    params['random_state'] = seed
    m = RandomForestClassifier(**params)
    m.fit(X_tr_scaled, y_tr)
    
    # Evaluate
    acc = m.score(X_te_scaled, y_te)
    
    # Feature importance top 3
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': m.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top3 = feature_importance.head(3)['feature'].tolist()
    
    stability_results.append({
        'seed': seed,
        'accuracy': acc,
        'top3_features': top3
    })
    
    print(f"   Seed {seed:3d}: Accuracy {acc:.1%}, Top 3 features: {', '.join(top3[:3])}")

# Analyze stability
accuracies = [r['accuracy'] for r in stability_results]
acc_mean = np.mean(accuracies)
acc_std = np.std(accuracies)
acc_range = max(accuracies) - min(accuracies)

print(f"\n📊 Phân tích Stability:")
print(f"   - Mean Accuracy: {acc_mean:.1%}")
print(f"   - Std Deviation: {acc_std:.3f}")
print(f"   - Range: {acc_range:.1%}")

if acc_std < 0.01:
    print(f"   ✅ Model rất ổn định (std < 1%)")
elif acc_std < 0.02:
    print(f"   ✅ Model ổn định (std < 2%)")
else:
    print(f"   ⚠️ Model có biến động (std ≥ 2%)")

# ============================================================================
# 5. BASELINE COMPARISON
# ============================================================================

print("\n" + "="*70)
print("5️⃣ SO SÁNH VỚI BASELINE NGÂY THƠ")
print("="*70)

# Baseline: always predict majority class (non-bestseller)
baseline_pred = np.zeros(len(y_test))  # Dự đoán tất cả là 0 (non-bestseller)
baseline_acc = accuracy_score(y_test, baseline_pred)

# Random baseline
random_acc = y_test.mean()  # Xác suất dự đoán đúng nếu random

print(f"   - Baseline 1 (Dự đoán tất cả = 0): {baseline_acc:.1%}")
print(f"   - Baseline 2 (Random guess): {random_acc:.1%}")
print(f"   - Model của tôi (Test Accuracy): {test_acc:.1%}")
print(f"   - Improvement vs Baseline 1: {test_acc - baseline_acc:+.1%}")
print(f"   - Improvement vs Random: {test_acc - random_acc:+.1%}")

if test_acc > baseline_acc + 0.10:
    print(f"   ✅ Model vượt trội so với baseline (>10%)")
elif test_acc > baseline_acc + 0.05:
    print(f"   ✅ Model tốt hơn baseline đáng kể (5-10%)")
else:
    print(f"   ⚠️ Model chưa thực sự tốt hơn baseline nhiều (<5%)")

# ============================================================================
# 6. CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("6️⃣ CONFUSION MATRIX & CLASSIFICATION REPORT")
print("="*70)

y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Confusion Matrix:")
print(f"                 Predicted: 0    Predicted: 1")
print(f"Actual: 0        {cm[0,0]:>6}       {cm[0,1]:>6}")
print(f"Actual: 1        {cm[1,0]:>6}       {cm[1,1]:>6}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Non-Bestseller', 'Bestseller'],
                          digits=3))

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("7️⃣ FEATURE IMPORTANCE")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Features quan trọng nhất:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<25} {row['importance']:.3f}")

# ============================================================================
# TỔNG KẾT & KẾT LUẬN
# ============================================================================

print("\n" + "="*70)
print("📝 TỔNG KẾT & KẾT LUẬN")
print("="*70)

print("\n✅ ĐIỂM MẠNH:")
strengths = []

if cv_std < 0.02:
    strengths.append("Model ổn định qua nhiều lần chia dữ liệu (CV std < 2%)")
    
if gap < 0.05:
    strengths.append("Không bị overfitting đáng kể (train-test gap < 5%)")
    
if avg_diff < 0.10:
    strengths.append("Calibration chấp nhận được (avg diff < 10%)")
    
if acc_std < 0.02:
    strengths.append("Kết quả ổn định với các random seed khác nhau")
    
if test_acc > baseline_acc + 0.10:
    strengths.append(f"Vượt trội so với baseline ({test_acc - baseline_acc:+.1%})")

for i, s in enumerate(strengths, 1):
    print(f"   {i}. {s}")

print("\n⚠️ HẠN CHẾ CẦN LƯU Ý:")
limitations = []

if cv_std >= 0.02:
    limitations.append(f"Model có biến động qua các fold (std = {cv_std:.3f})")
    
if gap >= 0.05:
    limitations.append(f"Có dấu hiệu overfitting (gap = {gap:.1%})")
    
if avg_diff >= 0.10:
    limitations.append(f"Calibration chưa tốt (avg diff = {avg_diff:.1%})")
    
if test_acc < baseline_acc + 0.10:
    limitations.append(f"Chưa vượt trội nhiều so với baseline")

# Thêm hạn chế về dữ liệu
limitations.append("Chỉ có 8 features, có thể thiếu thông tin quan trọng")
limitations.append("Data imbalance (30% bestseller, 70% non-bestseller)")
limitations.append("Định nghĩa bestseller dựa trên quantile 70%, có thể chủ quan")

for i, l in enumerate(limitations, 1):
    print(f"   {i}. {l}")

print("\n💡 ĐÁNH GIÁ CUỐI CÙNG:")
if test_acc > 0.80 and cv_std < 0.02 and gap < 0.10:
    print("   ✅ Model đáng tin cậy, có thể sử dụng cho Streamlit/Dashboard")
    print("   ✅ Kết quả có thể trình bày trong thesis với độ tin cậy cao")
elif test_acc > 0.75:
    print("   ⚠️ Model chấp nhận được, nhưng cần lưu ý hạn chế khi trình bày")
    print("   ⚠️ Nên đề cập đến các điểm yếu trong phần \"Hạn chế\" của thesis")
else:
    print("   ❌ Model chưa đủ tốt, cần cải thiện trước khi sử dụng")