"""
Cải thiện Book Recommendations bằng cách mở rộng vector đặc trưng từ 2 chiều → 4 chiều

Feature cũ (2D): price, rating_average
Feature mới (4D): price, rating_average, discount_rate, quantity_sold

Tất cả được chuẩn hóa (scaled) trước khi tính Euclidean distance
Output: book_recommendations_4d.csv (sẽ thay thế cái cũ)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import sys
import io
from datetime import datetime

# Fix encoding cho Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*80)
print("🚀 TẠO BOOK RECOMMENDATIONS VỚI VECTOR 4 CHIỀU")
print("="*80)
print(f"Thời gian: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("="*80)

# ============================================================================
# 1. LOAD DỮ LIỆU
# ============================================================================
print("\n📂 Đang load dữ liệu...")
df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
print(f"✓ Đã load {len(df):,} sách")

# ============================================================================
# 2. CHUẨN BỊ FEATURES (4 CHIỀU)
# ============================================================================
print("\n🔧 Chuẩn bị features 4 chiều...")

# Chọn 4 features
features = ['price', 'rating_average', 'discount_rate', 'quantity_sold']
X = df[features].copy()

print(f"  Features: {features}")
print(f"  Shape: {X.shape}")

# Kiểm tra missing values
if X.isnull().any().any():
    print("⚠️  Có missing values, đang fill...")
    X = X.fillna(X.mean())

# Chuẩn hóa tất cả features
print("\n  Chuẩn hóa features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  Scaled shape: {X_scaled.shape}")
print(f"  Mean (sau scale): {X_scaled.mean(axis=0)}")
print(f"  Std (sau scale): {X_scaled.std(axis=0)}")

# ============================================================================
# 3. TÍNH KHOẢNG CÁCH EUCLIDEAN GIỮA TẤT CẢ SÁCH
# ============================================================================
print("\n⚡ Tính Euclidean distance giữa tất cả sách...")
print("  (Này sẽ mất chút thời gian với 3,830 sách...)")

distances = euclidean_distances(X_scaled)

print(f"  ✓ Distance matrix shape: {distances.shape}")
print(f"  ✓ Distance min: {distances[distances > 0].min():.6f}")
print(f"  ✓ Distance max: {distances.max():.6f}")

# ============================================================================
# 4. CHUYỂN DISTANCE → SIMILARITY (Cosine similarity-like)
# ============================================================================
print("\n🔄 Chuyển distance → similarity...")

# Dùng công thức: similarity = 1 / (1 + distance)
# Khoảng cách nhỏ → similarity cao
similarities = 1 / (1 + distances)

print(f"  ✓ Similarity min: {similarities[similarities < 1].min():.6f}")
print(f"  ✓ Similarity max: {similarities.max():.6f}")
print(f"  ✓ Similarity mean: {similarities[similarities < 1].mean():.6f}")

# ============================================================================
# 5. TẠO RECOMMENDATIONS (TOP N SIMILAR BOOKS)
# ============================================================================
print("\n📋 Tạo danh sách recommendations...")

N_RECOMMENDATIONS = 5  # Mỗi sách được gợi ý N cuốn tương tự
recommendations = []

for i in range(len(df)):
    # Lấy similarity với sách thứ i
    sim_scores = similarities[i, :]
    
    # Sắp xếp theo similarity (bỏ qua chính nó - index i)
    sorted_indices = np.argsort(-sim_scores)
    
    # Lấy top N recommended books (bỏ qua index i)
    for j in range(N_RECOMMENDATIONS + 1):
        idx = sorted_indices[j]
        if idx != i:  # Bỏ qua chính nó
            sim = sim_scores[idx]
            
            recommendations.append({
                'source_id': int(df.iloc[i]['id']),
                'source_name': df.iloc[i]['name'],
                'recommended_id': int(df.iloc[idx]['id']),
                'recommended_name': df.iloc[idx]['name'],
                'similarity_score': float(sim)
            })
    
    if (i + 1) % 500 == 0:
        print(f"  Đã xử lý {i + 1:,}/{len(df):,} sách")

print(f"\n✓ Tạo xong {len(recommendations):,} recommendations")

# ============================================================================
# 6. LƯU VÀO CSV
# ============================================================================
print("\n💾 Lưu recommendations...")

df_rec = pd.DataFrame(recommendations)
output_path = 'data/clean/book_recommendations.csv'
df_rec.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"✓ Đã lưu: {output_path}")
print(f"  - Tổng recommendations: {len(df_rec):,}")
print(f"  - Similarity min: {df_rec['similarity_score'].min():.6f}")
print(f"  - Similarity max: {df_rec['similarity_score'].max():.6f}")
print(f"  - Similarity mean: {df_rec['similarity_score'].mean():.6f}")

print("\n" + "="*80)
print("✅ HOÀN THÀNH! book_recommendations.csv đã được tạo")
print("="*80 + "\n")
