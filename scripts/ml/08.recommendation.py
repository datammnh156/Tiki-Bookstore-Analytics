"""
HYBRID RECOMMENDATION SYSTEM 
Changes:
  1. FIX: Detect sparse TF-IDF vectors (< 2 keywords)
  2. If source OR target is sparse: set tfidf_similarity = 0
  3. Use Euclidean for numeric similarity
  4. Weights: 70% TF-IDF + 30% Numeric
Expected: Eliminate most fake similarity=1.0 cases
"""

import pandas as pd
import numpy as np
import re, sys, io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
print("\n" + "="*90)
print("🚀 HỆ THỐNG GỢI Ý SÁCH DỰA TRÊN NỘI DUNG KẾT HỢP ĐẶC TRƯNG SỐ")
print("="*90)

# Load data
print("\n[1/9] Loading data...")
df = pd.read_csv("data/clean/tiki_books_cleaned.csv", encoding="utf-8")
print(f"✓ Loaded {len(df):,} books")

# Preprocess
print("\n[2/9] Preprocessing text...")
def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["name_processed"] = df["name"].apply(preprocess_text)
print(f"✓ Processed {len(df):,} book names")

# TF-IDF
print("\n[3/9] Computing TF-IDF...")
stop_words = ["sách","tập","the","of","and","của","tặng","kèm","book","edition","bản","tiếng","tái","quyền"]
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), lowercase=True, stop_words=stop_words)
tfidf_matrix = tfidf.fit_transform(df["name_processed"])
print(f"✓ TF-IDF matrix: {tfidf_matrix.shape}")

tfidf_similarity = cosine_similarity(tfidf_matrix)

# 🔧 COMPREHENSIVE FIX FOR SPARSE VECTORS
print("\n[4/9] 🔧 FIXING: Comprehensive sparse vector handling...")

# Detect sparse vectors
non_zero_counts = np.array([tfidf_matrix[i].nnz for i in range(tfidf_matrix.shape[0])])
sparse_mask = non_zero_counts < 2
sparse_indices = np.where(sparse_mask)[0]

print(f"  Sparse books (< 2 keywords): {sparse_mask.sum()} / {len(df)} ({sparse_mask.sum()/len(df)*100:.2f}%)")
print(f"    - 0 keywords: {np.sum(non_zero_counts == 0)}")
print(f"    - 1 keyword: {np.sum(non_zero_counts == 1)}")

# Show examples
print(f"\n  Examples of sparse books:")
for idx in sparse_indices[:10]:
    print(f"    - {non_zero_counts[idx]:2d} kw: {df.iloc[idx]['name'][:55]}")

# Count BEFORE fix
sim1_before_mask = tfidf_similarity >= 0.9999
np.fill_diagonal(sim1_before_mask, False)
sim1_before = np.sum(sim1_before_mask)
print(f"\n  TF-IDF sim=1.0 BEFORE: {sim1_before}")

# Apply comprehensive fix
# If source OR target is sparse, set similarity = 0
print(f"\n  Applying fix: If source OR target is sparse, set sim=0...")
for i in sparse_indices:
    tfidf_similarity[i, :] = 0  # All pairs where source is sparse
    tfidf_similarity[:, i] = 0  # All pairs where target is sparse

sim1_after_mask = tfidf_similarity >= 0.9999
np.fill_diagonal(sim1_after_mask, False)
sim1_after = np.sum(sim1_after_mask)
reduction = sim1_before - sim1_after
reduction_pct = (reduction / sim1_before * 100) if sim1_before > 0 else 0

print(f"  TF-IDF sim=1.0 AFTER: {sim1_after}")
print(f"  ✓ Eliminated: {reduction} pairs ({reduction_pct:.1f}%)")

# Numeric similarity (Euclidean)
print("\n[5/9] Computing numeric similarity (Euclidean)...")
numeric_features = ["price","rating_average","discount_rate","quantity_sold","review_count","has_rating","is_bestseller"]
X_numeric = df[numeric_features].copy()
if X_numeric.isnull().any().any():
    X_numeric = X_numeric.fillna(X_numeric.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

distances = euclidean_distances(X_scaled)
numeric_similarity = 1 / (1 + distances)
print(f"✓ Numeric similarity (Euclidean): {numeric_similarity.shape}")

# Hybrid (70/30)
print("\n[6/9] Combining hybrid (70% TF-IDF + 30% Numeric)...")
final_similarity = 0.7 * tfidf_similarity + 0.3 * numeric_similarity
print("✓ Weights: TF-IDF 70% + Numeric 30%")

# Đếm các cặp sách khác nhau có độ tương đồng gần bằng 1
final_sim1_mask = final_similarity >= 0.9999

# Không tính một cuốn sách so sánh với chính nó
np.fill_diagonal(final_sim1_mask, False)

final_sim1 = np.sum(final_sim1_mask)
print(f"  Final sim=1.0: {final_sim1}")

# Recommendations
print("\n[7/9] Creating recommendations...")
recommendations = []

for i in range(len(df)):
    source_category = df.iloc[i]["category_name"]
    sim_scores = final_similarity[i, :]
    
    candidates = [(j, sim_scores[j]) for j in range(len(df)) 
                  if j != i and df.iloc[j]["category_name"] == source_category]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for idx, sim in candidates[:5]:
        confidence = "High" if sim >= 0.7 else ("Medium" if sim >= 0.5 else "Low")
        recommendations.append({
            "source_id": int(df.iloc[i]["id"]),
            "source_name": df.iloc[i]["name"],
            "recommended_id": int(df.iloc[idx]["id"]),
            "recommended_name": df.iloc[idx]["name"],
            "similarity_score": float(sim),
            "confidence_level": confidence
        })
    
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1:,}/{len(df):,}")

print(f"✓ Created {len(recommendations):,} recommendations")

# Save
print("\n[8/9] Saving to CSV...")
df_rec = pd.DataFrame(recommendations)
import os
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/clean/book_recommendations.csv'))
df_rec.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✓ Saved: {output_path}")

# Summary
print("\n[9/9] Summary:")
print("="*90)
print(f"  Sparse books: {sparse_mask.sum()} (0 kw: {np.sum(non_zero_counts == 0)}, 1 kw: {np.sum(non_zero_counts == 1)})")
print(f"  TF-IDF sim=1.0: {sim1_before} → {sim1_after} (eliminated: {reduction})")
print(f"  Recommendations: {len(recommendations):,}")

high = len([r for r in recommendations if r['confidence_level'] == 'High'])
med = len([r for r in recommendations if r['confidence_level'] == 'Medium'])
low = len([r for r in recommendations if r['confidence_level'] == 'Low'])
print(f"  Confidence: High={high:,} ({high/len(recommendations)*100:.1f}%) | Medium={med:,} ({med/len(recommendations)*100:.1f}%) | Low={low:,}")

print("\n" + "="*90)
print("✅ Hoàn Thành")
print("="*90)
