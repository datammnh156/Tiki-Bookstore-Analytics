"""
Kiểm tra chất lượng recommendation: Phân tích similarity score và price/rating trùng lặp
"""

import pandas as pd
import sys
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*80)
print("PHÂN TÍCH CHẤT LƯỢNG BOOK RECOMMENDATIONS")
print("="*80)

# Load data
print("\n📂 Đang load dữ liệu...")
df_books = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
df_rec = pd.read_csv('data/clean/book_recommendations.csv', encoding='utf-8')

print(f"✓ Books: {len(df_books):,} sách")
print(f"✓ Recommendations: {len(df_rec):,} gợi ý")

# ============================================================================
# 1. PHÂN TÍCH SIMILARITY DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("1️⃣  PHÂN PHỐI SIMILARITY SCORE")
print("="*80)

high_sim = df_rec[df_rec['similarity_score'] >= 0.99]
high_sim_pct = (len(high_sim) / len(df_rec)) * 100

print(f"\nTổng số gợi ý: {len(df_rec):,}")
print(f"Gợi ý có similarity >= 0.99: {len(high_sim):,} ({high_sim_pct:.1f}%)")
print(f"\nPhân bố:")
print(f"  - Similarity = 1.000: {len(df_rec[df_rec['similarity_score'] == 1.0]):,}")
print(f"  - Similarity >= 0.999: {len(df_rec[df_rec['similarity_score'] >= 0.999]):,}")
print(f"  - Similarity >= 0.99: {len(high_sim):,}")
print(f"  - Similarity >= 0.95: {len(df_rec[df_rec['similarity_score'] >= 0.95]):,}")
print(f"  - Similarity >= 0.90: {len(df_rec[df_rec['similarity_score'] >= 0.90]):,}")

# ============================================================================
# 2. KIỂM TRA PRICE/RATING TRÙNG KHỚP
# ============================================================================
print("\n" + "="*80)
print("2️⃣  KIỂM TRA PRICE/RATING TRÙNG KHỚP (Nhóm similarity >= 0.99)")
print("="*80)

# Merge với thông tin sách
df_check = high_sim.merge(
    df_books[['id', 'price', 'rating_average', 'category_name']], 
    left_on='source_id', 
    right_on='id', 
    how='left'
).merge(
    df_books[['id', 'price', 'rating_average', 'category_name']], 
    left_on='recommended_id', 
    right_on='id', 
    how='left',
    suffixes=('_source', '_rec')
)

# Kiểm tra trùng khớp CHÍNH XÁC
df_check['price_match'] = df_check['price_source'] == df_check['price_rec']
df_check['rating_match'] = df_check['rating_average_source'] == df_check['rating_average_rec']
df_check['both_match'] = df_check['price_match'] & df_check['rating_match']

price_match_count = df_check['price_match'].sum()
rating_match_count = df_check['rating_match'].sum()
both_match_count = df_check['both_match'].sum()

price_match_pct = (price_match_count / len(df_check)) * 100
rating_match_pct = (rating_match_count / len(df_check)) * 100
both_match_pct = (both_match_count / len(df_check)) * 100

print(f"\nTrong {len(df_check):,} gợi ý có similarity >= 0.99:")
print(f"  - Price trùng CHÍNH XÁC: {price_match_count:,} ({price_match_pct:.1f}%)")
print(f"  - Rating trùng CHÍNH XÁC: {rating_match_count:,} ({rating_match_pct:.1f}%)")
print(f"  - CẢ HAI trùng CHÍNH XÁC: {both_match_count:,} ({both_match_pct:.1f}%)")

# ============================================================================
# 3. TÌM VÍ DỤ SIMILARITY = 1.0 NHƯNG TÊN/CATEGORY KHÔNG LIÊN QUAN
# ============================================================================
print("\n" + "="*80)
print("3️⃣  VÍ DỤ SIMILARITY = 1.0 NHƯNG KHÔNG LIÊN QUAN")
print("="*80)

# Lọc similarity = 1.0
perfect_sim = df_rec[df_rec['similarity_score'] == 1.0].copy()

# Merge với category
perfect_check = perfect_sim.merge(
    df_books[['id', 'name', 'price', 'rating_average', 'category_name']], 
    left_on='source_id', 
    right_on='id', 
    how='left'
).merge(
    df_books[['id', 'name', 'price', 'rating_average', 'category_name']], 
    left_on='recommended_id', 
    right_on='id', 
    how='left',
    suffixes=('_source', '_rec')
)

# Tìm các cặp có category khác nhau (có thể không liên quan)
different_category = perfect_check[
    perfect_check['category_name_source'] != perfect_check['category_name_rec']
].head(10)

print(f"\nTìm thấy {len(perfect_check):,} cặp có similarity = 1.0")
print(f"Trong đó {len(perfect_check[perfect_check['category_name_source'] != perfect_check['category_name_rec']]):,} cặp có category khác nhau")

print("\n5 VÍ DỤ (có category khác nhau - có thể không liên quan):\n")

for idx, row in different_category.head(5).iterrows():
    print(f"{'─'*80}")
    print(f"Cặp {idx + 1}:")
    print(f"  Sách gốc:")
    print(f"    - Tên: {row['name_source'][:60]}...")
    print(f"    - Category: {row['category_name_source']}")
    print(f"    - Price: {row['price_source']:,.0f} | Rating: {row['rating_average_source']:.1f}")
    print(f"  Sách được gợi ý:")
    print(f"    - Tên: {row['name_rec'][:60]}...")
    print(f"    - Category: {row['category_name_rec']}")
    print(f"    - Price: {row['price_rec']:,.0f} | Rating: {row['rating_average_rec']:.1f}")
    print(f"  → Similarity: {row['similarity_score']:.10f}")
    print()

# ============================================================================
# 4. PHÂN TÍCH SÂU HƠN: CÙNG CATEGORY VS KHÁC CATEGORY
# ============================================================================
print("\n" + "="*80)
print("4️⃣  SO SÁNH SIMILARITY TRONG CÙNG/KHÁC CATEGORY")
print("="*80)

# Merge toàn bộ recommendations với category
all_check = df_rec.merge(
    df_books[['id', 'category_name']], 
    left_on='source_id', 
    right_on='id', 
    how='left'
).merge(
    df_books[['id', 'category_name']], 
    left_on='recommended_id', 
    right_on='id', 
    how='left',
    suffixes=('_source', '_rec')
)

same_cat = all_check[all_check['category_name_source'] == all_check['category_name_rec']]
diff_cat = all_check[all_check['category_name_source'] != all_check['category_name_rec']]

print(f"\nCùng category ({len(same_cat):,} gợi ý):")
print(f"  - Avg similarity: {same_cat['similarity_score'].mean():.4f}")
print(f"  - % có sim >= 0.99: {(len(same_cat[same_cat['similarity_score'] >= 0.99]) / len(same_cat) * 100):.1f}%")

print(f"\nKhác category ({len(diff_cat):,} gợi ý):")
if len(diff_cat) > 0:
    print(f"  - Avg similarity: {diff_cat['similarity_score'].mean():.4f}")
    print(f"  - % có sim >= 0.99: {(len(diff_cat[diff_cat['similarity_score'] >= 0.99]) / len(diff_cat) * 100):.1f}%")
else:
    print(f"  - Không có gợi ý khác category (100% cùng category = TỐT!)")

print("\n" + "="*80)
print("✅ HOÀN THÀNH PHÂN TÍCH")
print("="*80 + "\n")