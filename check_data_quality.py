import pandas as pd
import numpy as np

print("="*100)
print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU - data/raw/tiki_book_dataset_crawl.csv")
print("="*100)
print()

# Load data
df = pd.read_csv('data/raw/tiki_book_dataset_crawl.csv')

print(f"📊 Tổng số dòng: {len(df):,}")
print()

# 1. Kiểm tra trùng lặp ID
print("="*100)
print("1. KIỂM TRA TRÙNG LẶP ID")
print("="*100)
duplicates = df['id'].duplicated().sum()
duplicate_ids = df[df['id'].duplicated(keep=False)]['id'].unique()
print(f"Số ID bị trùng lặp: {duplicates:,}")
if duplicates > 0:
    print(f"Số ID unique bị trùng: {len(duplicate_ids):,}")
    print(f"% trùng lặp: {duplicates/len(df)*100:.2f}%")
else:
    print("✅ Không có ID trùng lặp")
print()

# 2. Sort bias - quantity_sold theo TỪNG category
print("="*100)
print("2. SORT BIAS - QUANTITY_SOLD THEO TỪNG CATEGORY")
print("="*100)
print(f"{'Category':<30} | {'Min':<10} | {'Median':<10} | {'Max':<10} | {'Count':<8}")
print("-"*30 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*8)

for category in sorted(df['category_name'].unique()):
    cat_data = df[df['category_name'] == category]['quantity_sold']
    min_qty = cat_data.min()
    median_qty = cat_data.median()
    max_qty = cat_data.max()
    count = len(cat_data)
    
    print(f"{category:<30} | {min_qty:<10.0f} | {median_qty:<10.0f} | {max_qty:<10.0f} | {count:<8}")

print()

# 3. % rating = 0 (has_rating)
print("="*100)
print("3. RATING = 0 (HAS_RATING)")
print("="*100)
df['has_rating'] = (df['rating_average'] > 0).astype(int)
rating_counts = df['has_rating'].value_counts()
print(f"Có rating (rating > 0):    {rating_counts.get(1, 0):,} ({rating_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Không có rating (rating=0): {rating_counts.get(0, 0):,} ({rating_counts.get(0, 0)/len(df)*100:.1f}%)")
print()

# 4. Các field quan trọng có null không
print("="*100)
print("4. KIỂM TRA FIELD NULL")
print("="*100)
important_fields = ['id', 'name', 'price', 'original_price', 'discount_rate', 
                    'rating_average', 'review_count', 'quantity_sold', 
                    'favourite_count', 'category_id', 'category_name']

print(f"{'Field':<25} | {'Null Count':<12} | {'% Null':<10}")
print("-"*25 + "-+-" + "-"*12 + "-+-" + "-"*10)

for field in important_fields:
    if field in df.columns:
        null_count = df[field].isna().sum()
        null_pct = null_count / len(df) * 100
        status = "⚠️" if null_count > 0 else "✅"
        print(f"{status} {field:<22} | {null_count:<12} | {null_pct:<9.2f}%")
    else:
        print(f"❌ {field:<22} | KHÔNG TỒN TẠI")

print()

# Thống kê phân bố category
print("="*100)
print("5. PHÂN BỐ THEO CATEGORY")
print("="*100)
category_dist = df['category_name'].value_counts().sort_index()
print(f"{'Category':<30} | {'Số lượng':<10} | {'% Tổng':<10}")
print("-"*30 + "-+-" + "-"*10 + "-+-" + "-"*10)
for category, count in category_dist.items():
    pct = count / len(df) * 100
    print(f"{category:<30} | {count:<10} | {pct:<9.1f}%")

print()
print("="*100)
print("HOÀN THÀNH KIỂM TRA")
print("="*100)
