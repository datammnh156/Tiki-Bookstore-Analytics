import pandas as pd
import numpy as np
import pickle
import sys, io
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load dữ liệu
print("Loading data...")
df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
category_dummies = pd.get_dummies(df['category_name'], prefix='cat', dtype=int)
df = pd.concat([df, category_dummies], axis=1)

# Features
feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
cat_cols = [col for col in df.columns if col.startswith('cat_')]
feature_cols.extend(cat_cols)

# Load model
print("Loading model...")
model = pickle.load(open('models/bestseller_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# 3 sách cố định
sample_ids = [
    279437167,  # Improvement: 7.4% (23% → 40%) - recommend_change=True
    278777851,  # Improvement: 18.1% (39% → 20%) - recommend_change=True
    278856619   # Improvement: 0.0% (giữ nguyên 20%) - recommend_change=False
]

print("\n" + "="*70)
print("KIỂM TRA KẾT QUẢ RECOMMEND_CHANGE CỦA 3 SÁCH")
print("="*70 + "\n")

for idx, book_id in enumerate(sample_ids, 1):
    book = df[df['id'] == book_id].iloc[0]
    
    # Tính xác suất tại discount hiện tại
    base_features = {col: book[col] for col in feature_cols if col != 'discount_rate'}
    current_discount = book['discount_rate']
    current_features = base_features.copy()
    current_features['discount_rate'] = current_discount
    X_current = pd.DataFrame([current_features], columns=feature_cols)
    X_current_scaled = scaler.transform(X_current)
    current_prob = model.predict_proba(X_current_scaled)[0, 1]
    
    # Mô phỏng tất cả mức discount
    discount_rates = np.arange(0, 55, 5)
    results = []
    for discount in discount_rates:
        features = base_features.copy()
        features['discount_rate'] = discount
        X_sim = pd.DataFrame([features], columns=feature_cols)
        X_sim_scaled = scaler.transform(X_sim)
        proba = model.predict_proba(X_sim_scaled)[0, 1]
        results.append({'discount_rate': discount, 'prob': proba})
    
    results_df = pd.DataFrame(results)
    candidate_idx = results_df['prob'].idxmax()
    recommended_discount = results_df.loc[candidate_idx, 'discount_rate']
    recommended_prob = results_df.loc[candidate_idx, 'prob']
    
    improvement = recommended_prob - current_prob
    recommend_change = improvement >= 0.05
    
    print(f"Sách {idx}: {book['name'][:50]}")
    print(f"  ID: {book['id']}")
    print(f"  Mức giảm giá hiện tại: {current_discount:.0f}%")
    print(f"  Xác suất hiện tại: {current_prob:.1%}")
    print(f"  Mức giảm giá đề xuất: {recommended_discount:.0f}%")
    print(f"  Xác suất đề xuất: {recommended_prob:.1%}")
    print(f"  Cải thiện: {improvement:+.1%}")
    print(f"  Đề xuất thay đổi: {'TRUE ' if recommend_change else 'FALSE '}")
    print()

print("="*70)
print("Hoàn Thành!")
print("="*70)
