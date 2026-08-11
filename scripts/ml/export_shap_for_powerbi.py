"""
Xuat SHAP values cho toan bo 3830 sach vao file CSV dang long format.
Danh cho Power BI visualization.
"""

import pandas as pd
import pickle
import sys
import numpy as np
import shap
import time

# Fix encoding cho Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Mapping feature names to Vietnamese
FEATURE_NAME_MAP = {
    'price': 'Giá bán',
    'discount_rate': 'Mức giảm giá',
    'rating_average': 'Điểm đánh giá',
    'has_rating': 'Có đánh giá',
    'cat_Sách giáo khoa': 'Thể loại: Sách giáo khoa',
    'cat_Sách nước ngoài': 'Thể loại: Sách nước ngoài',
    'cat_Sách tiếng Việt': 'Thể loại: Sách tiếng Việt',
    'cat_Truyện tranh, Manga': 'Thể loại: Truyện tranh, Manga'
}


def estimate_runtime():
    """
    Ước tính thời gian chạy cho 3830 sach
    """
    print("=" * 70)
    print("UONG TINH THOI GIAN CHAY")
    print("=" * 70)
    
    # Load du lieu
    df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/bestseller_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Tao features
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    print("\nTong so sach: {}".format(len(X)))
    print("Tong so features: {}".format(len(feature_cols)))
    
    # Test voi 100 samples
    print("\nDang test voi 100 samples dau tien...")
    explainer = shap.TreeExplainer(model)
    
    start = time.time()
    shap_values_test = explainer.shap_values(X_scaled[:100])
    elapsed = time.time() - start
    
    print("Thoi gian cho 100 samples: {:.2f} giay".format(elapsed))
    
    # Tinh toan thoi gian toan bo
    estimated_total = (len(X) / 100) * elapsed
    
    print("\nUONG TINH THOI GIAN TOAN BO:")
    print("- {} samples / 100 = {:.1f}x".format(len(X), len(X) / 100))
    print("- Thoi gian uu tinh: {:.1f} giay = {:.1f} phut".format(
        estimated_total, estimated_total / 60))
    
    if estimated_total > 180:  # > 3 phut
        print("\n⚠️  CANH BAO: Thoi gian > 3 phut!")
        print("Co the chay trong che do background hoac toi om.\n")
    
    return estimated_total <= 180


def export_shap_to_csv(output_path='data/clean/shap_values.csv'):
    """
    Xuat SHAP values cho toan bo dataset vao CSV long format
    """
    print("\n" + "=" * 70)
    print("XUAT SHAP VALUES RA CSV")
    print("=" * 70)
    
    # Load du lieu
    print("\nLoading data...")
    df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/bestseller_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Tao features
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    
    print("Tong so sach: {}".format(len(X)))
    
    # Tinh SHAP values
    print("\nTinh SHAP values cho toan bo dataset...")
    print("(This may take 1-2 minutes...)\n")
    
    explainer = shap.TreeExplainer(model)
    start_time = time.time()
    
    shap_values_raw = explainer.shap_values(X_scaled)
    
    elapsed = time.time() - start_time
    print("Thoi gian tinh SHAP: {:.1f} giay ({:.1f} phut)\n".format(
        elapsed, elapsed / 60))
    
    # Xu ly SHAP values (lay class 1 - Bestseller)
    if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]
    elif isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw
    
    print("SHAP values shape: {}".format(shap_values.shape))
    
    # Tao long format DataFrame
    print("\nTao long format DataFrame...")
    records = []
    
    for i in range(len(X)):
        book_id = df.iloc[i].get('id', i)
        book_name = df.iloc[i].get('name', 'Unknown')
        
        # Loai bo ky tu xuong dong va cac ky tu dac biet trong name
        # De tranh loi khi import vao Power BI
        book_name = str(book_name).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Loai bo dau ngoac kep
        book_name = book_name.replace('"', '')
        # Loai bo nhieu khoang trang lien tiep
        book_name = ' '.join(book_name.split())
        
        # Tinh tong SHAP values tuyet doi cho sach nay
        total_shap_abs = np.sum(np.abs(shap_values[i]))
        
        for j, feature in enumerate(feature_cols):
            shap_val = shap_values[i, j]
            feature_val = X.iloc[i, j]
            
            # Tinh % dong gop
            if total_shap_abs > 0:
                shap_percent = 100 * np.abs(shap_val) / total_shap_abs
            else:
                shap_percent = 0
            
            # Map feature name to Vietnamese
            feature_display = FEATURE_NAME_MAP.get(feature, feature)
            
            records.append({
                'id': book_id,
                'name': book_name,
                'feature_name': feature_display,
                'feature_name_original': feature,
                'shap_value': shap_val,
                'feature_value': feature_val,
                'shap_value_percent': shap_percent
            })
        
        # In progress
        if (i + 1) % 500 == 0:
            print("  - Da xu ly {} / {} sach".format(i + 1, len(X)))
    
    # Tao DataFrame
    result_df = pd.DataFrame(records)
    
    # Sap xep
    result_df = result_df.sort_values(['id', 'shap_value_percent'], 
                                      ascending=[True, False])
    
    # Luu CSV
    print("\nLuu CSV...")
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print("Da luu: {}".format(output_path))
    
    # Thong ke
    print("\n" + "=" * 70)
    print("THONG KE")
    print("=" * 70)
    print("\nTong so dong: {}".format(len(result_df)))
    print("Tong so cot: {}".format(len(result_df.columns)))
    print("\nFirst 5 rows:")
    print(result_df.head(10).to_string())
    
    # Phan tich top features
    print("\n\nTop 10 Feature-sach co SHAP value lon nhat:")
    print(result_df.nlargest(10, 'shap_value_percent')[
        ['id', 'name', 'feature_name', 'shap_value_percent']].to_string())
    
    return result_df


def main():
    """
    Ham chinh
    """
    print("=" * 70)
    print("EXPORT SHAP VALUES FOR POWER BI")
    print("=" * 70)
    print()
    
    # Buoc 1: Uong tinh thoi gian
    should_proceed = estimate_runtime()
    
    if not should_proceed:
        print("\n" + "=" * 70)
        response = input("\nChay toan bo co the mat 1-2 phut. Ban co muon tiep tuc? (y/n): ")
        if response.lower() != 'y':
            print("Da huy.")
            return
    
    # Buoc 2: Xuat CSV
    export_shap_to_csv()
    
    print("\n" + "=" * 70)
    print("HOAN THANH!")
    print("=" * 70)
    print("\nFile da tao: data/clean/shap_values.csv")
    print("Co the nhap vao Power BI de tao dashboard.")


if __name__ == "__main__":
    main()