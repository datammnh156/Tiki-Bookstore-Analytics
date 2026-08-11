"""
Chạy mô phỏng discount tối ưu cho TOÀN BỘ sách trong dataset
Chỉ đề xuất thay đổi nếu cải thiện xác suất bestseller >= 5%
Lưu kết quả vào file csv discount_recommendation.csv

CẢNH BÁO VỀ PHƯƠNG PHÁP LUẬN - HẠN CHẾ QUAN TRỌNG:

Script này mô phỏng tác động của discount_rate lên xác suất bestseller bằng cách:
1. Giữ nguyên tất cả features khác (price, rating, category, etc.)
2. Quét discount_rate từ 0% đến 75%
3. Dùng model.predict_proba() để tính xác suất bestseller

⚠️ ĐÂY LÀ SUY LUẬN TƯƠNG QUAN (CORRELATIONAL), KHÔNG PHẢI NHÂN QUẢ (CAUSAL):

- Model được train trên dữ liệu quan sát (observational data), không phải dữ liệu thực nghiệm
- Khi ta thay đổi discount_rate trong mô phỏng, model chỉ cho biết "sách có discount X% 
  thường có xác suất bestseller là Y%" dựa trên pattern đã học
- KHÔNG đảm bảo rằng "nếu ta tăng discount lên X% thì sách SẼ trở thành bestseller"
- Có thể tồn tại confounding factors: VD sách bestseller vốn được giảm giá nhiều hơn 
  (reverse causation), hoặc có yếu tố ẩn khác (brand, marketing campaign) chưa có trong data

HẠN CHẾ:
- Kết quả CHỈ mang tính tham khảo, không nên dùng trực tiếp cho quyết định kinh doanh
- Cần kiểm chứng bằng A/B test thực tế hoặc phương pháp causal inference 
  (propensity score matching, uplift modeling, instrumental variables)
- Model giả định "ceteris paribus" (các yếu tố khác không đổi) - không thực tế trong thực tiễn

KHUYẾN NGHỊ:
- Dùng kết quả để khám phá pattern và đưa ra giả thuyết
- Kiểm chứng bằng thực nghiệm trước khi triển khai thực tế
- Kết hợp với domain knowledge và business context
"""

import pandas as pd
import numpy as np
import pickle
import sys
import io
from datetime import datetime

# Fix encoding cho Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_optimal_discount_batch(books_df, model, scaler, feature_cols):
    """
    Tìm mức discount tối ưu cho nhiều sách cùng lúc (batch prediction)
    Thay vì gọi predict_proba() riêng lẻ, gộp tất cả vào 1 lần predict
    
    Returns:
    --------
    list của dict, mỗi dict chứa: current_discount, current_prob, optimal_discount, 
    optimal_prob, improvement
    """
    
    # Dãy discount từ 0 đến 50%, bước 5%
    discount_rates = np.arange(0, 55, 5)
    
    # Chuẩn bị dữ liệu: tạo 1 bảng lớn với tất cả discount levels
    # Mỗi sách sẽ có 11 hàng (1 cho mỗi discount level)
    all_features_list = []
    book_indices = []  # Track mỗi hàng thuộc sách nào
    
    for book_idx, (_, book) in enumerate(books_df.iterrows()):
        base_features = {}
        for col in feature_cols:
            if col != 'discount_rate':
                base_features[col] = book[col]
        
        # Tạo 11 hàng cho mỗi discount level
        for discount in discount_rates:
            features = base_features.copy()
            features['discount_rate'] = discount
            all_features_list.append(features)
            book_indices.append(book_idx)
    
    # Convert to DataFrame
    X_all = pd.DataFrame(all_features_list, columns=feature_cols)
    
    # Scale toàn bộ một lần
    X_all_scaled = scaler.transform(X_all)
    
    # Predict_proba cho tất cả cùng lúc (batch)
    all_probas = model.predict_proba(X_all_scaled)[:, 1]
    
    # Xử lý kết quả
    results = []
    n_discounts = len(discount_rates)
    
    for book_idx, (_, book) in enumerate(books_df.iterrows()):
        # Lấy xác suất cho sách này (11 mục)
        start_idx = book_idx * n_discounts
        end_idx = start_idx + n_discounts
        probabilities = all_probas[start_idx:end_idx]
        
        # Tìm optimal
        optimal_idx = np.argmax(probabilities)
        optimal_discount = discount_rates[optimal_idx]
        optimal_prob = probabilities[optimal_idx]
        
        # Tìm xác suất hiện tại
        current_discount = book['discount_rate']
        closest_idx = np.argmin(np.abs(discount_rates - current_discount))
        current_prob = probabilities[closest_idx]
        
        # Tính improvement
        improvement = optimal_prob - current_prob
        
        results.append({
            'current_discount': current_discount,
            'current_probability': current_prob,
            'optimal_discount': optimal_discount,
            'optimal_probability': optimal_prob,
            'improvement': improvement
        })
    
    return results


def batch_optimize_all_books(df, model, scaler, feature_cols, improvement_threshold=0.05):
    """
    Chạy mô phỏng cho toàn bộ sách trong dataset với BATCH PREDICTION
    Xử lý tất cả sách cùng lúc thay vì từng sách một
    
    Parameters:
    -----------
    improvement_threshold : float
        Ngưỡng cải thiện tối thiểu (default 5% = 0.05) để đề xuất thay đổi
    """
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU MÔ PHỎNG CHO TOÀN BỘ DATASET (BATCH MODE)")
    print("="*70)
    print(f"Tổng số sách: {len(df):,}")
    print(f"Ngưỡng cải thiện tối thiểu: {improvement_threshold:.1%}")
    print(f"Thời gian bắt đầu: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70 + "\n")
    
    # Sử dụng batch prediction cho TẤT CẢ sách cùng lúc
    print("⚡ Đang tính toán optimal discount cho tất cả sách (batch)...")
    batch_results = find_optimal_discount_batch(df, model, scaler, feature_cols)
    print("✅ Hoàn thành batch prediction!")
    
    # Xử lý kết quả
    results = []
    for idx, (book, result) in enumerate(zip(df.itertuples(), batch_results)):
        # Kiểm tra có nên đổi không (improvement >= threshold)
        recommend_change = result['improvement'] >= improvement_threshold
        
        # Nếu không đề xuất đổi, giữ nguyên discount hiện tại
        if not recommend_change:
            result['optimal_discount'] = result['current_discount']
            result['optimal_probability'] = result['current_probability']
            result['improvement'] = 0.0
        
        # Lưu kết quả
        results.append({
            'id': book.id,
            'name': book.name,
            'category_name': book.category_name,
            'current_discount': result['current_discount'],
            'current_probability': result['current_probability'],
            'optimal_discount': result['optimal_discount'],
            'optimal_probability': result['optimal_probability'],
            'improvement': result['improvement'],
            'recommend_change': recommend_change
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH MÔ PHỎNG!")
    print("="*70)
    print(f"Thời gian kết thúc: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70 + "\n")
    
    return results_df


def print_summary_statistics(results_df):
    """
    In thống kê tổng quan
    """
    
    print("\n" + "="*70)
    print("📊 THỐNG KÊ TỔNG QUAN")
    print("="*70)
    
    total = len(results_df)
    recommend_count = results_df['recommend_change'].sum()
    recommend_pct = (recommend_count / total) * 100
    
    print(f"\n1. TỔNG QUAN:")
    print(f"   - Tổng số sách: {total:,}")
    print(f"   - Số sách được đề xuất đổi discount: {recommend_count:,} ({recommend_pct:.1f}%)")
    print(f"   - Số sách giữ nguyên: {total - recommend_count:,} ({100 - recommend_pct:.1f}%)")
    
    # Thống kê nhóm được đề xuất đổi
    recommended = results_df[results_df['recommend_change'] == True]
    
    if len(recommended) > 0:
        avg_improvement = recommended['improvement'].mean()
        median_improvement = recommended['improvement'].median()
        max_improvement = recommended['improvement'].max()
        
        print(f"\n2. NHÓM ĐƯỢC ĐỀ XUẤT ĐỔI DISCOUNT ({len(recommended):,} sách):")
        print(f"   - Mức cải thiện trung bình: {avg_improvement:.1%}")
        print(f"   - Mức cải thiện median: {median_improvement:.1%}")
        print(f"   - Mức cải thiện cao nhất: {max_improvement:.1%}")
        
        # Phân bố theo category
        print(f"\n3. PHÂN BỐ THEO CATEGORY (nhóm được đề xuất):")
        category_stats = recommended.groupby('category_name').agg({
            'id': 'count',
            'improvement': 'mean'
        }).round(3)
        category_stats.columns = ['Số sách', 'Cải thiện TB']
        category_stats = category_stats.sort_values('Số sách', ascending=False)
        print(category_stats)
        
        # Top 10 sách có improvement cao nhất
        print(f"\n4. TOP 10 SÁCH CÓ CẢI THIỆN CAO NHẤT:")
        top10 = recommended.nlargest(10, 'improvement')[
            ['name', 'category_name', 'current_discount', 'optimal_discount', 'improvement']
        ]
        
        for idx, row in top10.iterrows():
            print(f"\n   {row['name'][:60]}...")
            print(f"   - Category: {row['category_name']}")
            print(f"   - Discount hiện tại: {row['current_discount']:.0f}%")
            print(f"   - Discount tối ưu: {row['optimal_discount']:.0f}%")
            print(f"   - Cải thiện: {row['improvement']:.1%}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🔧 ĐANG TẢI DỮ LIỆU VÀ MODEL...")
    print("="*70)
    
    # Load data
    df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
    print(f"✅ Đã load data: {len(df):,} sách")
    
    # Create one-hot encoding for category
    category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    print(f"✅ Tạo xong {len(category_dummies.columns)} cột category")
    
    # Load model & scaler
    with open('models/bestseller_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Đã load model")
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Đã load scaler")
    
    # Define features
    feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    feature_cols.extend(cat_cols)
    print(f"✅ Sử dụng {len(feature_cols)} features")
    
    # Run batch optimization với ngưỡng 5%
    results_df = batch_optimize_all_books(
        df=df, 
        model=model, 
        scaler=scaler, 
        feature_cols=feature_cols,
        improvement_threshold=0.05  # 5%
    )
    
    # Save results
    output_path = 'data/clean/discount_recommendations.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 Đã lưu kết quả vào: {output_path}")
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    print("="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70 + "\n")
    