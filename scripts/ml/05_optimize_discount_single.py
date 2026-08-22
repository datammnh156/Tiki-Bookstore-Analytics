# Mô phỏng % giảm giá tối ưu cho 1 sản phẩm
# LƯU Ý: Đây là mô phỏng dựa trên model dự đoán (correlational), không phải phân tích nhân quả (causal).
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import io

# Fix encoding cho Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def simulate_optimal_discount(product_id, df, model, scaler, feature_cols):
    """
    Mô phỏng % giảm giá tối ưu cho 1 sản phẩm
    """
    
    # Tìm sách trong dataset
    book = df[df['id'] == product_id]
    
    if len(book) == 0:
        print(f"❌ Không tìm thấy sách với ID: {product_id}")
        return None
    
    book = book.iloc[0]
    
    # In thông tin sách
    print("\n" + "="*70)
    print("THÔNG TIN SÁCH")
    print("="*70)
    print(f"ID               : {book['id']}")
    print(f"Tên              : {book['name'][:60]}...")
    print(f"Danh mục         : {book['category_name']}")
    print(f"Giá              : {book['price']:,.0f} VND")
    print(f"Discount hiện tại: {book['discount_rate']:.1f}%")
    print(f"Rating           : {book['rating_average']:.1f}/5.0")
    print(f"Số lượng bán     : {book['quantity_sold']:,} cuốn")
    print(f"Bestseller       : {'Có' if book['is_bestseller'] == 1 else 'Không'}")
    print("="*70)
    
    # Chuẩn bị features gốc
    base_features = {}
    for col in feature_cols:
        if col != 'discount_rate':
            base_features[col] = book[col]
    
    # Tạo dãy discount_rate từ 0% đến 50%
    discount_rates = np.arange(0, 55, 5)
    
    # Mô phỏng xác suất bestseller
    results = []
    for discount in discount_rates:
        features = base_features.copy()
        features['discount_rate'] = discount
        X_sim = pd.DataFrame([features], columns=feature_cols)
        X_sim_scaled = scaler.transform(X_sim)
        proba = model.predict_proba(X_sim_scaled)[0, 1]
        results.append({
            'discount_rate': discount,
            'bestseller_probability': proba
        })
    
    results_df = pd.DataFrame(results)
    
    # Tìm discount tối ưu
    optimal_idx = results_df['bestseller_probability'].idxmax()
    optimal_discount = results_df.loc[optimal_idx, 'discount_rate']
    optimal_prob = results_df.loc[optimal_idx, 'bestseller_probability']
    
    # In bảng kết quả
    print("\nKẾT QUẢ MÔ PHỎNG")
    print("-"*70)
    print(f"{'Discount':>10}  {'Xác suất':>12}  {'Trend':>10}  {'Ghi chú':<20}")
    print("-"*70)
    
    prev_prob = 0
    for _, row in results_df.iterrows():
        discount = row['discount_rate']
        prob = row['bestseller_probability']
        
        # Trend
        if prob > prev_prob + 0.01:
            trend = "↑"
        elif prob < prev_prob - 0.01:
            trend = "↓"
        else:
            trend = "→"
        
        # Ghi chú
        note = ""
        if discount == optimal_discount:
            note = "← TỐI ƯU"
        elif abs(discount - book['discount_rate']) < 2.5:
            note = "← HIỆN TẠI"
        
        print(f"{discount:>9.0f}%  {prob:>11.1%}  {trend:>10}  {note:<20}")
        prev_prob = prob
    
    print("-"*70)
    
    # Tính cải thiện
    current_discount = book['discount_rate']
    current_prob_data = results_df[results_df['discount_rate'] == current_discount]['bestseller_probability'].values
    
    if len(current_prob_data) > 0:
        current_prob = current_prob_data[0]
    else:
        closest_discount = min(discount_rates, key=lambda x: abs(x - current_discount))
        current_prob = results_df[results_df['discount_rate'] == closest_discount]['bestseller_probability'].values[0]
    
    improvement = optimal_prob - current_prob
    
    # In gợi ý
    print(f"\nGỢI Ý TỐI ƯU:")
    print(f"  Discount hiện tại: {current_discount:>5.1f}% → Xác suất: {current_prob:>6.1%}")
    print(f"  Discount tối ưu  : {optimal_discount:>5.0f}% → Xác suất: {optimal_prob:>6.1%}")
    print(f"  Cải thiện        : {improvement:>+6.1%}")
    print()
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    
    plt.plot(results_df['discount_rate'], 
             results_df['bestseller_probability'] * 100,
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    plt.scatter([optimal_discount], [optimal_prob * 100], 
                color='#A23B72', s=200, zorder=5, 
                label=f'Tối ưu: {optimal_discount:.0f}% ({optimal_prob:.1%})')
    
    # Hiển thị điểm "Hiện tại"
    if current_discount <= 50:
        # Nếu discount hiện tại nằm trong range
        current_data = results_df[results_df['discount_rate'] == current_discount]
        if len(current_data) > 0:
            current_idx = current_data.index[0]
            current_prob_pct = results_df.loc[current_idx, 'bestseller_probability'] * 100
            current_prob_display = results_df.loc[current_idx, 'bestseller_probability']
            plt.scatter([current_discount], [current_prob_pct],
                        color='#F18F01', s=200, zorder=5,
                        label=f'Hiện tại: {current_discount:.0f}% ({current_prob_display:.1%})')
        else:
            # Nếu không có chính xác, tìm closest
            closest_discount = min(discount_rates, key=lambda x: abs(x - current_discount))
            closest_data = results_df[results_df['discount_rate'] == closest_discount]
            if len(closest_data) > 0:
                closest_idx = closest_data.index[0]
                closest_prob_pct = results_df.loc[closest_idx, 'bestseller_probability'] * 100
                closest_prob_display = results_df.loc[closest_idx, 'bestseller_probability']
                plt.scatter([closest_discount], [closest_prob_pct],
                            color='#F18F01', s=200, zorder=5, alpha=0.7,
                            label=f'Hiện tại (≈{closest_discount:.0f}%): {closest_prob_display:.1%}')
    else:
        # Nếu discount hiện tại > 50%, tìm closest trong range
        closest_discount = min(discount_rates, key=lambda x: abs(x - current_discount))
        closest_data = results_df[results_df['discount_rate'] == closest_discount]
        if len(closest_data) > 0:
            closest_idx = closest_data.index[0]
            closest_prob_pct = results_df.loc[closest_idx, 'bestseller_probability'] * 100
            closest_prob_display = results_df.loc[closest_idx, 'bestseller_probability']
            plt.scatter([closest_discount], [closest_prob_pct],
                        color='#F18F01', s=200, zorder=5, alpha=0.7,
                        label=f'Hiện tại (≈{closest_discount:.0f}%): {closest_prob_display:.1%}')
    
    plt.xlabel('Mức giảm giá (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Xác suất Bestseller (%)', fontsize=12, fontweight='bold')
    plt.title(f'Mô phỏng: Giảm giá vs Xác suất Bestseller\n{book["name"][:60]}...', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.xticks(discount_rates)
    plt.tight_layout()
    
    # Save plot
    import os
    plot_filename = f'outputs/charts/discount_optimization_{product_id}.png'
    os.makedirs('outputs/charts', exist_ok=True)
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Đã lưu biểu đồ: {plot_filename}\n")
    
    # Force flush output before showing plot
    sys.stdout.flush()
    
    # Show plot
    plt.show()
    plt.close()
    
    return {
        'results': results_df,
        'optimal_discount': optimal_discount,
        'optimal_probability': optimal_prob,
        'book_info': book
    }


def run_discount_optimization_demo(df, model, scaler, feature_cols, sample_ids=None):
    """Chạy demo cho nhiều sản phẩm"""
    
    print("\n" + "="*70)
    print("DEMO: MÔ PHỎNG % GIẢM GIÁ TỐI ƯU")
    print("="*70)
    
    if sample_ids is None:
        bestsellers = df[df['is_bestseller'] == 1].sample(n=1, random_state=42)
        non_bestsellers = df[df['is_bestseller'] == 0].sample(n=2, random_state=42)
        sample_books = pd.concat([bestsellers, non_bestsellers])
        sample_ids = sample_books['id'].tolist()
    
    results_summary = []
    
    for idx, product_id in enumerate(sample_ids, 1):
        print(f"\n{'='*70}")
        print(f"SÁCH {idx}/{len(sample_ids)}")
        
        result = simulate_optimal_discount(product_id, df, model, scaler, feature_cols)
        
        if result:
            results_summary.append({
                'product_id': product_id,
                'name': result['book_info']['name'][:40],
                'current_discount': result['book_info']['discount_rate'],
                'optimal_discount': result['optimal_discount'],
                'optimal_probability': result['optimal_probability'],
                'is_bestseller': result['book_info']['is_bestseller']
            })
    
    # Tổng kết
    print("\n" + "="*70)
    print("TỔNG KẾT")
    print("="*70)
    
    summary_df = pd.DataFrame(results_summary)
    
    for idx, row in summary_df.iterrows():
        print(f"\nSách {idx+1}: {row['name']}...")
        print(f"  ID          : {row['product_id']}")
        print(f"  Bestseller  : {'Có' if row['is_bestseller'] == 1 else 'Không'}")
        print(f"  Discount    : {row['current_discount']:.1f}% → {row['optimal_discount']:.0f}%")
        print(f"  Xác suất    : {row['optimal_probability']:.1%}")
    
    print("\n" + "="*70 + "\n")
    
    return summary_df


if __name__ == "__main__":
    import pickle
    
    print("\n" + "="*70)
    print("ĐANG LOAD DỮ LIỆU VÀ MODEL")
    print("="*70)
    
    try:
        print("\nĐang load dữ liệu...")
        df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
        print(f"✓ Đã load {len(df):,} sách")
        
        print("\nĐang load model...")
        model = pickle.load(open('models/bestseller_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        print("✓ Đã load Random Forest model & scaler")
        
        print("\nĐang chuẩn bị features...")
        category_dummies = pd.get_dummies(df['category_name'], prefix='cat')
        df = pd.concat([df, category_dummies], axis=1)
        
        feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
        cat_cols = [col for col in df.columns if col.startswith('cat_')]
        feature_cols.extend(cat_cols)
        print(f"✓ Chuẩn bị xong {len(feature_cols)} features")
        
        print("\n" + "="*70)
        print("SẴN SÀNG!")
        print("="*70)
        
        results = run_discount_optimization_demo(df, model, scaler, feature_cols)
        
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: Không tìm thấy file: {e}")
        print("Vui lòng chạy: python scripts/ml/01_train_random_forest.py")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")