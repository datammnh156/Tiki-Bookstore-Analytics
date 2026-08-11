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
    
    Parameters:
    -----------
    product_id : int
        ID của sản phẩm cần mô phỏng
    df : DataFrame
        DataFrame chứa dữ liệu sách (đã clean)
    model : sklearn model
        Model đã train (Random Forest)
    scaler : StandardScaler
        Scaler đã fit từ training data
    feature_cols : list
        List tên các features theo đúng thứ tự
        
    Returns:
    --------
    dict : Kết quả bao gồm bảng dữ liệu và % giảm giá tối ưu
    """
    
    # Tìm sách trong dataset
    book = df[df['id'] == product_id]
    
    if len(book) == 0:
        print(f"❌ Không tìm thấy sách với ID: {product_id}")
        return None
    
    book = book.iloc[0]
    
    # In thông tin sách
    print("\n" + "="*70)
    print(f"📚 THÔNG TIN SÁCH")
    print("="*70)
    print(f"ID: {book['id']}")
    print(f"Tên: {book['name'][:80]}...")
    print(f"Danh mục: {book['category_name']}")
    print(f"Giá gốc: {book['price']:,.0f} VND")
    print(f"Giảm giá hiện tại: {book['discount_rate']:.1f}%")
    print(f"Rating: {book['rating_average']:.1f}")
    print(f"Số lượng đã bán: {book['quantity_sold']:,.0f}")
    print(f"Bestseller thực tế: {'✅ Có' if book['is_bestseller'] == 1 else '❌ Không'}")
    print("="*70 + "\n")
    
    # Chuẩn bị features gốc của sách (không bao gồm discount_rate)
    base_features = {}
    for col in feature_cols:
        if col != 'discount_rate':
            base_features[col] = book[col]
    
    # Tạo dãy discount_rate từ 0% đến 50%
    discount_rates = np.arange(0, 55, 5)  # 0, 5, 10, ..., 50
    
    # Mô phỏng xác suất bestseller cho mỗi mức discount
    results = []
    
    for discount in discount_rates:
        # Tạo feature vector cho mức discount này
        features = base_features.copy()
        features['discount_rate'] = discount
        
        # Tạo DataFrame với đúng thứ tự columns
        X_sim = pd.DataFrame([features], columns=feature_cols)
        
        # Scale features (dùng scaler đã fit từ trước)
        X_sim_scaled = scaler.transform(X_sim)
        
        # Dự đoán xác suất (class 1 = bestseller)
        proba = model.predict_proba(X_sim_scaled)[0, 1]
        
        results.append({
            'discount_rate': discount,
            'bestseller_probability': proba
        })
    
    # Chuyển thành DataFrame
    results_df = pd.DataFrame(results)
    
    # Tìm mức discount tối ưu (xác suất cao nhất)
    optimal_idx = results_df['bestseller_probability'].idxmax()
    optimal_discount = results_df.loc[optimal_idx, 'discount_rate']
    optimal_prob = results_df.loc[optimal_idx, 'bestseller_probability']
    
    # In bảng kết quả
    print("📊 KẾT QUẢ MÔ PHỎNG:")
    print("-" * 70)
    print(f"{'Giảm giá (%)':<15} {'Xác suất Bestseller':<25} {'Ghi chú':<30}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        discount = row['discount_rate']
        prob = row['bestseller_probability']
        
        # Highlight optimal
        if discount == optimal_discount:
            note = "⭐ TỐI ƯU"
        elif discount == book['discount_rate']:
            note = "📍 HIỆN TẠI"
        else:
            note = ""
        
        print(f"{discount:>13.0f}%  {prob:>22.1%}  {note:<30}")
    
    print("-" * 70)
    print(f"\n✅ GỢI Ý GIẢM GIÁ TỐI ƯU: {optimal_discount:.0f}%")
    print(f"   → Xác suất bestseller: {optimal_prob:.1%}")
    
    # So sánh với giảm giá hiện tại
    current_discount = book['discount_rate']
    current_prob_data = results_df[results_df['discount_rate'] == current_discount]['bestseller_probability'].values
    
    if len(current_prob_data) > 0:
        current_prob = current_prob_data[0]
        improvement = optimal_prob - current_prob
        print(f"   → Cải thiện so với hiện tại ({current_discount:.0f}%): {improvement:+.1%}")
    else:
        # Nếu current_discount không nằm trong dãy mô phỏng, tìm closest value
        closest_discount = min(discount_rates, key=lambda x: abs(x - current_discount))
        current_prob = results_df[results_df['discount_rate'] == closest_discount]['bestseller_probability'].values[0]
        improvement = optimal_prob - current_prob
        print(f"   → Cải thiện so với hiện tại (≈{closest_discount:.0f}%): {improvement:+.1%}")
    
    print("\n")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    
    # Line plot
    plt.plot(results_df['discount_rate'], 
             results_df['bestseller_probability'] * 100,
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Highlight optimal point
    plt.scatter([optimal_discount], [optimal_prob * 100], 
                color='#A23B72', s=200, zorder=5, 
                label=f'Tối ưu: {optimal_discount:.0f}% (xác suất {optimal_prob:.1%})')
    
    # Highlight current point if in range
    if current_discount <= 50:
        current_data = results_df[results_df['discount_rate'] == current_discount]
        if len(current_data) > 0:
            current_idx = current_data.index[0]
            current_prob_pct = results_df.loc[current_idx, 'bestseller_probability'] * 100
            current_prob_display = results_df.loc[current_idx, 'bestseller_probability']
            plt.scatter([current_discount], [current_prob_pct],
                        color='#F18F01', s=200, zorder=5,
                        label=f'Hiện tại: {current_discount:.0f}% (xác suất {current_prob_display:.1%})')
        else:
            # Nếu current_discount không nằm trong dãy, dùng closest
            closest_discount = min(discount_rates, key=lambda x: abs(x - current_discount))
            closest_data = results_df[results_df['discount_rate'] == closest_discount]
            if len(closest_data) > 0:
                closest_idx = closest_data.index[0]
                closest_prob_pct = results_df.loc[closest_idx, 'bestseller_probability'] * 100
                closest_prob_display = results_df.loc[closest_idx, 'bestseller_probability']
                plt.scatter([closest_discount], [closest_prob_pct],
                            color='#F18F01', s=200, zorder=5, alpha=0.7,
                            label=f'Hiện tại (≈{closest_discount:.0f}%): {closest_prob_display:.1%}')
    
    # Styling
    plt.xlabel('Mức giảm giá (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Xác suất trở thành Bestseller (%)', fontsize=12, fontweight='bold')
    plt.title(f'Mô phỏng: Giảm giá vs Xác suất Bestseller\n{book["name"][:60]}...', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    # Set x-axis ticks
    plt.xticks(discount_rates)
    
    plt.tight_layout()
    
    # Save plot - Use path from root directory
    import os
    plot_filename = f'outputs/charts/discount_optimization_{product_id}.png'
    
    # Create directory if not exists
    os.makedirs('outputs/charts', exist_ok=True)
    
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"💾 Đã lưu biểu đồ: {plot_filename}\n")
    
    plt.show()
    
    return {
        'results': results_df,
        'optimal_discount': optimal_discount,
        'optimal_probability': optimal_prob,
        'book_info': book
    }


def run_discount_optimization_demo(df, model, scaler, feature_cols, sample_ids=None):
    """
    Chạy demo mô phỏng cho nhiều sản phẩm
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame chứa dữ liệu sách
    model : sklearn model
        Model đã train
    scaler : StandardScaler
        Scaler đã fit
    feature_cols : list
        List tên features
    sample_ids : list, optional
        List các product_id cần test. Nếu None, sẽ random 3 sách
    """
    
    print("\n" + "="*70)
    print("🚀 DEMO: MÔ PHỎNG % GIẢM GIÁ TỐI ƯU")
    print("="*70)
    
    if sample_ids is None:
        # Random 3 sách từ dataset
        # Lấy 1 bestseller và 2 non-bestseller
        bestsellers = df[df['is_bestseller'] == 1].sample(n=1, random_state=42)
        non_bestsellers = df[df['is_bestseller'] == 0].sample(n=2, random_state=42)
        sample_books = pd.concat([bestsellers, non_bestsellers])
        sample_ids = sample_books['id'].tolist()
    
    results_summary = []
    
    for idx, product_id in enumerate(sample_ids, 1):
        print(f"\n{'='*70}")
        print(f"SÁCH {idx}/{len(sample_ids)}")
        print(f"{'='*70}")
        
        result = simulate_optimal_discount(product_id, df, model, scaler, feature_cols)
        
        if result:
            results_summary.append({
                'product_id': product_id,
                'name': result['book_info']['name'][:50],
                'current_discount': result['book_info']['discount_rate'],
                'optimal_discount': result['optimal_discount'],
                'optimal_probability': result['optimal_probability'],
                'is_bestseller': result['book_info']['is_bestseller']
            })
    
    # In tổng kết
    print("\n" + "="*70)
    print("📋 TỔNG KẾT KẾT QUẢ")
    print("="*70)
    
    summary_df = pd.DataFrame(results_summary)
    
    for _, row in summary_df.iterrows():
        print(f"\n📚 {row['name']}...")
        print(f"   ID: {row['product_id']}")
        print(f"   Bestseller thực tế: {'✅ Có' if row['is_bestseller'] == 1 else '❌ Không'}")
        print(f"   Giảm giá hiện tại: {row['current_discount']:.0f}%")
        print(f"   💡 Gợi ý tối ưu: {row['optimal_discount']:.0f}%")
        print(f"   → Xác suất bestseller: {row['optimal_probability']:.1%}")
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH MÔ PHỎNG!")
    print("="*70 + "\n")
    
    return summary_df


# ============================================================================
# MAIN: Chạy demo nếu file được execute trực tiếp
# ============================================================================

if __name__ == "__main__":
    print("\n⚠️  Để chạy demo, import các biến từ file train_model.py:")
    print("   - df (cleaned data)")
    print("   - model (trained Random Forest)")
    print("   - scaler (fitted StandardScaler)")
    print("   - feature_cols (list of feature names)")
    print("\nVí dụ:")
    print("   from discount_optimization import run_discount_optimization_demo")
    print("   results = run_discount_optimization_demo(df, model, scaler, feature_cols)")