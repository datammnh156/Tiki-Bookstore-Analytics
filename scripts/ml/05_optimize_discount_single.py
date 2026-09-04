# Mô phỏng và đề xuất mức giảm giá cho 1 sản phẩm
# LƯU Ý: Đây là mô phỏng dựa trên model dự đoán, không phải phân tích nhân quả.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import io
import pickle

# Fix encoding khi chạy trên Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def simulate_optimal_discount(product_id, df, model, scaler, feature_cols):
    """Mô phỏng và đề xuất mức giảm giá cho 1 sản phẩm."""

    # Tìm sách theo ID
    book = df[df['id'] == product_id]

    if len(book) == 0:
        print(f"❌ Không tìm thấy sách với ID: {product_id}")
        return None

    book = book.iloc[0]

    # In thông tin sách
    print("\n" + "=" * 70)
    print("THÔNG TIN SÁCH")
    print("=" * 70)
    print(f"ID               : {book['id']}")
    print(f"Tên              : {book['name'][:60]}...")
    print(f"Danh mục         : {book['category_name']}")
    print(f"Giá              : {book['price']:,.0f} VND")
    print(f"Discount hiện tại: {book['discount_rate']:.1f}%")
    print(f"Rating           : {book['rating_average']:.1f}/5.0")
    print(f"Số lượng bán     : {book['quantity_sold']:,} cuốn")
    print(f"Bestseller       : {'Có' if book['is_bestseller'] == 1 else 'Không'}")
    print("=" * 70)

    # Giữ nguyên các feature khác, chỉ thay đổi discount_rate
    base_features = {}
    for col in feature_cols:
        if col != 'discount_rate':
            base_features[col] = book[col]

    # Các mức giảm được mô phỏng: 0%, 5%, ..., 50%
    discount_rates = np.arange(0, 55, 5)

    # Mô phỏng xác suất Bestseller ở từng mức giảm
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

    # Tìm mức giảm có xác suất cao nhất trong các mức đã mô phỏng
    candidate_idx = results_df['bestseller_probability'].idxmax()
    candidate_discount = results_df.loc[candidate_idx, 'discount_rate']
    candidate_prob = results_df.loc[candidate_idx, 'bestseller_probability']

    # Tính xác suất tại đúng mức giảm hiện tại của sách
    current_discount = book['discount_rate']
    current_features = base_features.copy()
    current_features['discount_rate'] = current_discount

    X_current = pd.DataFrame([current_features], columns=feature_cols)
    X_current_scaled = scaler.transform(X_current)
    current_prob = model.predict_proba(X_current_scaled)[0, 1]

    # Chỉ đề xuất thay đổi nếu xác suất tăng ít nhất 5 điểm phần trăm
    improvement_threshold = 0.05
    improvement = candidate_prob - current_prob

    if improvement >= improvement_threshold:
        recommended_discount = candidate_discount
        recommended_prob = candidate_prob
        recommend_change = True
    else:
        recommended_discount = current_discount
        recommended_prob = current_prob
        improvement = 0
        recommend_change = False

    # In bảng kết quả mô phỏng
    print("\nKẾT QUẢ MÔ PHỎNG")
    print("-" * 70)
    print(f"{'Discount':>10}  {'Xác suất':>12}  {'Trend':>10}  {'Ghi chú':<20}")
    print("-" * 70)

    prev_prob = None

    for _, row in results_df.iterrows():
        discount = row['discount_rate']
        prob = row['bestseller_probability']

        # So sánh với mức mô phỏng trước đó
        if prev_prob is None:
            trend = "-"
        elif prob > prev_prob + 0.01:
            trend = "↑"
        elif prob < prev_prob - 0.01:
            trend = "↓"
        else:
            trend = "→"

        note = ""
        if recommend_change and discount == recommended_discount:
            note = "← ĐỀ XUẤT"

        print(f"{discount:>9.0f}%  {prob:>11.1%}  {trend:>10}  {note:<20}")
        prev_prob = prob

    print("-" * 70)

    # In kết quả đề xuất
    print("\nKẾT QUẢ ĐỀ XUẤT:")
    print(f"  Discount hiện tại : {current_discount:>5.1f}% → Xác suất: {current_prob:>6.1%}")

    if recommend_change:
        print(f"  Discount đề xuất  : {recommended_discount:>5.0f}% → Xác suất: {recommended_prob:>6.1%}")
        print(f"  Cải thiện         : {improvement:>+6.1%}")
    else:
        print(f"  Đề xuất           : GIỮ NGUYÊN {current_discount:.1f}%")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Đường mô phỏng
    ax.plot(
        results_df['discount_rate'],
        results_df['bestseller_probability'] * 100,
        marker='o',
        markersize=5,
        linewidth=2,
        color='#1f77b4',
        label='Mô phỏng',
        zorder=2
    )

    # Điểm hiện tại (tròn, màu cam)
    ax.scatter(
        current_discount,
        current_prob * 100,
        s=150,
        marker='o',
        color='#ff7f0e',
        edgecolors='black',
        linewidth=1.5,
        zorder=5,
        label='Hiện tại'
    )

    # Annotation cho hiện tại
    ax.annotate(
        f'Hiện tại: {current_discount:.0f}% ({current_prob:.1%})',
        xy=(current_discount, current_prob * 100),
        xytext=(10, 15),
        textcoords='offset points',
        fontsize=9,
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ff7f0e', alpha=0.2),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#ff7f0e', lw=1)
    )

    # Điểm đề xuất (chỉ khi recommend_change == True)
    if recommend_change:
        ax.scatter(
            recommended_discount,
            recommended_prob * 100,
            s=150,
            marker='o',
            color='#2ca02c',
            edgecolors='black',
            linewidth=1.5,
            zorder=5,
            label='Đề xuất'
        )

        # Annotation cho đề xuất
        ax.annotate(
            f'Đề xuất: {recommended_discount:.0f}% ({recommended_prob:.1%})',
            xy=(recommended_discount, recommended_prob * 100),
            xytext=(10, -25),
            textcoords='offset points',
            fontsize=9,
            ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ca02c', alpha=0.2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#2ca02c', lw=1)
        )

    # Cấu hình trục
    ax.set_xlabel('Mức giảm giá (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Xác suất dự đoán Bestseller (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(discount_rates)
    ax.set_xticklabels([f'{int(x)}' for x in discount_rates])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Tiêu đề
    book_name = book['name'][:55] + ('...' if len(book['name']) > 55 else '')
    ax.set_title(f'Mô phỏng mức giảm giá\n{book_name}', fontsize=12, fontweight='bold', pad=15)

    # Grid nhẹ
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.show()

    # Trả kết quả để dùng cho phần tổng kết
    return {
        'results': results_df,
        'recommended_discount': recommended_discount,
        'recommended_probability': recommended_prob,
        'current_probability': current_prob,
        'improvement': improvement,
        'recommend_change': recommend_change,
        'book_info': book
    }


def run_discount_optimization_demo(df, model, scaler, feature_cols, sample_ids=None):
    """Chạy demo mô phỏng cho 3 sản phẩm."""

    print("\n" + "=" * 70)
    print("DEMO: MÔ PHỎNG VÀ ĐỀ XUẤT MỨC GIẢM GIÁ")
    print("=" * 70)

    # Nếu không truyền ID thì dùng 3 sách mẫu cố định
    # Chọn dựa trên kết quả thực tế của model:
    # - 2 sách có recommend_change = True (improvement >= 5%)
    # - 1 sách có recommend_change = False (improvement < 5%)
    if sample_ids is None:
        sample_ids = [
            279437167,  # Improvement: 7.4% (23% → 40%) - recommend_change=True
            278777851,  # Improvement: 18.1% (39% → 20%) - recommend_change=True
            278856619   # Improvement: 0.0% (giữ nguyên 20%) - recommend_change=False
        ]

    results_summary = []

    for idx, product_id in enumerate(sample_ids, 1):
        print(f"\n{'=' * 70}")
        print(f"SÁCH {idx}/{len(sample_ids)}")

        result = simulate_optimal_discount(product_id, df, model, scaler, feature_cols)

        if result:
            results_summary.append({
                'product_id': product_id,
                'name': result['book_info']['name'][:40],
                'current_discount': result['book_info']['discount_rate'],
                'recommended_discount': result['recommended_discount'],
                'recommended_probability': result['recommended_probability'],
                'improvement': result['improvement'],
                'recommend_change': result['recommend_change'],
                'is_bestseller': result['book_info']['is_bestseller']
            })

    # Tổng kết 3 sách
    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)

    summary_df = pd.DataFrame(results_summary)

    for idx, row in summary_df.iterrows():
        print(f"\nSách {idx + 1}: {row['name']}...")
        print(f"  ID          : {row['product_id']}")
        print(f"  Bestseller  : {'Có' if row['is_bestseller'] == 1 else 'Không'}")

        if row['recommend_change']:
            print(f"  Discount    : {row['current_discount']:.1f}% → {row['recommended_discount']:.0f}%")
            print(f"  Xác suất    : {row['recommended_probability']:.1%}")
            print(f"  Cải thiện   : {row['improvement']:+.1%}")
        else:
            print(f"  Discount    : Giữ nguyên {row['current_discount']:.1f}%")
            print(f"  Xác suất    : {row['recommended_probability']:.1%}")

    print("\n" + "=" * 70 + "\n")

    return summary_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ĐANG LOAD DỮ LIỆU VÀ MODEL")
    print("=" * 70)

    try:
        # Load dữ liệu
        print("\nĐang load dữ liệu...")
        df = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
        print(f"✓ Đã load {len(df):,} sách")

        # Load model và scaler
        print("\nĐang load model...")
        model = pickle.load(open('models/bestseller_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        print("✓ Đã load Random Forest model & scaler")

        # Tạo biến giả cho danh mục sách
        print("\nĐang chuẩn bị features...")
        category_dummies = pd.get_dummies(df['category_name'], prefix='cat', dtype=int)
        df = pd.concat([df, category_dummies], axis=1)

        # Features phải giống với lúc train model
        feature_cols = ['price', 'discount_rate', 'rating_average', 'has_rating']
        cat_cols = [col for col in df.columns if col.startswith('cat_')]
        feature_cols.extend(cat_cols)

        print(f"✓ Chuẩn bị xong {len(feature_cols)} features")
        print("\n" + "=" * 70)
        print("SẴN SÀNG!")
        print("=" * 70)

        # Chạy demo 3 sách
        results = run_discount_optimization_demo(df, model, scaler, feature_cols)

    except FileNotFoundError as e:
        print(f"\n❌ LỖI: Không tìm thấy file: {e}")
        print("Vui lòng kiểm tra lại đường dẫn dữ liệu, model và scaler.")

    except Exception as e:
        print(f"\n❌ LỖI: {e}")