"""
Phân tích mô tả (EDA) dữ liệu sách Tiki đã làm sạch.
Chạy: D:\Anaconda\python.exe scripts\eda\descriptive_analysis.py data/clean/tiki_books_cleaned.csv

"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def phan_tich(file_path: str):
    df = pd.read_csv(file_path)

    # Loại trùng lặp trước khi phân tích (phòng trường hợp file chưa de_    duplicate)
    truoc = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"Đã loại {truoc - len(df)} dòng trùng lặp. Còn lại {len(df)} sản phẩm.\n")

    cot_category = "category_name" if "category_name" in df.columns else "category_id"

    # Nếu chưa có has_rating (file gốc chưa qua clean_data.py) thì tự tạo
    if "has_rating" not in df.columns:
        df["has_rating"] = (df["rating_average"] > 0).astype(int)

    # Tính lại nhãn is_bestseller (top 30% quantity_sold theo category)
    df["threshold"] = df.groupby(cot_category)["quantity_sold"].transform(
        lambda x: x.quantile(0.7)
    )
    df["is_bestseller"] = (df["quantity_sold"] >= df["threshold"]).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Phân bố giá theo category
    sns.boxplot(data=df, x=cot_category, y="price", ax=axes[0, 0])
    axes[0, 0].set_title("Phân bố giá theo danh mục")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Phân bố quantity_sold theo category (log scale vì chênh lệch lớn)
    sns.boxplot(data=df, x=cot_category, y="quantity_sold", ax=axes[0, 1])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Phân bố quantity_sold theo danh mục (log scale)")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # 3. Tương quan discount_rate vs quantity_sold
    sns.scatterplot(data=df, x="discount_rate", y="quantity_sold",
                     hue="is_bestseller", alpha=0.5, ax=axes[0, 2])
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Giảm giá % vs Số lượng bán")

    # 4. Phân bố rating_average, chỉ tính nhóm đã có rating (tránh spike ở 0)
    sns.histplot(df[df["has_rating"] == 1]["rating_average"], bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Phân bố rating (chỉ sách đã có đánh giá)")

    # 5. Tỷ lệ bestseller theo category (kiểm tra cân bằng)
    ty_le = df.groupby(cot_category)["is_bestseller"].mean() * 100
    ty_le.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Tỷ lệ % bestseller theo danh mục")
    axes[1, 1].set_ylabel("% bestseller")
    axes[1, 1].tick_params(axis="x", rotation=30)

    # 6. Ma trận tương quan giữa các biến số
    cot_so = [c for c in ["price", "discount_rate", "rating_average",
                           "review_count", "quantity_sold", "favourite_count",
                           "has_rating"]
              if c in df.columns]
    corr = df[cot_so].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
    axes[1, 2].set_title("Ma trận tương quan")

    plt.tight_layout()
    plt.savefig("outputs/charts/eda_tong_quan.png", dpi=150)
    print("Đã lưu biểu đồ tổng quan vào outputs/charts/eda_tong_quan.png")

    # In thêm vài số liệu quan trọng để đọc kèm biểu đồ
    print("\n--- Tương quan với quantity_sold ---")
    print(corr["quantity_sold"].sort_values(ascending=False))

    print("\n--- Min/Median/Max quantity_sold theo category ---")
    print(df.groupby(cot_category)["quantity_sold"].agg(["min", "median", "max"]))

    print("\n--- Bestseller rate: có rating vs không có rating ---")
    print(df.groupby("has_rating")["is_bestseller"].mean() * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python descriptive_analysis.py <file_csv>")
    else:
        phan_tich(sys.argv[1])