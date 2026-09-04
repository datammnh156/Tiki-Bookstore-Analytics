"""
Phan tich mo ta (EDA) du lieu sach Tiki da lam sach.
"""

import sys
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix encoding cho Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sns.set_style("whitegrid")


def phan_tich(file_path: str):
    df = pd.read_csv(file_path)

    # Loai trung lap truoc khi phan tich
    truoc = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"Da loai {truoc - len(df)} dong trung lap. Con lai {len(df)} san pham.\n")

    cot_category = "category_name" if "category_name" in df.columns else "category_id"

    # Neu chua co has_rating thi tu tao
    if "has_rating" not in df.columns:
        df["has_rating"] = (df["rating_average"] > 0).astype(int)

    # Tinh lai nhan is_bestseller (top 30% quantity_sold theo category)
    df["threshold"] = df.groupby(cot_category)["quantity_sold"].transform(
        lambda x: x.quantile(0.7)
    )
    df["is_bestseller"] = (df["quantity_sold"] >= df["threshold"]).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Phan bo gia theo category
    sns.boxplot(data=df, x=cot_category, y="price", ax=axes[0, 0])
    axes[0, 0].set_title("Phan bo gia theo danh muc")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # 2. Phan bo quantity_sold theo category (log scale)
    sns.boxplot(data=df, x=cot_category, y="quantity_sold", ax=axes[0, 1])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Phan bo quantity_sold theo danh muc (log scale)")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # 3. Tuong quan discount_rate vs quantity_sold
    sns.scatterplot(data=df, x="discount_rate", y="quantity_sold",
                     hue="is_bestseller", alpha=0.5, ax=axes[0, 2])
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Giam gia % vs So luong ban")

    # 4. Phan bo rating_average
    sns.histplot(df[df["has_rating"] == 1]["rating_average"], bins=20, ax=axes[1, 0])
    axes[1, 0].set_title("Phan bo rating (chi sach da co danh gia)")

    # 5. Ty le bestseller theo category
    ty_le = df.groupby(cot_category)["is_bestseller"].mean() * 100
    ty_le.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Ty le % bestseller theo danh muc")
    axes[1, 1].set_ylabel("% bestseller")
    axes[1, 1].tick_params(axis="x", rotation=30)

    # 6. Ma tran tuong quan
    cot_so = [c for c in ["price", "discount_rate", "rating_average",
                           "review_count", "quantity_sold",
                           "has_rating"]
              if c in df.columns]
    corr = df[cot_so].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
    axes[1, 2].set_title("Ma tran tuong quan")

    plt.tight_layout()
    
    # Tao folder outputs/charts neu chua co
    os.makedirs("outputs/charts", exist_ok=True)
    
    plt.savefig("outputs/charts/eda_tong_quan.png", dpi=150)
    print("Da luu bieu do tong quan vao outputs/charts/eda_tong_quan.png")

    # In them vai so lieu quan trong
    print("\n--- Tuong quan voi quantity_sold ---")
    print(corr["quantity_sold"].sort_values(ascending=False))

    print("\n--- Min/Median/Max quantity_sold theo category ---")
    print(df.groupby(cot_category)["quantity_sold"].agg(["min", "median", "max"]))

    print("\n--- Bestseller rate: co rating vs khong co rating ---")
    print(df.groupby("has_rating")["is_bestseller"].mean() * 100)


if __name__ == "__main__":
    default_path = "data/clean/tiki_books_cleaned.csv"
    
    if not os.path.exists(default_path):
        default_path = "../../data/clean/tiki_books_cleaned.csv"
    
    if os.path.exists(default_path):
        print("[EDA] PHAN TICH MO TA - DESCRIPTIVE ANALYSIS")
        print("="*80)
        print(f"Dang phan tich file: {default_path}\n")
        phan_tich(default_path)
        print("\n" + "="*80)
        print("HOAN THANH PHAN TICH ")
        print("\n" + "="*80)

    else:
        print("="*80)
        print("Khong tim thay file CSV")
        print("="*80)
        print("Duong dan tim kiem:")
        print("  1. data/clean/tiki_books_cleaned.csv")
        print("  2. ../../data/clean/tiki_books_cleaned.csv")
        print("\nVui long chay tu thu muc goc project: D:\\DoAnNganh")
