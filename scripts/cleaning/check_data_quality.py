#Cach dung py check_data_quality_simple.py <file.csv>
import csv
import sys
from collections import Counter, defaultdict
import os
import io

# Fix UTF-8 encoding 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def kiem_tra(file_path):
    # Doc file CSV
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        columns = reader.fieldnames

    tong_so_dong = len(data)

    print("=" * 70)
    print("1. THONG TIN CHUNG")
    print("=" * 70)
    print(f"So dong: {tong_so_dong}")
    print(f"So cot: {len(columns)}")
    print(f"Cac cot: {columns}\n")

    # ==================================================
    # Kiem tra gia tri thieu
    # ==================================================
    print("=" * 70)
    print("2. KIEM TRA GIA TRI THIEU")
    print("=" * 70)

    for column in columns:
        so_null = 0

        for row in data:
            value = row.get(column)

            if not value:
                so_null += 1

            elif column in ["author_name", "brand_name"] and value == "0":
                so_null += 1

        ty_le = (so_null / tong_so_dong * 100) if tong_so_dong else 0

        print(f"{column:20s}: {so_null:4d} ({ty_le:.1f}%)")

    print()

    # ==================================================
    # Kiem tra ID trung lap
    # ==================================================
    print("=" * 70)
    print("3. KIEM TRA ID TRUNG")
    print("=" * 70)

    if "id" in columns:
        dem_id = Counter(row["id"] for row in data)

        id_trung = {
            product_id: count
            for product_id, count in dem_id.items()
            if count > 1
        }

        if id_trung:
            print(f"Co {len(id_trung)} ID bi trung:")

            for product_id, count in list(id_trung.items())[:5]:
                print(f"  ID {product_id}: {count} lan")

        else:
            print("Khong co ID trung")

    print()

    # ==================================================
    # Thong ke category
    # ==================================================
    print("=" * 70)
    print("4. PHAN BO CATEGORY")
    print("=" * 70)

    category_col = None

    if "category_name" in columns:
        category_col = "category_name"
    elif "category_id" in columns:
        category_col = "category_id"

    if category_col:
        dem_category = Counter(
            row[category_col]
            for row in data
        )

        for category, count in dem_category.most_common():
            print(f"{category}: {count}")

    print()

    # ==================================================
    # Thong ke cac cot so
    # ==================================================
    print("=" * 70)
    print("5. THONG KE CAC COT SO")
    print("=" * 70)

    numeric_columns = [
        "price",
        "original_price",
        "discount_rate",
        "rating_average",
        "review_count",
        "quantity_sold"
    ]

    for column in numeric_columns:

        if column not in columns:
            continue

        values = []

        for row in data:
            try:
                values.append(
                    float(row.get(column, 0) or 0)
                )
            except ValueError:
                continue

        if not values:
            continue

        values.sort()

        print(f"\n{column}")
        print(f"  Min    : {min(values):.1f}")
        print(f"  Median : {values[len(values)//2]:.1f}")
        print(f"  Max    : {max(values):.1f}")
        print(f"  Mean   : {sum(values)/len(values):.1f}")

    print()

    # ==================================================
    # Kiem tra quantity_sold theo category
    # ==================================================
    print("=" * 70)
    print("6. PHAN BO QUANTITY_SOLD")
    print("=" * 70)

    if category_col and "quantity_sold" in columns:

        sales_by_category = defaultdict(list)

        for row in data:
            try:
                qty = float(row.get("quantity_sold", 0) or 0)

                if qty > 0:
                    sales_by_category[
                        row[category_col]
                    ].append(qty)

            except ValueError:
                continue

        for category, sales in sales_by_category.items():

            sales.sort()

            print(f"\n{category}")
            print(f"  Min    : {sales[0]:.0f}")
            print(f"  Median : {sales[len(sales)//2]:.0f}")
            print(f"  Max    : {sales[-1]:.0f}")

    print()

    # ==================================================
    # Thử tính nhãn bestseller
    # ==================================================
    print("=" * 70)
    print("7. THU TINH NHAN BESTSELLER")
    print("=" * 70)

    if category_col and "quantity_sold" in columns:

        threshold_by_category = {}
        sales_by_category = defaultdict(list)

        for row in data:
            try:
                qty = float(row.get("quantity_sold", 0) or 0)

                sales_by_category[
                    row[category_col]
                ].append(qty)

            except ValueError:
                continue

        # Lay moc top 20%
        for category, sales in sales_by_category.items():

            sales.sort()

            vi_tri = int(len(sales) * 0.8)

            threshold_by_category[category] = sales[vi_tri]

        so_bestseller = 0

        for row in data:

            try:
                qty = float(row.get("quantity_sold", 0) or 0)

                threshold = threshold_by_category[
                    row[category_col]
                ]

                if qty >= threshold:
                    so_bestseller += 1

            except ValueError:
                continue

        ty_le = so_bestseller / tong_so_dong

        print(
            f"Ti le bestseller: "
            f"{so_bestseller}/{tong_so_dong} "
            f"({ty_le:.1%})"
        )

    print()

    # ==================================================
    # Ket luan
    # ==================================================
    print("=" * 70)
    print("8. KET LUAN")
    print("=" * 70)

    canh_bao = []

    if tong_so_dong < 500:
        canh_bao.append(
            "So mau qua it (<500)"
        )

    if category_col:
        so_category = len(
            set(row[category_col] for row in data)
        )

        if so_category < 2:
            canh_bao.append(
                "Chi co 1 category"
            )

    if canh_bao:
        for loi in canh_bao:
            print(f"- {loi}")
    else:
        print("Du lieu co ve on de train thu model")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        csv_files = [f for f in os.listdir(".") if f.startswith("tiki_book") and f.endswith(".csv")]
        
        if not csv_files:
            print("Khong tim thay file CSV!")
            print("Cach dung: py check_data_quality.py <file.csv>")
            sys.exit(1)
        
        # Sort by modification time, get latest
        csv_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        file_path = csv_files[0]
        print(f"Dang kiem tra file: {file_path}\n")
    
    if not os.path.exists(file_path):
        print(f"File khong ton tai: {file_path}")
        sys.exit(1)
    
    kiem_tra(file_path)
