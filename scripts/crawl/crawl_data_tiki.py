#crawl data_tiki.py
import urllib.request
import urllib.parse
import json
import csv
import time
from datetime import datetime

BOOK_SUBCATEGORIES = {
    316: "Sách tiếng Việt",
    320: "Sách nước ngoài",
    1084: "Truyện tranh, Manga",
    7358: "Sách giáo khoa",
}

PRODUCTS_PER_CATEGORY = 1000  # đổi số này nếu muốn cào nhiều/ít hơn
DELAY_BETWEEN_REQUESTS = 1.8  
PAGE_SIZE = 40  # số sản phẩm mỗi lần gọi API (giới hạn thường gặp của Tiki)

LISTING_API = "https://tiki.vn/api/personalish/v1/blocks/listings"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

CAC_FIELD_CAN_LAY = [
    "id", "name", "price", "original_price", "discount_rate",
    "rating_average", "review_count", "quantity_sold", "favourite_count",
    "badges_new",
]

#Lay so luong san pham da ban
def get_quantity_sold(sp):
    quantity_sold = sp.get("quantity_sold")

    if type(quantity_sold) == dict:
        return quantity_sold.get("value", 0)

    return quantity_sold if quantity_sold else 0

def crawl_1_category(category_id: int, category_name: str, so_luong: int):
    ket_qua = []
    page = 1
    da_thay_id = set()

    while len(ket_qua) < so_luong:
        params = {
            "limit": PAGE_SIZE,
            "page": page,
            "category": category_id,
        }
        try:
#Goi API de lay danh sach san pham
            # Build URL with query parameters
            url = LISTING_API + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=HEADERS)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                response_data = response.read().decode('utf-8')
                json_data = json.loads(response_data)
                data = json_data.get("data", [])
        except Exception as e:
            print(f"  Lỗi ở page {page} của '{category_name}': {e}")
            break
#Neu khong con san pham thi dung
        if not data:
            print(f"  Hết sản phẩm ở '{category_name}' (page {page}), dừng sớm.")
            break
#Duyet de lay thong tin san pham
        for sp in data:
            sp_id = sp.get("id")
            if sp_id in da_thay_id:
                continue
            da_thay_id.add(sp_id)
#Lay cac field can thiet
            dong = {k: sp.get(k) for k in CAC_FIELD_CAN_LAY}
#Lay so luong da ban va category
            dong["quantity_sold"] = get_quantity_sold(sp)
            dong["category_id"] = category_id
            dong["category_name"] = category_name
            ket_qua.append(dong)
#Neu da du so luong san pham thi dung
            if len(ket_qua) >= so_luong:
                break

        print(f"  [{category_name}] Đã crawl {len(ket_qua)}/{so_luong}")
        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)
#Kiem tra neu page > 100 de phong truong hop API tra ve du lieu bat thuong gay vong lap
        if page > 100:  # phòng vòng lặp vô hạn nếu API trả dữ liệu bất thường
            break

    return ket_qua


def main():
    all_products = []

    # Crawl tung danh muc sach
    for category_id, category_name in BOOK_SUBCATEGORIES.items():
        print(f"\nDang crawl: {category_name} (ID: {category_id})")

        products = crawl_1_category(
            category_id,
            category_name,
            PRODUCTS_PER_CATEGORY
        )

        # Gop ket qua vao danh sach chung
        all_products.extend(products)

    # Khong co du lieu thi dung
    if not all_products:
        print("Khong lay duoc san pham nao.")
        return

    # Tao ten file theo thoi gian de tranh ghi de
    file_name = f"tiki_books_{datetime.now():%Y%m%d_%H%M%S}.csv"

    # Ghi du lieu ra file CSV
    with open(file_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_products[0].keys()
        )

        writer.writeheader()        # Ghi ten cot
        writer.writerows(all_products)  # Ghi toan bo du lieu

    print(f"\nHoan tat. Tong so san pham: {len(all_products)}")
    print(f"File da luu: {file_name}")


if __name__ == "__main__":
    main()