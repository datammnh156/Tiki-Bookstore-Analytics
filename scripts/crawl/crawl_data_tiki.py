#crawl data_tiki.py - Version 2 (Chi tiết categories)
import urllib.request
import urllib.parse
import json
import csv
import time
from datetime import datetime

# 9 category_id CHÍNH XÁC từ Tiki.vn (đã xác minh thủ công)
BOOK_SUBCATEGORIES = {
    839: "Văn học",
    320: "Sách tiếng Anh",
    846: "Kinh tế",
    393: "Truyện thiếu nhi",
    870: "Kỹ năng sống",
    2321: "Giáo khoa giáo trình",
    1084: "Truyện tranh",
    2320: "Sách tham khảo",
    887: "Ngoại ngữ từ điển",
}

PRODUCTS_PER_CATEGORY = 445  # ~4000 sách tổng cộng ÷ 9 categories
DELAY_BETWEEN_REQUESTS = 1.8  
PAGE_SIZE = 40

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

def get_quantity_sold(sp):
    quantity_sold = sp.get("quantity_sold")
    if type(quantity_sold) == dict:
        return quantity_sold.get("value", 0)
    return quantity_sold if quantity_sold else 0

def crawl_1_category(category_id, category_name, so_luong, 
                     total_target, total_crawled, start_time):
    """Crawl một category với in tiến trình chi tiết mỗi 50 sách"""
    ket_qua = []
    page = 1
    da_thay_id = set()
    last_report = 0

    while len(ket_qua) < so_luong:
        params = {
            "limit": PAGE_SIZE,
            "page": page,
            "category": category_id,
        }
        try:
            url = LISTING_API + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=HEADERS)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                response_data = response.read().decode('utf-8')
                json_data = json.loads(response_data)
                data = json_data.get("data", [])
        except Exception as e:
            print(f"  ❌ Lỗi ở page {page} của '{category_name}': {e}")
            break
        
        if not data:
            print(f"  ⚠️  Hết sản phẩm ở '{category_name}' (page {page})")
            break
        
        for sp in data:
            sp_id = sp.get("id")
            if sp_id in da_thay_id:
                continue
            da_thay_id.add(sp_id)
            
            dong = {k: sp.get(k) for k in CAC_FIELD_CAN_LAY}
            dong["quantity_sold"] = get_quantity_sold(sp)
            dong["category_id"] = category_id
            dong["category_name"] = category_name
            ket_qua.append(dong)
            
            # In tiến trình mỗi 50 sách
            if len(ket_qua) - last_report >= 50 or len(ket_qua) == so_luong:
                last_report = len(ket_qua)
                current_total = total_crawled + len(ket_qua)
                elapsed = int(time.time() - start_time)
                rate = current_total / elapsed if elapsed > 0 else 0
                remaining = (total_target - current_total) / rate if rate > 0 else 0
                
                print(f"  📖 Đang crawl {category_name}: {len(ket_qua):,}/{so_luong:,}")
                print(f"  📊 Tổng: {current_total:,}/{total_target:,} | "
                      f"⏱️  Đã trôi: {elapsed//60:02d}:{elapsed%60:02d} | "
                      f"⏳ Còn lại: ~{int(remaining)//60:02d}:{int(remaining)%60:02d}")
            
            if len(ket_qua) >= so_luong:
                break
        
        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        if page > 100:
            break

    return ket_qua


def main():
    print("="*80)
    print("🚀 CRAWL TIKI BOOKS - CHI TIẾT CATEGORIES")
    print("="*80)
    print()
    
    all_products = []
    start_time = time.time()
    total_categories = len(BOOK_SUBCATEGORIES)
    total_target = PRODUCTS_PER_CATEGORY * total_categories
    
    for idx, (category_id, category_name) in enumerate(BOOK_SUBCATEGORIES.items(), 1):
        print(f"[{idx}/{total_categories}] 🔍 Đang crawl: {category_name} (ID: {category_id})")
        print("-" * 80)
        
        products = crawl_1_category(
            category_id,
            category_name,
            PRODUCTS_PER_CATEGORY,
            total_target,
            len(all_products),
            start_time
        )
        
        all_products.extend(products)
        print(f"✅ Hoàn thành {category_name}: {len(products):,} sách")
        print()
    
    if not all_products:
        print("❌ Không lấy được sản phẩm nào.")
        return
    
    # Tạo file CSV
    file_name = f"data/raw/tiki_book_dataset_crawl.csv"
    
    with open(file_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_products[0].keys())
        writer.writeheader()
        writer.writerows(all_products)
    
    # Tính toán thống kê
    total_time = int(time.time() - start_time)
    category_dist = {}
    for product in all_products:
        cat = product.get("category_name", "Unknown")
        category_dist[cat] = category_dist.get(cat, 0) + 1
    
    print("="*80)
    print("📊 TỔNG KẾT")
    print("="*80)
    print(f"✅ Tổng số sách: {len(all_products):,}")
    print()
    print("📈 Phân bố theo category:")
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        pct = count / len(all_products) * 100
        print(f"  {cat:<30} {count:>5,} ({pct:>5.1f}%)")
    
    print()
    print(f"⏱️  Thời gian chạy: {total_time//3600}h {(total_time%3600)//60:02d}m {total_time%60:02d}s")
    print(f"💾 File đã lưu: {file_name}")
    print("="*80)


if __name__ == "__main__":
    main()
