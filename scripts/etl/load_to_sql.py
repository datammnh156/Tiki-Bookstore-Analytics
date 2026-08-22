"""
Load dữ liệu từ CSV vào SQL Server database TikiBookStore


"""
import pandas as pd
import pyodbc
import sys
import io
from datetime import datetime

# Fix encoding cho Windows PowerShell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CẤU HÌNH KẾT NỐI SQL SERVER
# ============================================================================

SQL_SERVER = 'DANHTAM\\SERVERONHA'  # SQL Server instance name (domain\instance)
DATABASE = 'TikiBookStore'
DRIVER = '{ODBC Driver 17 for SQL Server}'
SQL_USER = 'sa'
SQL_PASSWORD = '123'

# Tạo connection string (SQL Server Authentication)
connection_string = f'Driver={DRIVER};Server={SQL_SERVER};Database={DATABASE};UID={SQL_USER};PWD={SQL_PASSWORD};'

def get_connection():
    """Tạo kết nối đến SQL Server"""
    try:
        conn = pyodbc.connect(connection_string)
        conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-8')
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        sys.exit(1)

def truncate_tables(conn):
    """Xóa dữ liệu cũ trong các bảng (để reset data)"""
    cursor = conn.cursor()
    try:
        print("\n🗑️  Đang xóa dữ liệu cũ...")
        cursor.execute("TRUNCATE TABLE shap_values")
        cursor.execute("TRUNCATE TABLE book_recommendations")
        cursor.execute("TRUNCATE TABLE discount_recommendations")
        cursor.execute("DELETE FROM books")
        cursor.execute("DELETE FROM dim_category")
        conn.commit()
        print("✓ Đã xóa xong dữ liệu cũ")
    except Exception as e:
        print(f"⚠️  Không thể xóa (có thể bảng trống): {e}")
    finally:
        cursor.close()

def load_categories(conn, df_books):
    """Load danh mục sách vào bảng dim_category"""
    print("\n" + "="*70)
    print("1️⃣  LOAD DIM_CATEGORY")
    print("="*70)
    
    # Lấy danh sách category duy nhất
    categories = df_books[['category_id', 'category_name']].drop_duplicates()
    categories = categories.sort_values('category_id')
    
    cursor = conn.cursor()
    inserted = 0
    
    try:
        for _, row in categories.iterrows():
            cursor.execute(
                "INSERT INTO dim_category (category_id, category_name) VALUES (?, ?)",
                int(row['category_id']),
                row['category_name']
            )
            inserted += 1
        
        conn.commit()
        print(f"✓ Insert thành công: {inserted} danh mục")
        
    except pyodbc.IntegrityError as e:
        print(f"⚠️  ID trùng (bỏ qua): {e}")
        conn.rollback()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
    finally:
        cursor.close()

def load_books(conn, df_books):
    """Load sách vào bảng books"""
    print("\n" + "="*70)
    print("2️⃣  LOAD BOOKS")
    print("="*70)
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    try:
        for _, row in df_books.iterrows():
            try:
                cursor.execute(
                    """INSERT INTO books 
                       (id, category_id, name, price, original_price, discount_rate, 
                        rating_average, review_count, quantity_sold, favourite_count, 
                        has_rating, is_bestseller)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    int(row['id']),
                    int(row['category_id']),
                    row['name'],
                    float(row['price']),
                    float(row['original_price']),
                    float(row['discount_rate']),
                    float(row['rating_average']),
                    int(row['review_count']),
                    int(row['quantity_sold']),
                    int(row['favourite_count']),
                    int(row['has_rating']),
                    int(row['is_bestseller'])
                )
                inserted += 1
            except pyodbc.IntegrityError:
                skipped += 1
                continue
        
        conn.commit()
        print(f"✓ Insert thành công: {inserted} sách")
        if skipped > 0:
            print(f"⚠️  Bỏ qua (ID trùng): {skipped} sách")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
    finally:
        cursor.close()

def load_discount_recommendations(conn, df_discount):
    """Load gợi ý giảm giá"""
    print("\n" + "="*70)
    print("3️⃣  LOAD DISCOUNT_RECOMMENDATIONS")
    print("="*70)
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    try:
        for _, row in df_discount.iterrows():
            try:
                cursor.execute(
                    """INSERT INTO discount_recommendations 
                       (id, current_discount, current_probability, optimal_discount, 
                        optimal_probability, improvement, recommend_change)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    int(row['id']),
                    float(row['current_discount']),
                    float(row['current_probability']),
                    float(row['optimal_discount']),
                    float(row['optimal_probability']),
                    float(row['improvement']),
                    1 if row['recommend_change'] == True else 0
                )
                inserted += 1
            except pyodbc.IntegrityError:
                skipped += 1
                continue
        
        conn.commit()
        print(f"✓ Insert thành công: {inserted} gợi ý")
        if skipped > 0:
            print(f"⚠️  Bỏ qua (ID không tồn tại): {skipped} bản ghi")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
    finally:
        cursor.close()

def load_book_recommendations(conn, df_recommendations):
    """Load gợi ý sách liên quan"""
    print("\n" + "="*70)
    print("4️⃣  LOAD BOOK_RECOMMENDATIONS")
    print("="*70)
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    try:
        for _, row in df_recommendations.iterrows():
            try:
                cursor.execute(
                    """INSERT INTO book_recommendations 
                       (source_id, recommended_id, similarity_score)
                       VALUES (?, ?, ?)""",
                    int(row['source_id']),
                    int(row['recommended_id']),
                    float(row['similarity_score'])
                )
                inserted += 1
            except pyodbc.IntegrityError:
                skipped += 1
                continue
        
        conn.commit()
        print(f"✓ Insert thành công: {inserted} gợi ý")
        if skipped > 0:
            print(f"⚠️  Bỏ qua (FK constraint hoặc duplicate key): {skipped} bản ghi")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
    finally:
        cursor.close()

def load_shap_values(conn, df_shap):
    """Load feature importance từ SHAP"""
    print("\n" + "="*70)
    print("5️⃣  LOAD SHAP_VALUES")
    print("="*70)
    
    if df_shap is None or len(df_shap) == 0:
        print("⚠️  File shap_values.csv trống hoặc không tồn tại - bỏ qua")
        return
    
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    
    try:
        for _, row in df_shap.iterrows():
            try:
                # Xác định các cột có sẵn
                book_id = int(row['id']) if 'id' in row else int(row.get('book_id', 0))
                feature_name = str(row.get('feature_name', 'unknown'))
                shap_value = float(row.get('shap_value', 0))
                feature_value = str(row.get('feature_value', ''))
                shap_value_percent = float(row.get('shap_value_percent', 0))
                
                cursor.execute(
                    """INSERT INTO shap_values 
                       (book_id, feature_name, shap_value, feature_value, shap_value_percent)
                       VALUES (?, ?, ?, ?, ?)""",
                    book_id,
                    feature_name,
                    shap_value,
                    feature_value,
                    shap_value_percent
                )
                inserted += 1
            except Exception as e:
                skipped += 1
                continue
        
        conn.commit()
        print(f"✓ Insert thành công: {inserted} feature importance")
        if skipped > 0:
            print(f"⚠️  Bỏ qua (lỗi hoặc FK constraint): {skipped} bản ghi")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        conn.rollback()
    finally:
        cursor.close()

def verify_data(conn):
    """Kiểm tra số dòng trong mỗi bảng"""
    print("\n" + "="*70)
    print("📊 KIỂM TRA SỐ DÒNG TRONG CÁC BẢNG")
    print("="*70)
    
    cursor = conn.cursor()
    
    tables = [
        'dim_category',
        'books',
        'discount_recommendations',
        'book_recommendations',
        'shap_values'
    ]
    
    try:
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:30s} : {count:,} dòng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        cursor.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🚀 BẮT ĐẦU LOAD DỮ LIỆU VÀO SQL SERVER")
    print("="*70)
    print(f"Server: {SQL_SERVER}")
    print(f"Database: {DATABASE}")
    print(f"Thời gian: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)
    
    try:
        # Tạo kết nối
        print("\n🔌 Kết nối đến SQL Server...")
        conn = get_connection()
        print("✓ Kết nối thành công")
        
        # Xóa dữ liệu cũ
        truncate_tables(conn)
        
        # Load dữ liệu
        print("\n📂 Đang load dữ liệu từ CSV...")
        
        # 1. Tiki Books
        print("   - Đang đọc tiki_books_cleaned.csv...")
        df_books = pd.read_csv('data/clean/tiki_books_cleaned.csv', encoding='utf-8')
        print(f"     Tổng: {len(df_books):,} dòng")
        
        # 2. Discount Recommendations
        print("   - Đang đọc discount_recommendations.csv...")
        df_discount = pd.read_csv('data/clean/discount_recommendations.csv', encoding='utf-8')
        print(f"     Tổng: {len(df_discount):,} dòng")
        
        # 3. Book Recommendations
        print("   - Đang đọc book_recommendations.csv...")
        df_recommendations = pd.read_csv('data/clean/book_recommendations.csv', encoding='utf-8')
        print(f"     Tổng: {len(df_recommendations):,} dòng")
        
        # 4. SHAP Values (nếu tồn tại)
        try:
            print("   - Đang đọc shap_values.csv...")
            df_shap = pd.read_csv('data/clean/shap_values.csv', encoding='utf-8')
            print(f"     Tổng: {len(df_shap):,} dòng")
        except FileNotFoundError:
            print("   - shap_values.csv không tìm thấy (sẽ bỏ qua)")
            df_shap = None
        
        # Insert dữ liệu
        load_categories(conn, df_books)
        load_books(conn, df_books)
        load_discount_recommendations(conn, df_discount)
        load_book_recommendations(conn, df_recommendations)
        load_shap_values(conn, df_shap)
        
        # Verify
        verify_data(conn)
        
        print("\n" + "="*70)
        print("✅ HOÀN THÀNH! Dữ liệu đã được load vào SQL Server")
        print("="*70)
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)