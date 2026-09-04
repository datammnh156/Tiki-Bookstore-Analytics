"""
FIX: Add missing columns to book_recommendations table
"""

import pyodbc
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*90)
print("🔧 FIXING TABLE SCHEMA - Adding missing columns")
print("="*90)

try:
    conn_str = r'Driver={ODBC Driver 17 for SQL Server};Server=.\SERVERONHA;Database=TikiBookStore;Trusted_Connection=yes;'
    conn = pyodbc.connect(conn_str, timeout=10)
    conn.autocommit = False
    cursor = conn.cursor()
    
    print("\n✓ Connected to TikiBookStore")
    
    # Add missing columns
    print("\n📝 Adding missing columns...")
    
    try:
        cursor.execute("ALTER TABLE book_recommendations ADD source_name NVARCHAR(MAX)")
        print("  ✓ Added source_name column")
    except Exception as e:
        if "already exists" in str(e):
            print("  - source_name already exists")
        else:
            print(f"  ✗ Error adding source_name: {e}")
    
    try:
        cursor.execute("ALTER TABLE book_recommendations ADD recommended_name NVARCHAR(MAX)")
        print("  ✓ Added recommended_name column")
    except Exception as e:
        if "already exists" in str(e):
            print("  - recommended_name already exists")
        else:
            print(f"  ✗ Error adding recommended_name: {e}")
    
    conn.commit()
    
    # Verify new schema
    print("\n✅ Verifying new schema...")
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'book_recommendations'
        ORDER BY ORDINAL_POSITION
    """)
    
    columns = cursor.fetchall()
    print("\n   Updated columns:")
    for col_name, data_type, pos in columns:
        print(f"   {pos}. {col_name:<25} {data_type}")
    
    conn.close()
    print("\n" + "="*90)
    print("✅ SCHEMA FIXED - Ready to load data")
    print("="*90 + "\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
