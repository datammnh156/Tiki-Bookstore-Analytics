"""
ETL: Load Hybrid Recommendations to SQL Server via pyodbc
"""

import pandas as pd
import pyodbc
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*90)
print("🔄 ETL: Load Hybrid Recommendations to SQL Server")
print("="*90)

# 1. Load CSV
print("\n📂 Loading CSV...")
df = pd.read_csv('data/clean/book_recommendations.csv', encoding='utf-8')
print(f"✓ Loaded {len(df):,} rows")

# Validate
required = ['source_id', 'source_name', 'recommended_id', 'recommended_name', 
            'similarity_score', 'confidence_level']
if not all(col in df.columns for col in required):
    print("❌ Missing required columns")
    sys.exit(1)

print(f"\n  Confidence distribution:")
for level, count in df['confidence_level'].value_counts().sort_index().items():
    print(f"    {level}: {count:,} ({count/len(df)*100:.1f}%)")

# 2. Connect to SQL Server
print("\n🔗 Connecting to SQL Server...")

try:
    # Try multiple connection strings
    connection_strings = [
        r'Driver={ODBC Driver 17 for SQL Server};Server=.\SERVERONHA;Database=TikiBookStore;Trusted_Connection=yes;',
        'Driver={ODBC Driver 17 for SQL Server};Server=DANHITAM\\SERVERONHA;Database=TikiBookStore;Trusted_Connection=yes;',
        'Driver={ODBC Driver 17 for SQL Server};Server=SERVERONHA;Database=TikiBookStore;Trusted_Connection=yes;',
    ]
    
    conn = None
    for i, conn_str in enumerate(connection_strings, 1):
        try:
            print(f"  Attempt {i}: {conn_str[:60]}...")
            conn = pyodbc.connect(conn_str, timeout=10)
            print(f"✓ Connected successfully with attempt {i}!")
            break
        except Exception as e:
            if i == len(connection_strings):
                raise e
            continue
    
    if not conn:
        raise Exception("All connection attempts failed")
    
    conn.autocommit = False
    cursor = conn.cursor()
    print(f"  Database: TikiBookStore")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

# 3. Truncate old data
print("\n📥 STEP 1: Clean old data")
try:
    cursor.execute("SELECT COUNT(*) FROM book_recommendations")
    old_count = cursor.fetchone()[0]
    print(f"  Current rows: {old_count:,}")
    
    cursor.execute("TRUNCATE TABLE book_recommendations")
    conn.commit()
    print(f"✓ Table truncated")
    
    cursor.execute("SELECT COUNT(*) FROM book_recommendations")
    new_count = cursor.fetchone()[0]
    print(f"  Rows after truncate: {new_count:,}")
    
except Exception as e:
    print(f"❌ Truncate failed: {e}")
    conn.rollback()
    conn.close()
    sys.exit(1)

# 4. Insert new data
print("\n📥 STEP 2: Load 19,150 hybrid recommendations")

try:
    cursor.execute("ALTER TABLE book_recommendations NOCHECK CONSTRAINT ALL")
    
    batch_size = 100
    inserted = 0
    
    for idx in range(0, len(df), batch_size):
        batch = df.iloc[idx:idx+batch_size]
        
        for _, row in batch.iterrows():
            cursor.execute(
                """INSERT INTO book_recommendations 
                   (source_id, source_name, recommended_id, recommended_name, similarity_score, confidence_level)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    int(row['source_id']),
                    str(row['source_name']),
                    int(row['recommended_id']),
                    str(row['recommended_name']),
                    float(row['similarity_score']),
                    str(row['confidence_level'])
                )
            )
            inserted += 1
        
        if inserted % 1000 == 0:
            print(f"  Inserted {inserted:,} rows...")
    
    cursor.execute("ALTER TABLE book_recommendations WITH CHECK CHECK CONSTRAINT ALL")
    conn.commit()
    
    print(f"✓ Insert complete: {inserted:,} rows")
    
except Exception as e:
    print(f"❌ Insert failed: {e}")
    conn.rollback()
    conn.close()
    sys.exit(1)

# 5. Verify
print("\n✅ Verifying data in SQL...")

try:
    cursor.execute('SELECT COUNT(*) FROM book_recommendations')
    total = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT confidence_level, COUNT(*) as count 
        FROM book_recommendations 
        GROUP BY confidence_level 
        ORDER BY confidence_level
    ''')
    conf_dist = cursor.fetchall()
    
    cursor.execute('SELECT TOP 5 source_id, source_name, recommended_name, similarity_score, confidence_level FROM book_recommendations ORDER BY source_id')
    samples = cursor.fetchall()
    
    print(f"\n📊 Final Verification:")
    print(f"  Expected: 19,150 rows")
    print(f"  Actual: {total:,} rows")
    
    if total == 19150:
        print(f"  ✅ Row count CORRECT!")
    else:
        print(f"  ❌ Row count MISMATCH!")
    
    print(f"\n  Confidence distribution:")
    for level, count in conf_dist:
        pct = count / total * 100 if total > 0 else 0
        print(f"    {level}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  Sample data (first 5 rows):")
    for row in samples:
        print(f"    {row[0]} → {row[2][:30]}... (score: {row[3]:.4f}, conf: {row[4]})")
    
    conn.close()
    
except Exception as e:
    print(f"⚠ Verification error: {e}")
    if conn:
        conn.close()
    sys.exit(1)

print("\n" + "="*90)
print("🎉 ETL COMPLETE - Data loaded successfully!")
print("="*90 + "\n")
