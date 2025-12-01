import sqlite3
import os

# Correct path to the database
db_path = "data/csv_database.db"

if not os.path.exists(db_path):
    print(f"❌ Database not found at: {db_path}")
else:
    print(f"✓ Database found at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\n📊 Tables in database: {len(tables)}")
    for table in tables:
        table_name = table[0]
        print(f"\n  Table: {table_name}")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"  Columns ({len(columns)}):")
        for col in columns:
            print(f"    - {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
    
    conn.close()
