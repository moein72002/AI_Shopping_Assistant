import pandas as pd
import sqlite3
import os

def load_parquet_to_sqlite():
    db_path = '/datasets/torob.db'
    data_dir = 'db_data'
    
    # Remove the old database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.parquet'):
            table_name = file_name.replace('.parquet', '')
            file_path = os.path.join(data_dir, file_name)
            
            df = pd.read_parquet(file_path)
            
            # Convert list columns to string for SQLite compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if the first non-null element is a list
                    first_non_null = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(first_non_null, list) or isinstance(first_non_null, dict):
                         df[col] = df[col].astype(str)

            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Loaded {table_name} into {db_path}")

    conn.close()

if __name__ == '__main__':
    load_parquet_to_sqlite()
