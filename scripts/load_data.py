import pandas as pd
import sqlite3
import os
from typing import Optional
import time

try:
    # Prefer shared resolver if available (running inside app)
    from utils.utils import get_db_path as _get_db_path
except Exception:
    _get_db_path = None

def load_parquet_to_sqlite(db_path: Optional[str] = None, data_dir: str = 'db_data'):
    if db_path is None:
        db_path = _get_db_path() if _get_db_path else os.path.join(os.environ.get('DATASETS_DIR') or '/datasets/', 'torob.db')
    
    # Ensure output directory exists
    out_dir = os.path.dirname(db_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Simple lock to avoid concurrent rebuilds
    lock_path = os.path.join(out_dir or '.', '.build_db.lock')
    for _ in range(50):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        # Remove the old database file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)
        
        conn = sqlite3.connect(db_path)
    
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.parquet'):
                table_name = file_name.replace('.parquet', '')
                file_path = os.path.join(data_dir, file_name)
                
                df = pd.read_parquet(file_path)
                
                # Convert list/dict columns to string for SQLite compatibility
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if the first non-null element is a list/dict
                        series = df[col].dropna()
                        first_non_null = series.iloc[0] if not series.empty else None
                        if isinstance(first_non_null, (list, dict)):
                            df[col] = df[col].astype(str)

                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"Loaded {table_name} into {db_path}")

        conn.close()
    finally:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass

if __name__ == '__main__':
    load_parquet_to_sqlite()
