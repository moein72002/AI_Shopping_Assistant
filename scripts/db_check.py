import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torob.db')
print('DB:', DB_PATH, 'exists:', os.path.exists(DB_PATH))
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
for t in ['base_products','members','shops']:
    try:
        n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        print(t, n)
    except Exception as e:
        print('ERR', t, e)
try:
    rows = c.execute('SELECT random_key, substr(persian_name,1,80) FROM base_products LIMIT 5').fetchall()
    for r in rows:
        print('sample:', r)
except Exception as e:
    print('ERR sample', e)
conn.close()



