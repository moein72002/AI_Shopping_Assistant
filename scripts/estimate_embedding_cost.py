import os
import sqlite3
import math

DATASET_PATH = os.path.abspath("/datasets/")
DB_PATH = os.path.join(DATASET_PATH, "torob.db")

# Pricing assumptions (USD)
PRICE_PER_M_TOKEN = 0.02  # text-embedding-3-small
CHARS_PER_TOKEN = 4.0     # rough average; actual varies by language/content

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*),
               COALESCE(SUM(LENGTH(COALESCE(persian_name, ''))
                             + LENGTH(COALESCE(english_name, ''))
                             + LENGTH(COALESCE(extra_features, ''))), 0)
        FROM base_products
        """
    )
    row_count, char_sum = cur.fetchone()
    conn.close()

    est_tokens = math.ceil(char_sum / CHARS_PER_TOKEN)
    est_cost_usd = (est_tokens / 1_000_000.0) * PRICE_PER_M_TOKEN

    print("Rows:", row_count)
    print("Total chars:", char_sum)
    print("Estimated tokens (~chars/4):", est_tokens)
    print("Price per 1M tokens (USD):", PRICE_PER_M_TOKEN)
    print("Estimated embedding cost (USD):", round(est_cost_usd, 6))
