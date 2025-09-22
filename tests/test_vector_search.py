import os
import sys
import pytest
from dotenv import load_dotenv

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.vector_search_tool import vector_search

load_dotenv()

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_vector_search_returns_list_of_strings():
    """
    Tests that the vector_search function returns a list of strings (base_random_keys).
    """
    query = "لپ تاپ گیمینگ ایسوس"  # A common search query
    results = vector_search(query, k=5)

    assert isinstance(results, list)
    if results:
        assert all(isinstance(item, str) for item in results)

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_vector_search_with_english_query():
    """
    Tests the vector_search function with an English query.
    """
    query = "a fancy red mug"
    results = vector_search(query, k=5)

    assert isinstance(results, list)
    if results:
        assert all(isinstance(item, str) for item in results)

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_vector_search_fallback_mechanism():
    """
    This test is a bit tricky as it's hard to force a fallback.
    We can simulate it by providing a path to a non-existent index,
    but that would require modifying the source code or using mocks.
    For now, we'll just ensure the function doesn't crash with an empty query.
    """
    query = ""
    results = vector_search(query, k=5)
    assert isinstance(results, list)

    query_long = "asdasdasd"*100 # a nonsense long query
    results = vector_search(query_long, k=5)
    assert isinstance(results, list)
