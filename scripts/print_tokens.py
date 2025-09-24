import os
import re

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]

q = os.environ.get("Q", "سلام گوشی آیفون ۱۶ پرومکس")
print(simple_tokenize(q))


