"""Check that local multilingual embeddings work on your machine (needs internet once,
to download the model, ~0.5 GB). Run:  python scripts/check_embeddings.py
"""
import time
t0 = time.time()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("intfloat/multilingual-e5-small")
print(f"model loaded in {time.time() - t0:.0f}s")
docs = ["passage: Return windows by category. Books and stationery can be returned within 10 days.",
        "passage: Refunds to UPI take 1 to 3 business days."]
qs = ["query: मुझे नोटबुक लौटाने के लिए कितने दिन मिलेंगे",      # Hindi: how many days to return a notebook
      "query: how long does a refund to upi take"]
d = m.encode(docs, normalize_embeddings=True)
q = m.encode(qs, normalize_embeddings=True)
print("similarity matrix (rows=queries, cols=docs):\n", (q @ d.T).round(3))
print("OK if the Hindi notebook query scores higher with doc 0 and the UPI query with doc 1.")
