# import os
# import uuid
# from sentence_transformers import SentenceTransformer
# from agent import vectors

# # 1. Load the model on your Mac (bypassing HF API)
# print("🧠 Loading local SBERT model for testing...")
# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# resume = "Senior Python Developer with FastAPI experience"
# filters = {"source": "All Sources"}

# try:
#     # 2. Generate vector locally
#     print("✨ Generating vector on Mac Pro...")
#     vec = model.encode(resume).tolist()
    
#     # 3. Search Qdrant Cloud
#     print("🔍 Searching Qdrant Cloud...")
#     results = vectors.search(vec, filters, limit=5)
    
#     print(f"✅ Success! Found {len(results)} matching jobs:")
#     for r in results:
#         print(f"- {r['payload'].get('title')} at {r['payload'].get('company')} (Score: {r['score']})")

# except Exception as e:
#     print(f"❌ Error: {e}")
from agent import vectors
import os

import os
from agent import vectors

# print(f"DEBUG: Qdrant URL is -> '{vectors.QDRANT_URL}'")
# print(f"DEBUG: HF URL is     -> '{vectors.HF_EMBED_URL}'")

# if not vectors.QDRANT_URL.startswith("https://"):
#     print("❌ ERROR: QDRANT_URL must start with https://")

# if " " in vectors.QDRANT_URL:
#     print("❌ ERROR: There is a space in your QDRANT_URL variable!")


# Set dummy filters
filters = {"source": "All Sources"}
# A sample resume string
resume = "I am a Senior Backend Developer specialized in Python, FastAPI, and PostgreSQL."

print("🔍 Testing Qdrant + HF API connection...")
results = vectors.recommend(resume, filters, limit=5)

if results:
    print(f"✅ Success! Found {len(results)} matching jobs:")
    for r in results:
        print(f"- {r['title']} at {r['company']} (Match Score: {r['score']}%)")
else:
    print("❌ No matches found. Check your QDRANT_COLLECTION name in vectors.py")