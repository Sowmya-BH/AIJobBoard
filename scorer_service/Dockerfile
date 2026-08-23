# Scorer sidecar — full ResumeHQ (torch/SBERT). Runs in its OWN container with
# its own memory budget, separate from the ~512 MB app container.
FROM python:3.11-slim

WORKDIR /srv

# System deps some wheels want
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Full ResumeHQ (this pulls sentence-transformers + torch) + the web layer
RUN pip install --no-cache-dir resumehq fastapi uvicorn[standard]

# Pre-download NLTK data and the SBERT model at BUILD time so the image is
# self-contained and the first request is fast (no runtime download needed).
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('wordnet','omw-1.4','punkt','stopwords')]" || true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

COPY app.py /srv/app.py

# Render injects $PORT for private services too; bind it (shell form).
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8100} --workers 1
