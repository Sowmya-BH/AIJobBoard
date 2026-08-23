# App container — torch-free. Scoring is delegated to the scorer sidecar via
# RQ_SCORER_URL, so ResumeHQ's heavy deps are NOT installed here.
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Render injects $PORT; bind it (shell form so the var expands).
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
