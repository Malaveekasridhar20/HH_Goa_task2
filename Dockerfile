FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/indexes/ data/indexes/

# Ensure standard PORT handling for Cloud Run (defaults to 8080)
ENV PORT=8080

# Run uvicorn using sh so that $PORT is evaluated
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir backend"
