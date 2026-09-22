
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first for better Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY api/ ./api/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
