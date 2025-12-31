# SpaceProof Multi-Domain Verification API
# Entropy-based counterfeit detection for aerospace, food, and medical domains
#
# Build: docker build -t spaceproof:2.0 .
# Run:   docker run -p 8000:8000 spaceproof:2.0
# API:   http://localhost:8000/api/v1/docs

FROM python:3.11-slim

# Labels
LABEL maintainer="SpaceProof Team"
LABEL version="2.0.0"
LABEL description="Multi-Domain Verification API: Aerospace, Food, Medical"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create receipts directory and ensure writable
RUN mkdir -p /app/receipts && chmod 755 /app/receipts
RUN touch /app/receipts.jsonl && chmod 644 /app/receipts.jsonl

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run with hypercorn for production (4 workers as specified)
CMD ["hypercorn", "api.server:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
