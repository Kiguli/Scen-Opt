FROM python:3.12-slim

# System dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev git cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt requirements-solvers.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-solvers.txt \
    && pip install --no-cache-dir gunicorn

# Copy application code
COPY app.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "7200", "app:app"]
