FROM python:3.12-slim

LABEL org.opencontainers.image.title="Scen-Opt" \
      org.opencontainers.image.description="A Scenario Optimization Toolbox for Data-Driven Convex Programming. Solves LP, QP, and SDP problems using the scenario approach with rigorous probabilistic guarantees." \
      org.opencontainers.image.source="https://github.com/Kiguli/Scen-Opt" \
      org.opencontainers.image.licenses="MIT"

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

# Default worker count. Override with -e GUNICORN_CMD_ARGS="--workers 4".
ENV GUNICORN_CMD_ARGS="--workers 2"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "7200", "app:app"]
