# Builder stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --prefix=/install .

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Copy app code
COPY src/ /app/src/

# Set env
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "-m", "mcp_finance.server"]
