# Multi-stage build for production
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install -e ".[viz,excel,parquet]"

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ src/
COPY README.md .
COPY pyproject.toml .

# Install application
RUN pip install -e .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create directories for data and output
RUN mkdir -p /app/data /app/output

# Set default command
CMD ["provider-dedupe", "--help"]

# Labels for metadata
LABEL org.opencontainers.image.title="Provider Dedupe"
LABEL org.opencontainers.image.description="Health Provider deduplication system"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Your Organization"
LABEL org.opencontainers.image.source="https://github.com/yourorg/provider-dedupe"