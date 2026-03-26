# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements/metadata to leverage cache
COPY pyproject.toml .
# Create a dummy folder so pip can install dependencies accurately
RUN mkdir email_triage_env && touch email_triage_env/__init__.py
RUN pip install --no-cache-dir .

# Now copy the actual source and install properly
COPY . .
RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code for observability
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose Hugging Face Spaces standard port
EXPOSE 7860

# HEALTHCHECK specification
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c 'import email_triage_env; env = email_triage_env.make(); env.reset()' || exit 1

# Entry point starts the FastAPI health app
CMD ["python", "email_triage_env/app.py"]
