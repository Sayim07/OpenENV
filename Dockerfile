# Stage 1: Build the frontend Next.js dashboard
FROM node:20-alpine AS builder-frontend
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# Stage 2: Build Backend & Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Ensure we don't write PYC to save space and keep image small
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Copy the exported static frontend from Stage 1 into dashboard/out
COPY --from=builder-frontend /app/dashboard/out /app/dashboard/out

# Install the local email_triage_env package
RUN pip install -e ./email_triage_env

EXPOSE 7860

# Add a HEALTHCHECK that imports the environment and calls reset()
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import email_triage_env; email_triage_env.make().reset(); print('OK')" || exit 1

CMD ["python", "app.py"]
