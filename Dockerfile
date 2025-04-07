FROM python:3.11-slim

# Set up working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright and browser
RUN pip install playwright && \
    playwright install chromium --with-deps

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p ./data/scraped ./data/documents ./data/cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]