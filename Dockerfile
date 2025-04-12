FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright dependencies
RUN pip install --no-cache-dir playwright==1.51.0 && \
    playwright install chromium && \
    playwright install-deps chromium

# Copy requirements first for better caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV DEV_MODE=False
ENV ALLOWED_DIRS=./data
ENV MEMORY_FILE_PATH=./data/memory.json
ENV FILE_CACHE_ENABLED=True
ENV FILE_CACHE_MAX_AGE=3600
ENV DEFAULT_COMMIT_USERNAME=OtherTales
ENV DEFAULT_COMMIT_EMAIL=system@othertales.com
ENV SCRAPER_MIN_DELAY=1.0
ENV SCRAPER_MAX_DELAY=3.0
ENV USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
ENV SEARCH_PROVIDER=serper
ENV SEARCH_DEFAULT_COUNTRY=us
ENV SEARCH_DEFAULT_LOCALE=en
ENV SEARCH_TIMEOUT=30
ENV SEARCH_MAX_RETRIES=3
ENV SEARCH_RETRY_DELAY=2
ENV USE_GRAPH_DB=False
ENV VECTOR_EMBEDDING_ENABLED=True
ENV VECTOR_MODEL_NAME=all-MiniLM-L6-v2
ENV SCRAPER_DATA_PATH=./data/scraper
ENV SEARCH_API_KEY=demo-key

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
