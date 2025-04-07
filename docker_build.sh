#!/bin/bash
# Build and run the Docker container with improved error handling

# Build the Docker image
echo "Building Docker image..."
docker build -t unified-tools-server . --no-cache

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Starting container..."
    docker run -p 8000:8000 -v "$(pwd)/data:/app/data" unified-tools-server
else
    echo "Build failed. Check the error messages above."
    exit 1
fi
