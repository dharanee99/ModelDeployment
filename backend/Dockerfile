# Use a larger Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Upgrade pip and install smaller dependencies first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pillow python-multipart

# Install larger dependencies in separate layers to avoid memory overload
RUN pip install --no-cache-dir torch torchvision

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8000"]
