# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt ./requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src/ ./src/
COPY tests/ ./tests/
COPY app.py ./

# Expose the Streamlit port
EXPOSE 8510

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8510", "--server.address=0.0.0.0"]