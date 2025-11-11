# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build tools for ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (including model folder)
COPY . .

# Make setup.sh executable
RUN chmod +x setup.sh

# Expose Streamlit port (default inside Docker, overridden by $PORT)
EXPOSE 8501

# Run Streamlit using bash -c to handle setup.sh and $PORT
CMD bash -c "bash setup.sh && streamlit run app.py --server.port=\$PORT --server.headless=true --server.enableCORS=false"
