# ✅ Dockerfile
# ----------------------------------
FROM python:3.10-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
