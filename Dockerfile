FROM python:3.10-slim

# Install library sistem Linux yang dibutuhkan Rasterio (GDAL) & PostgreSQL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode proyek
COPY . .

# Port standar Hugging Face Spaces
EXPOSE 7860

# Perintah untuk menjalankan Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]