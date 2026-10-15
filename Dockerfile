FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi
COPY . .

# Buat folder penyimpanan lokal di container
RUN mkdir -p uploads outputs previews static

# Hugging Face Spaces mewajibkan aplikasi berjalan di port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]