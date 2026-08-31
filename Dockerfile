FROM python:3.10-slim

WORKDIR /app

# Copy và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Cloud Run tự động cấp cổng PORT
ENV PORT 8080

# Chạy uvicorn bắt trực tiếp biến môi trường PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT