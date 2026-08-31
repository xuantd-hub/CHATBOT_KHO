FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện trước để tận dụng Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Khai báo cổng mặc định cho Cloud Run
ENV PORT=8080
EXPOSE 8080

# Lệnh chạy ứng dụng bắt chuẩn biến môi trường PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}