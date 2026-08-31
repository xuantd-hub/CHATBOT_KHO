FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Mở cổng cố định cho Cloud Run
ENV PORT=8080
EXPOSE 8080

# Chạy uvicorn qua module python để tránh triệt để lỗi click
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]