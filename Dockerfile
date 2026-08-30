FROM python:3.10-slim

WORKDIR /app

# Coppy va cai dat Thu vien
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Coppy toan bo ma nguồn
COPY . .

# Cloud Run tự động cấp cổng PORT thông qua biến môi trường
ENV PORT 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]