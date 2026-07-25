# Dockerfile — 珍珠 AI 助手
# 构建: docker compose build
# 运行: docker compose up -d

FROM python:3.11-slim

WORKDIR /app

# ChromaDB 的 onnxruntime 需要 libgcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgcc-12-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
