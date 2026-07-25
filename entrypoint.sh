#!/bin/bash
# Docker 容器启动脚本：先建知识库（如果 chroma_db/ 是空的），再启动服务
if [ ! -d "/app/chroma_db" ] || [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
    echo "🔨 首次启动，构建知识库..."
    python build_knowledge.py
fi
echo "🚀 启动服务..."
exec python app.py
