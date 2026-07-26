"""
build_knowledge.py — 珍珠知识库构建脚本
把 data/ 文件夹里的 .txt 文件转成向量，存到 ChromaDB
跑一次就行，以后知识库内容有改动就重新跑一遍

Embedding 改为 DashScope API 调用（不再本地加载模型）。
"""
import os
import sys
from pathlib import Path

# 加载 .env 里的配置
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# ============================================================
# 配置区
# ============================================================

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "pearl_knowledge"
SPLIT_SEPARATOR = "\n\n"

# Embedding API 配置
EMBEDDING_API_KEY = os.getenv("DEEPSEEK_API_KEY")
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")


def load_knowledge_files(data_dir: Path) -> list[str]:
    """
    读取 data/ 文件夹下所有 .txt 文件，
    按空行（\n\n）切分成一个个知识块
    返回：知识块字符串的列表
    """
    if not data_dir.exists():
        print(f"❌ 知识库目录不存在：{data_dir}")
        print("   请先在项目根目录创建 data/ 文件夹，把知识库 .txt 放进去")
        sys.exit(1)

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"❌ 在 {data_dir} 里没找到 .txt 文件")
        print("   请至少放一个知识库文件（比如 pearl_knowledge.txt）")
        sys.exit(1)

    all_chunks = []

    for txt_file in txt_files:
        print(f"📄 正在读取：{txt_file.name}")
        # utf-8 编码读取，兼容中文
        content = txt_file.read_text(encoding="utf-8").strip()

        # 按空行切分成块（每个块是一个独立的问答对或知识点）
        chunks = content.split(SPLIT_SEPARATOR)

        # 过滤掉空白块（去掉首尾空格后为空的行）
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        print(f"   → 切分出 {len(chunks)} 个知识块")
        all_chunks.extend(chunks)

    return all_chunks


def build_knowledge_base(chunks: list[str]):
    """用 DashScope Embedding API 向量化知识块，存入 ChromaDB"""
    print(f"\n🔗 连接 Embedding API: {EMBEDDING_MODEL} ...")
    client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)

    def embed_batch(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    # 创建 ChromaDB 客户端
    print(f"\n🗄️  正在连接 ChromaDB（存储目录：{CHROMA_DIR}）")
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # 删除旧集合（如果存在）
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"   → 删除旧集合「{COLLECTION_NAME}」，准备重建")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "珍珠知识库 — 客服回复 + 小红书内容的共用知识",
            "hnsw:space": "cosine",
        },
    )

    # 批量向量化并存入
    print(f"\n🔢 正在通过 API 向量化 {len(chunks)} 个知识块...")
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"doc_{j}" for j in range(i, i + len(batch))]
        embeddings = embed_batch(batch)
        collection.add(ids=ids, embeddings=embeddings, documents=batch)
        progress = min(i + batch_size, len(chunks))
        print(f"   → 已处理 {progress}/{len(chunks)} 条")

    # 写版本标记（rag.py 的 _migrate_chroma_to_api 用）
    (CHROMA_DIR / "embedding_version.txt").write_text("api-v2")

    print(f"\n✅ 知识库构建完成，共 {len(chunks)} 条记录（API Embedding: {EMBEDDING_MODEL}）")
    print(f"   向量数据存储在：{CHROMA_DIR}")
    print(f"\n💡 以后知识库内容有改动，重新跑 python build_knowledge.py 就行")


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("=" * 55)
    print("🦪  珍珠 AI 助手 — 知识库构建工具")
    print("=" * 55)

    # 第 1 步：读取知识库文件，切成块
    chunks = load_knowledge_files(DATA_DIR)

    # 第 2 步：向量化 + 存入 ChromaDB
    build_knowledge_base(chunks)
