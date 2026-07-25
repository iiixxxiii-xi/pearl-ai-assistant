"""
build_knowledge.py — 珍珠知识库构建脚本
把 data/ 文件夹里的 .txt 文件转成向量，存到 ChromaDB
跑一次就行，以后知识库内容有改动就重新跑一遍
"""

import os
import sys
from pathlib import Path

# ⚠️ 必须在 import sentence_transformers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# 加载 .env 里的配置（虽然这里不直接调 API，但保持项目一致性）
from dotenv import load_dotenv
load_dotenv()

# ChromaDB：存向量 + 原文的轻量数据库
import chromadb
from chromadb.config import Settings

# sentence-transformers：把中文文本转成向量
from sentence_transformers import SentenceTransformer


# ============================================================
# 配置区 —— 需要改的地方都在这
# ============================================================

# 知识库源文件目录（放 .txt 文件的地方）
DATA_DIR = Path(__file__).parent / "data"

# ChromaDB 持久化存储目录（自动生成，不用手动创建）
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# ChromaDB 里的集合名（相当于数据库里的"表"）
COLLECTION_NAME = "pearl_knowledge"

# 使用的 Embedding 模型（中文效果好、免费、本地跑）
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 分割符 —— 用空行来切分不同的问答块
SPLIT_SEPARATOR = "\n\n"


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
    """
    核心流程：
    1. 加载 Embedding 模型
    2. 把每个知识块转成向量
    3. 把向量 + 原文一起存到 ChromaDB
    """
    print(f"\n🧠 正在加载 Embedding 模型：{EMBEDDING_MODEL}")
    print("   （第一次用会从 HuggingFace 下载模型，大概几百MB，等一下就好）")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    print("   ✅ 模型加载完成")

    # 创建 ChromaDB 客户端（持久化存储，数据存到 chroma_db/ 目录）
    print(f"\n🗄️  正在连接 ChromaDB（存储目录：{CHROMA_DIR}）")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # 获取或创建集合。如果已存在同名集合就先删掉重建（保证每次构建都是最新数据）
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   → 删除旧集合「{COLLECTION_NAME}」，准备重建")
    except Exception:
        pass  # 集合不存在就不用删

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "珍珠知识库 — 客服回复 + 小红书内容的共用知识",
            "hnsw:space": "cosine",  # 余弦距离，1.0 - dist = 余弦相似度
        },
    )

    # 批量向量化并存入
    print(f"\n🔢 正在将 {len(chunks)} 个知识块向量化并存入数据库...")
    print("   （这一步需要一点时间，取决于知识库大小和你的电脑性能）")

    batch_size = 10  # 每批处理 10 条，避免一次性吃太多内存
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # 生成每条数据的唯一 ID
        ids = [f"doc_{j}" for j in range(i, i + len(batch))]

        # embedding 函数 —— 把每条文本转成向量
        embeddings = model.encode(batch).tolist()

        # 存入 ChromaDB：向量 + 原文 + ID
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=batch
        )

        progress = min(i + batch_size, len(chunks))
        print(f"   → 已处理 {progress}/{len(chunks)} 条")

    print(f"\n✅ 知识库构建完成，共 {len(chunks)} 条记录")
    print(f"   向量数据存储在：{CHROMA_DIR}")
    print(f"   集合名称：{COLLECTION_NAME}")
    print(f"\n💡 提示：以后知识库内容有改动，重新跑 python build_knowledge.py 就行")


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🦪  珍珠 AI 助手 — 知识库构建工具")
    print("=" * 55)

    # 第 1 步：读取知识库文件，切成块
    chunks = load_knowledge_files(DATA_DIR)

    # 第 2 步：向量化 + 存入 ChromaDB
    build_knowledge_base(chunks)
