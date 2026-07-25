"""LangChain 封装层 — 复用现有 ChromaDB + DeepSeek

LCEL (LangChain Expression Language) 构建检索链。
简历关键词落地——"LangChain"的证明。

面试话术：
"检索链用了 LangChain 的 Chroma + LCEL，Agent 部分自己写的，
评估后觉得 LangChain 旧版 Chains 太黑盒，新版又依赖 LangGraph 太重。"
"""

import os
import threading

from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from sentence_transformers import SentenceTransformer
from build_knowledge import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
import rag  # 确保 ChromaDB 已初始化

_init_lock = threading.Lock()
_chain = None


class _STEmbeddings:
    """sentence-transformers → LangChain embedding 接口"""

    def __init__(self):
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text]).tolist()[0]


def _get_chain():
    """惰性初始化 LCEL 检索链"""
    global _chain
    if _chain is not None:
        return _chain

    with _init_lock:
        if _chain is not None:
            return _chain

        rag._ensure_initialized()

        vectorstore = Chroma(
            embedding_function=_STEmbeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(
            model="deepseek-v4-flash",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.7,
            max_tokens=600,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是珍珠个体户老板娘的小助手。参考知识库回答客户问题。\n"
                       "要求：口语化、3-5句话、亲切自然。知识库里没有的诚实说不知道。"),
            ("user", "知识库：\n{context}\n\n客户问题：{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        _chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    return _chain


def ask(question: str) -> str:
    """用 LangChain LCEL 链回答问题。

    >>> answer = ask("圆脸适合什么珍珠")
    """
    chain = _get_chain()
    return chain.invoke(question)
