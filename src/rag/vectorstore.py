# src/rag/vectorstore.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vs = self._load_or_create()

    def _load_or_create(self):
        """加载现有向量库或创建新的"""
        if os.path.exists(settings.FAISS_INDEX_PATH):
            logger.info("📂 加载现有向量库")
            return FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        logger.info("🆕 创建新向量库")
        # 用一个空文本初始化，避免报错
        return FAISS.from_texts(["init"], self.embeddings)

    def add_documents(self, docs: List[Document]):
        """✨ 增量添加文档，不用重建"""
        if docs:
            self.vs.add_documents(docs)
            # 确保保存目录存在
            os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
            self.vs.save_local(settings.FAISS_INDEX_PATH)
            logger.info(f"✅ 增量添加 {len(docs)} 个文档片段")

    def as_retriever(self, search_kwargs: dict = None):
        """获取检索器对象"""
        return self.vs.as_retriever(
            search_kwargs=search_kwargs or {"k": settings.RAG_TOP_K}
        )