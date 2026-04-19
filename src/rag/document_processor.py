# src/rag/document_processor.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )

    def load_and_split_documents(self) -> List[Document]:
        """加载并切分知识库目录下的所有 Markdown 文件"""
        docs = []
        kb_dir = settings.KNOWLEDGE_BASE_DIR

        if not os.path.exists(kb_dir):
            logger.warning(f"知识库目录不存在: {kb_dir}")
            return docs

        for filename in os.listdir(kb_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(kb_dir, filename)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
                    logger.info(f"加载知识库文件: {filename}")
                except Exception as e:
                    logger.error(f"加载文件失败 {filename}: {e}")

        if docs:
            splits = self.splitter.split_documents(docs)
            logger.info(f"文档切分完成: {len(docs)} 个文件 -> {len(splits)} 个片段")
            return splits

        return docs