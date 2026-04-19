# src/rag/retriever.py
from src.rag.vectorstore import VectorStoreManager
from src.core.config import settings
from langchain_community.document_compressors import DashScopeRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

def build_retriever(vs_manager: VectorStoreManager):
    """构建 RAG 检索器（含重排）"""
    # 基础向量检索
    base_retriever = vs_manager.as_retriever(
        search_kwargs={"k": settings.RAG_TOP_K}
    )

    # 重排模型
    rerank = DashScopeRerank(
        model="gte-rerank",
        top_n=settings.RERANK_TOP_N,
        dashscope_api_key=settings.QWEN_API_KEY
    )

    # 上下文压缩检索器
    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=rerank
    )

    return retriever