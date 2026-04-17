import os
from src.config import settings
from src.embeddings import QwenEmbeddings

# ========================
# LangChain 1.0+ 正确导入
# ========================
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.storage import LocalFileStore
from langchain_community.embeddings.cache import CacheBackedEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import DashScopeRerank

class UpliftRAGService:
    def __init__(self):
        self.base_embed = QwenEmbeddings()
        self.store = LocalFileStore(settings.EMBEDDING_CACHE_DIR)
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            self.base_embed, self.store, namespace=settings.QWEN_EMBEDDING_MODEL
        )

        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.vs = FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vs = self._build_vectorstore()

        self._build_retriever()

    def _build_vectorstore(self):
        docs = []
        for f in os.listdir(settings.KNOWLEDGE_BASE_DIR):
            if f.endswith(".md"):
                loader = TextLoader(
                    os.path.join(settings.KNOWLEDGE_BASE_DIR, f),
                    encoding="utf-8"
                )
                docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )
        splits = splitter.split_documents(docs)
        vs = FAISS.from_documents(splits, self.embeddings)
        vs.save_local(settings.FAISS_INDEX_PATH)
        return vs

    def _build_retriever(self):
        bm25 = BM25Retriever.from_documents(self.vs.docstore._dict.values())
        bm25.k = 10
        vec_ret = self.vs.as_retriever(k=10)

        self.ensemble = EnsembleRetriever(
            retrievers=[vec_ret, bm25],
            weights=[0.7, 0.3]
        )
        self.rerank = DashScopeRerank(
            model="gte-rerank",
            top_n=3,
            api_key=settings.QWEN_API_KEY
        )

    def retrieve(self, query: str):
        docs = self.ensemble.invoke(query)
        compressed = self.rerank.compress_documents(docs, query)
        return [d.page_content for d in compressed]