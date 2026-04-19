import os
from src.config import settings
from src.embeddings import QwenEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ==============================================
# ✅ 官方1.x标准导入（langchain 1.2.15 完美适配）
# ==============================================
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import DashScopeRerank


class UpliftRAGService:
    def __init__(self):
        self.embeddings = QwenEmbeddings()

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
        base_retriever = self.vs.as_retriever(search_kwargs={"k": 20})

        rerank = DashScopeRerank(
            model="gte-rerank",
            top_n=5,
            dashscope_api_key=settings.QWEN_API_KEY
        )

        self.retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=rerank
        )

    def retrieve(self, query: str):
        docs = self.retriever.invoke(query)
        return [d.page_content for d in docs]