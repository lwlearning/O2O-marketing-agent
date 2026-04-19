import dashscope
from dashscope import TextEmbedding
from typing import List
from src.core.config import settings
from langchain_core.embeddings import Embeddings

class QwenEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = settings.QWEN_API_KEY
        self.model = settings.QWEN_EMBEDDING_MODEL
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        valid = [t.strip() for t in texts if t.strip()]
        if not valid: return []
        embeds = []
        for i in range(0, len(valid), 10):
            batch = valid[i:i+10]
            try:
                resp = TextEmbedding.call(model=self.model, input=batch, timeout=30)
                if resp.status_code == 200:
                    embeds.extend([e["embedding"] for e in resp.output["embeddings"]])
            except:
                continue
        return embeds

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]