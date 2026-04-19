from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # ==========================================
    # 通义千问 API 配置（必填）
    # ==========================================
    QWEN_API_KEY: str
    QWEN_CHAT_MODEL: str = "qwen-turbo"
    QWEN_EMBEDDING_MODEL: str = "text-embedding-v3"
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ==========================================
    # RAG 检索配置
    # ==========================================
    KNOWLEDGE_BASE_DIR: str = "data/raw"
    FAISS_INDEX_PATH: str = "data/vectorstore/faiss_index"
    EMBEDDING_CACHE_DIR: str = "./data/embeddings_cache"
    RAG_TOP_K: int = 20
    RERANK_TOP_N: int = 5

    # ==========================================
    # 数据路径配置
    # ==========================================
    DATA_DIR: str = "data"
    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"
    UPLIFT_MODEL_PATH: str = "models/uplift_model.pkl"

    # ==========================================
    # 项目运行配置
    # ==========================================
    ENV: Literal["dev", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # ✅ 可选：如果还有其他未知配置，允许额外输入

# 全局单例
settings = Settings()