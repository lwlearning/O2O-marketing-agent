from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY")
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL")
    QWEN_CHAT_MODEL: str = os.getenv("QWEN_CHAT_MODEL")

    KNOWLEDGE_BASE_DIR: str = os.getenv("KNOWLEDGE_BASE_DIR")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH")
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR")

settings = Settings()