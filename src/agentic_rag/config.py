"""应用配置：从环境变量 / .env 读取，集中管理。"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（agentic-rag/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """所有可配置项。环境变量优先级高于 .env 文件。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM（OpenAI 兼容接口，如 DeepSeek / Qwen / Moonshot / 智谱）----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    temperature: float = 0.0

    # ---- Embedding（注意：DeepSeek 不提供 embedding，需单独配服务）----
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"

    # ---- 数据与存储 ----
    data_dir: Path = PROJECT_ROOT / "data"
    seed_dir: Path = PROJECT_ROOT / "data" / "seed"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    bm25_path: Path = PROJECT_ROOT / "data" / "bm25_index.pkl"
    trace_db: Path = PROJECT_ROOT / "data" / "traces.sqlite3"
    memory_dir: Path = PROJECT_ROOT / "data" / "memory"

    # ---- 检索 ----
    top_k: int = 6
    bm25_top_k: int = 20
    hybrid_alpha: float = 0.5  # 0=纯向量，1=纯 BM25
    rerank: bool = True
    rerank_top_n: int = 3

    # ---- Agent ----
    max_iterations: int = 8

    # ---- API 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def embedding_ready(self) -> bool:
        return bool(self.embedding_api_key)


settings = Settings()
