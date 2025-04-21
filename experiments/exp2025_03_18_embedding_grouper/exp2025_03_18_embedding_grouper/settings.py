from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class EnvSettings(BaseSettings, case_sensitive=True):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: Optional[str]
    DEVICE: Optional[str]
    HUGGINGFACE_TOKEN: Optional[str]
    EMBEDDER_TYPO: Optional[float]
    RERANKER_MODEL: Optional[str]
    RERANKER_THRESHOLD: Optional[float]
    NEXT_RERANKER_THRESHOLD: Optional[float]
