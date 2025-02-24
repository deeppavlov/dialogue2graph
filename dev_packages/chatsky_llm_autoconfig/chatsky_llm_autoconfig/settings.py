from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Any
import os

class EnvSettings(BaseSettings, case_sensitive=True):

    # model_config: SettingsConfigDict
    # model_config = SettingsConfigDict(env_file='dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/.env', env_file_encoding='utf-8')
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: Optional[str]
    GENERATION_MODEL_NAME: Optional[str]
    COMPARE_MODEL_NAME: Optional[str]
    GENERATION_SAVE_PATH: Optional[str]
    METRICS_SAVE_PATH: Optional[str]
    FORMATTER_MODEL_NAME: Optional[str]
    TEST_DATA_PATH: Optional[str]
    RESULTS_PATH: Optional[str]
    EMBEDDER_MODEL: Optional[str]
    EMBEDDER_THRESHOLD: Optional[float]
    ONE_WORD_TH: Optional[float]
    EMBEDDER_TYPO: Optional[float]
    DIALOGUE_MAX: Optional[int]
    EMBEDDER_DEVICE: Optional[str]
    RERANKER_MODEL: Optional[str]
    RERANKER_THRESHOLD: Optional[float]
    NEXT_RERANKER_THRESHOLD: Optional[float]
    SIM_THRESHOLD: Optional[float]
    HUGGINGFACE_TOKEN: Optional[str]
    TEST_DATASET: Optional[str]
    GRAPH_SAVED: Optional[str]
    GRAPHS_TO_FIX: Optional[str]
    TOPICS_DATA: Optional[str]
