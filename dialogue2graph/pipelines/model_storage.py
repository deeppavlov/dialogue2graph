import yaml
import logging
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class StoredData(BaseModel):
    key: str = Field(description="Key for the stored model")
    config: dict = Field(description="Configuration for the stored model")
    model_type: Union[Literal["llm"], Literal["emb"]] = Field(description="Type of the stored model")
    model: Union[BaseChatModel, HuggingFaceEmbeddings] = Field(description="Model object", default=None)


class ModelStorage(BaseModel):

    storage: dict[str, StoredData] = Field(default_factory=dict)

    def load(self, path: str):
        """
        Load model configurations from a YAML file into the storage.

        Args:
            path (str): The file path to the YAML file containing model configurations.
        """
        logger.info(f"Attempting to load model configurations from {path}")
        try:
            with open(path, "r") as f:
                loaded_storage = yaml.safe_load(f)
                for key, config in loaded_storage.items():
                    self.add(key=key, config=config, model_type=config.pop("model_type"))
                    logger.info(f"Successfully loaded model '{key}' from {path}")
        except Exception as e:
            logger.error(f"Failed to load model configurations from {path}: {e}")
            raise

    def add(self, key: str, config: dict, model_type: Union[Literal["llm"], Literal["emb"]]):
        """
        Add a new model configuration to the storage.
        Args:
            key (str): The unique identifier for the model configuration.
            config (dict): The configuration dictionary for initializing the model.
            model_type (Union[Literal["llm"], Literal["emb"]]): The type of the model to be added.
                - "llm": Large Language Model, initialized using `ChatOpenAI`.
                - "emb": Embedding model, initialized using `HuggingFaceEmbeddings`.
        """
        if key in self.storage:
            logger.warning(f"Key '{key}' already exists in storage. Overwriting.")
        try:
            if model_type == "llm":
                logger.info(f"Initializing LLM model for key '{key}' with config: {config}")
                model = ChatOpenAI(**config)
                logger.info(f"Initialized LLM model for key '{key}'")
            elif model_type == "emb":
                device = config.pop("device", None)
                config["model_kwargs"] = {"device": device}
                logger.info(f"Initializing embedding model for key '{key}' with config: {config}")
                model = HuggingFaceEmbeddings(**config)
                logger.info(f"Initialized embedding model for key '{key}'")

            item = StoredData(key=key, config=config, model_type=model_type, model=model)
            self.storage[key] = item
            logger.info(f"Successfully added model '{key}' to storage")
        except Exception as e:
            logger.error(f"Failed to add model '{key}' to storage: {e}")
            raise

    def save(self, path: str):
        """
        Save the current model storage to a YAML file.

        Args:
            path (str): The file path where the storage data will be saved.
        """
        logger.info(f"Attempting to save model storage to {path}")
        try:
            with open(path, "w") as f:
                storage_dump = {k: v.config for k, v in self.storage.items()}
                yaml.dump(storage_dump, f)
            logger.info(f"Successfully saved model storage to {path}")
        except Exception as e:
            logger.error(f"Failed to save model storage to {path}: {e}")
            raise
