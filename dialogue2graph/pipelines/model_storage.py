import yaml
import logging
from typing import Literal, Union, Dict
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class StoredData(BaseModel):
    key: str = Field(description="Key for the stored model")
    config: dict = Field(description="Configuration for the stored model")
    model_type: Union[Literal["llm"], Literal["emb"]] = Field(description="Type of the stored model")
    model: Union[HuggingFaceEmbeddings, BaseChatModel] = Field(description="Model object")

    @model_validator(mode="before")
    def validate_model(cls, values):
        if values.get("model_type") == "llm":
            if not isinstance(values.get("model"), BaseChatModel):
                raise ValueError("LLM model must be an instance of BaseChatModel")
        elif values.get("model_type") == "emb":
            if not isinstance(values.get("model"), HuggingFaceEmbeddings):
                raise ValueError("Embedding model must be an instance of HuggingFaceEmbeddings")
        return values

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = False


class ModelStorage(BaseModel):

    storage: Dict[str, StoredData] = Field(default_factory=dict)

    def load(self, path: str):
        """
        Load model configurations from a YAML file into the storage.

        Args:
            path (str): The file path to the YAML file containing model configurations.
        """
        logger.debug(f"Attempting to load model configurations from {path}")
        try:
            with open(path, "r") as f:
                loaded_storage = yaml.safe_load(f)
                for key, config in loaded_storage.items():
                    self.add(key=key, config=config, model_type=config.pop("model_type"))
                    logger.debug(f"Loaded model configuration for '{key}'")
            logger.info(f"Successfully loaded {len(loaded_storage)} models from {path}")
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
        logger.debug("Current storage keys: %s", list(self.storage.keys()))
        if key in self.storage:
            logger.warning(f"Key '{key}' already exists in storage. Overwriting.")
        try:
            if model_type == "llm":
                logger.debug(f"Initializing LLM model for key '{key}' with config: {config}")
                model_instance = ChatOpenAI(**config)
            elif model_type == "emb":
                device = config.pop("device", None)
                if device:
                    config["model_kwargs"] = {"device": device}
                logger.debug(f"Initializing embedding model for key '{key}' with config: {config}")
                model_instance = HuggingFaceEmbeddings(**config)

            logger.debug("Created model instance of type: %s", type(model_instance))
            item = StoredData(key=key, config=config, model_type=model_type, model=model_instance)
            self.storage[key] = item
            logger.info(f"Added {model_type} model '{key}' to storage")
        except Exception as e:
            logger.error(f"Failed to add model '{key}' to storage: {e}", exc_info=True)
            raise

    def save(self, path: str):
        """
        Save the current model storage to a YAML file.

        Args:
            path (str): The file path where the storage data will be saved.
        """
        logger.debug(f"Attempting to save model storage to {path}")
        try:
            with open(path, "w") as f:
                storage_dump = {k: v.config for k, v in self.storage.items()}
                yaml.dump(storage_dump, f)
            logger.info(f"Saved {len(self.storage)} models to {path}")
        except Exception as e:
            logger.error(f"Failed to save model storage to {path}: {e}")
            raise
