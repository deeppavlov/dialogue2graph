import yaml
import logging
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


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
        with open(path, "r") as f:
            loaded_storage = yaml.safe_load(f)
            for key, config in loaded_storage.items():
                self.add(key=key, config=config, model_type=config.pop("model_type"))
                logger.info(f"Loaded {key} from {path}")

    def add(self, key: str, config: dict, model_type: Union[Literal["llm"], Literal["emb"]]):
        """
        Add a new model configuration to the storage.
        Args:
            key (str): The unique identifier for the model configuration.
            config (dict): The configuration dictionary for initializing the model.
            model_type (Union[Literal["llm"], Literal["emb"]]): The type of the model to be added.
                - "llm": Large Language Model, initialized using `ChatOpenAI`.
                - "emb": Embedding model, initialized using `HuggingFaceEmbeddings`.
        Raises:
            Warning: Logs a warning if the key already exists in the storage, indicating that the existing entry will be overwritten.
        Side Effects:
            - Initializes the model based on the provided configuration and type.
            - Stores the model and its configuration in the storage under the specified key.
        """
        if key in self.storage:
            logger.warning(f"Key {key} already exists in storage. Overwriting.")
        if model_type == "llm":
            model = ChatOpenAI(**item.config)
        elif model_type == "emb":
            # remove "device" from config and pass it to the "model_kwargs" parameter
            device = item.config.pop("device", None)
            item.config["model_kwargs"] = {"device": device}
            model = HuggingFaceEmbeddings(**item.config)

        item = StoredData(key=key, config=config, model_type=model_type, model=model)

        self.storage[key] = item

    def save(self, path: str):
        """
        Save the current model storage to a YAML file.

        Args:
            path (str): The file path where the storage data will be saved.

        The method serializes the `storage` attribute, which is expected to be a dictionary
        where the values have a `config.model_dump()` method. The serialized data is then
        written to the specified file in YAML format.
        """
        with open(path, "w") as f:
            storage_dump = {k: v.config for k, v in self.storage.items()}
            yaml.dump(storage_dump, f)
