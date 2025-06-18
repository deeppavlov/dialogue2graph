"""ModelStorage module for managing and storing models.
This module provides classes and functions for managing and storing models, including loading configurations.
"""

import yaml
import re
import dotenv
from typing import Union, Dict, Type
from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from dialog2graph.utils.logger import Logger

logger = Logger(__name__)

dotenv.load_dotenv()


class GetModelInstance:
    """Model instantiation utility class.
    This class provides methods to load and instantiate models based on their configuration.
    Attributes:
        config (dict): Configuration dictionary containing details about the model.
    Methods:
        instantiate(class_name):
            Instantiates a model based on the provided class name and configuration.
    """

    config: dict

    def __init__(self, config: dict):
        """
        Initializes a GetModelInstance object with the given configuration.

        Args:
            config (dict): Configuration dictionary containing details about the model to be instantiated.
        """
        self.config = config

    def instantiate(self, class_name) -> BaseChatModel | HuggingFaceEmbeddings:
        """
        Instantiates a model based on the provided class name and configuration.

        Args:
            class_name: The class name of the model to be instantiated.

        Returns:
            The instantiated model object.
        """
        return class_name(**self.config)


class StoredData(BaseModel):
    """
    StoredData is a Pydantic model that represents the storage structure for a model, its configuration, and metadata.

    Attributes:
        key (str): Key for the stored model.
        config (dict): Configuration for the stored model.
        model_type (Type[BaseChatModel | HuggingFaceEmbeddings]): Type of the stored model, for example ChatOpenAI, HuggingFaceEmbeddings.
        model (Union[HuggingFaceEmbeddings, BaseChatModel]): The actual model object, which can either be a HuggingFaceEmbeddings instance or a BaseChatModel instance.

    Methods:
        validate_model(cls, values):
            Validates the `model` attribute based on the `model_type`. Ensures that:
            - If `model_type` is "llm", the `model` must be an instance of BaseChatModel.
            - If `model_type` is "emb", the `model` must be an instance of HuggingFaceEmbeddings.
            Raises:
                ValueError: If the `model` does not match the expected type for the given `model_type`.
    """

    key: str = Field(description="Key for the stored model")
    config: dict = Field(description="Configuration for the stored model")
    model_type: Type[BaseChatModel | HuggingFaceEmbeddings] = Field(description="Type of the stored model")
    model: Union[
        HuggingFaceEmbeddings,
        BaseChatModel,
    ] = Field(description="Model object")

    @model_validator(mode="before")
    def validate_model(cls, values):
        if values.get("model_type") == ChatOpenAI:
            if not isinstance(values.get("model"), BaseChatModel):
                raise ValueError("LLM model must be an instance of BaseChatModel")
        elif values.get("model_type") == HuggingFaceEmbeddings:
            if not isinstance(values.get("model"), HuggingFaceEmbeddings):
                raise ValueError(
                    "HuggingFaceEmbeddings model must be an instance of HuggingFaceEmbeddings"
                )
        return values

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = False


class ModelStorage(BaseModel):
    """
    ModelStorage is a class for managing the storage of model configurations and instances.
    It provides functionality to load configurations from a YAML file, add new models to the storage,
    and save the current storage state back to a YAML file.

    Attributes:
        storage (Dict[str, StoredData]): A dictionary that holds the stored model configurations
            and their corresponding instances.
    """

    storage: Dict[str, StoredData] = Field(default_factory=dict)

    def load(self, path: Path):
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
                    self.add(
                        key=key,
                        config=config.pop("config"),
                        model_type=eval(config.pop("model_type")),
                    )
                    logger.debug(f"Loaded model configuration for '{key}'")
            logger.info(f"Successfully loaded {len(loaded_storage)} models from {path}")
        except Exception as e:
            logger.error(f"Failed to load model configurations from {path}: {e}")
            raise

    def add(
        self,
        key: str,
        config: dict,
        model_type: Type[BaseChatModel | HuggingFaceEmbeddings],
        overwright: bool = False,
    ):
        """
                Add a new model configuration to the storage.
                The configuration is validated based on the specified model_type.
                If the configuration is valid, a new StoredData object is created and added to the storage.
                If a model with the same key already exists, the overwright parameter determines whether to replace it.
                If overwright is False and the model already exists, the existing model will not be replaced.
                Otherwise, the model will not be added to the storage.
                In case of failure to add the model logs the error.

                Args:
                    key (str): The unique identifier for the model configuration.
                    config (dict): The configuration dictionary for initializing the model.
                    model_type (Type[BaseChatModel | HuggingFaceEmbeddings]): The type of the model to be added.
                    overwright (bool): Whether to overwright model existing under same key
        .
                Raises:
                    KeyError: If configuration keys are invalid for the specified model_type.
        """
        logger.debug("Current storage keys: %s", list(self.storage.keys()))
        if key in self.storage:
            if overwright:
                logger.warning(f"Key '{key}' already exists in storage. Overwriting.")
            else:
                if (
                    self.storage[key].model_type == model_type
                    and self.storage[key].config == config
                ):
                    logger.warning(
                        f"Key '{key}' already exists in storage with same config. Skipping."
                    )
                else:
                    logger.warning(
                        f"Key '{key}' already exists in storage with different model type or config. Skipping."
                    )
                return
        try:
            logger.debug(
                "Initializing model %s for key '%s' with config: %s"
                % (model_type, key, config)
            )
            if "name" in config:
                raise KeyError(
                    f"Instead of 'name' parameter for model {key} of type {model_type} please use 'model_name'"
                )
            if not all(p in model_type.model_fields.keys() for p in config):
                raise KeyError(
                    f"Invalid parameter names for model '{key}': {[p for p in config if p not in model_type.model_fields.keys()]}"
                )
            model_getter = GetModelInstance(config)
            model_instance = model_getter.instantiate(model_type)

            logger.debug("Created model instance of type: %s", type(model_instance))

            item = StoredData(
                key=key, config=config, model_type=model_type, model=model_instance
            )
            self.storage[key] = item
            logger.info(f"Added {model_type} model '{key}' to storage")
        except Exception as e:
            logger.error(f"Failed to add model '{key}' to storage: {e}", exc_info=True)
            raise

    def save(self, path: str):
        """
        Save the current model storage to a YAML file.
        In case of failure logs the error.

        Args:
            path (str): The file path where the storage data will be saved.
        """
        logger.debug(f"Attempting to save model storage to {path}")
        try:
            with open(path, "w") as f:
                storage_dump = {}
                for model_key in self.storage:
                    storage_dump[model_key] = {}
                    storage_dump[model_key]["config"] = self.storage[model_key].config
                    storage_dump[model_key]["model_type"] = (
                        re.sub(r"<class '", "", str(self.storage[model_key].model_type))
                        .replace("'>", "")
                        .split(".")[-1]
                    )
                    keys_to_pop = []
                    for key in storage_dump[model_key]["config"]:
                        if "api_key" in key or "api_base" in key or "base_url" in key:
                            keys_to_pop.append(key)
                    for key in keys_to_pop:
                        storage_dump[model_key]["config"].pop(key, None)
                yaml.dump(storage_dump, f)
            logger.info(f"Saved {len(self.storage)} models to {path}")
        except Exception as e:
            logger.error(f"Failed to save model storage to {path}: {e}")
            raise
