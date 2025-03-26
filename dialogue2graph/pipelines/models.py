import logging
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ModelsAPI(metaclass=Singleton):
    """Singleton for preloaded LLMs and embedders"""

    preloaded_models: dict = {}

    def __call__(self, model_type: str, **model_kwargs) -> BaseChatModel | HuggingFaceEmbeddings | None:
        model_name = model_kwargs["name"]
        if model_type == "llm":
            parameter = model_kwargs["temp"]
        elif model_type == "similarity":
            parameter = model_kwargs["device"]
        else:
            logger.error("Wrong model type")
            return None

        if model_name not in self.preloaded_models or parameter not in self.preloaded_models[model_name]:
            if model_type == "llm":
                model = ChatOpenAI(model=model_name, temperature=parameter)
            else:
                model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": parameter})
        if model_name in self.preloaded_models:
            if parameter not in self.preloaded_models[model_name]:
                self.preloaded_models[model_name].append({parameter: model})
        else:
            self.preloaded_models[model_name] = {parameter: model}
        return self.preloaded_models[model_name][parameter]
