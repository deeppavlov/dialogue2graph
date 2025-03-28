# from d2g import ModelStorage, Pipeline1, Pipeline2


# ms = ModelStorage()

# # ms.load("file.yaml")

# ms.add("custom_model_key1", config={})

# ms.add("mix:version1112", config={})

# ms.add("woz/bge", config={"device": 1, "model_type": "emb"})


# pipe1 = Pipeline1(ms)  # graph_gen_model="gpt:version1111"

# pipe2 = Pipeline2(ms, graph_gen_model2=None)

# pipe3 = Pipeline3(ms, graph_gen_model1=None, graph_gen_model2=None, emb_model="woz/bge")


# ms.save("file.yaml")
import yaml
from typing import Literal, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings


class ModelStorage(BaseModel):

    storage: dict = Field(default_factory=dict)

    def load(self, path: str):
        raise NotImplementedError

    def add(self, key: str, config: dict, model_type: Union[Literal["llm"], Literal["emb"]]):
        if model_type == "llm":
            model = ChatOpenAI(**config)
        elif model_type == "emb":
            model = HuggingFaceEmbeddings(**config)

        self.storage[key] = {"model": model, "config": config}

    def save(self, path: str):
        with open(path, "w") as f:
            # do not dump the model object
            storage_dump = {k: v["config"] for k, v in self.storage.items()}
            yaml.dump(storage_dump, f)
