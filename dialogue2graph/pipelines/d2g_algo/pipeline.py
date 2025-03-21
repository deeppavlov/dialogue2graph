# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
import json
from typing import List
from pathlib import PosixPath
from pydantic import BaseModel, TypeAdapter, ValidationError
from dialogue2graph.pipelines.core.dialogue import Dialogue, DialogueMessage
from dialogue2graph.pipelines.core.graph import Graph
from .three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator

def parse_data(data: Dialogue|list[Dialogue]|dict|list[list]|list[dict]|str) -> List[Dialogue]:

    try:
        validation = TypeAdapter(Dialogue|List[DialogueMessage]|List[List[DialogueMessage]]|PosixPath|List[Dialogue]).validate_python(data)
    except ValidationError:
        return False
    if isinstance(validation, PosixPath) and data.endswith(".json"):
        with open(data) as f:
            dialogues = json.load(f)
            if isinstance(dialogues, dict):
                dialogues = [dialogues]
    elif isinstance(validation, Dialogue):
            dialogues = [validation]
    elif isinstance(validation, List) and len(data) > 0:
        if isinstance(validation[0], Dialogue):
            dialogues = validation
        elif isinstance(validation[0], DialogueMessage):
            dialogues = [Dialogue(messages=validation)]
        elif isinstance(validation[0], List) and isinstance(validation[0][0], DialogueMessage):
            dialogues = [Dialogue(messages=dialogue) for dialogue in validation]
    else:
        dialogues = []
    return dialogues



# class Pipeline(BasePipeline):
class Pipeline(BaseModel):
    _preloaded_generators = {}

    def _validate_pipeline(self):
        pass

    def invoke(self, data: Dialogue|list[Dialogue]|dict|list[list]|list[dict]) -> Graph:

        if "algo" not in self._preloaded_generators:
            self._preloaded_generators["algo"] = AlgoGenerator()
        dialogues = parse_data(data)

        graph = self._preloaded_generators["algo"].invoke(dialogues)
        return graph