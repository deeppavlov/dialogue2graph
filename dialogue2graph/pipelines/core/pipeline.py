from typing import Union
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.algorithms import DialogAugmentation, DialogueGenerator, GraphGenerator


class Pipeline(BaseModel):
    steps: list[Union[DialogueGenerator, DialogAugmentation, GraphGenerator]] = Field(default_factory=list)

    def _validate_pipeline(self):
        pass

    def invoke(self, data):
        for step in self.steps:
            output = step.invoke(data)
            data = output
        return output
