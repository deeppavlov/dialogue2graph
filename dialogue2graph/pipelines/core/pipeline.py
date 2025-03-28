from typing import Union, Optional
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.core.algorithms import DialogAugmentation, DialogueGenerator, GraphGenerator, GraphExtender, InputParser


class Pipeline(BaseModel):
    model_storage: Optional[ModelStorage] = Field(default_factory=ModelStorage)
    steps: list[Union[InputParser, DialogueGenerator, DialogAugmentation, GraphGenerator, GraphExtender]] = Field(default_factory=list)

    def _validate_pipeline(self):
        """
        Check if input and output types of steps are compatible.
        """
        raise NotImplementedError

    def invoke(self, data):
        for step in self.steps:
            output = step.invoke(data)
            data = output
        return output
