import time
from typing import Union
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.algorithms import DialogAugmentation, DialogueGenerator, GraphGenerator, GraphExtender, InputParser, DataParser


class PipelineRawDataType(BaseModel):
    dialgos: RawDialogsType
    supported_graph: Optional[RawGraphType]
    true_graph: Optional[RawGraphType]
class PipelineDataType(BaseModel):
    dialgos: list[Dialogue]
    supported_graph: Optional[Graph]
    true_graph: Optional[Graph]
class Pipeline(BaseModel):
    steps: list[Union[InputParser, DialogueGenerator, DialogAugmentation, GraphGenerator, GraphExtender]] = Field(default_factory=list)
    
    steps = list[InputParser(), DialogueGenerator(model_sig="asdasd", start_evals=[PreDGEvalBase], end_evals=[DGEvalBase], enable_evals=False), DialogAugmentation(end_evals=[DGEvalBase]), GraphGenerator(step1_evals=[DGEvalBase]), GraphExtender]]

    def _validate_pipeline(self):
        pass

    def invoke(self, raw_data: PipelineRawDataType, enable_evals=False):
        data:PipelineDataType = DataParser().invoke(raw_data)
        report = Report(service =self.name)
        st_time = time.time()
        output = data
        for step in self.steps:
            output, report = step.invoke(output, enable_evals=enable_evals)
            report.add_subreport(report)
        end_time = time.time()
        report.add_property("time", end_time - st_time)
        report.add_property("simple_graph_comparison", simple_graph_comparison(output, data))
        if enable_evals:
            report.add_property("complex_graph_comparison", complex_graph_comparison(output, data))
        
        return output, report
