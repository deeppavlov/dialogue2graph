# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from pydantic import BaseModel
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.cycled_graphs.three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator
from dialogue2graph.pipelines.cycled_graphs.three_stages_llm import ThreeStagesGraphGenerator as LLMGenerator
from dialogue2graph.pipelines.cycled_graphs.three_stages_extender import ThreeStagesGraphGenerator as Extender


# class Pipeline(BasePipeline):
class Pipeline(BaseModel):
    _preloaded_generators = {}

    def _validate_pipeline(self):
        pass

    def invoke(self, data: Dialogue|list[Dialogue]|Graph, config_name="algo") -> Graph:

        if config_name not in self._preloaded_generators:
            if config_name == "algo":
                self._preloaded_generators[config_name] = AlgoGenerator()
            elif config_name == "llm":
                self._preloaded_generators[config_name] = LLMGenerator()
            elif config_name == "extender":
                self._preloaded_generators[config_name] = Extender()
                if "algo" not in self._preloaded_generators:
                    self._preloaded_generators["algo"] = AlgoGenerator()
        if isinstance(data, Graph):
            if "sampler" not in self._preloaded_generators:
                self._preloaded_generators["sampler"] = RecursiveDialogueSampler()
            dialogues = self._preloaded_generators["sampler"].invoke(data,15)
        elif isinstance(data, Dialogue):
            dialogues = [data]
        elif len(data) > 0 and isinstance(data[0], Dialogue):
            dialogues = data
        elif len(data) > 0 and isinstance(data[0], dict):
            dialogues = [Dialogue().from_list(d['messages']) for d in data]

        if config_name in ["algo","llm"]:
            graph = self._preloaded_generators[config_name].invoke(dialogues)
        elif config_name == "extender":
            graph = self._preloaded_generators["algo"].invoke(dialogues[:1])
            for idx in range(1,len(data)):
                graph, _ = self._preloaded_generators[config_name].invoke(dialogues[idx:idx+1],graph)

        return graph