import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai  import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

from chatsky_llm_autoconfig.algorithms.base import GraphGenerator
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from chatsky_llm_autoconfig.schemas import DialogueGraph, DialogueNodes
from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.utils import call_llm_api, nodes2graph
from chatsky_llm_autoconfig.settings import EnvSettings
from chatsky_llm_autoconfig.missing_edges_prompt import three_1, three_2
from chatsky_llm_autoconfig.prompts import (
 graph_example_1, part_1, part_2_v3
)
from chatsky_llm_autoconfig.metrics.automatic_metrics import (
    is_same_structure,
    compare_graphs
)


env_settings = EnvSettings()

@AlgorithmRegistry.register(input_type=list[Dialogue], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of diaolgues.
    Thee stages:
    1. Algorithmic grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """
    prompt_name: str = ""
    model_name: str = ""

    def __init__(self, model_name, prompt_name: str=""):
        super().__init__()
        self.prompt_name = prompt_name
        self.model_name = model_name

    def invoke(self, dialogue: list[Dialogue] = None, graph: DialogueGraph = None, topic: str = "") -> BaseGraph:

        partial_variables = {}
        prompt_extra = part_2_v3
        for idx, dial in enumerate(dialogue):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(template=part_1+"{graph_example_1}. "+prompt_extra, input_variables=["graph_example_1"], partial_variables=partial_variables)

        base_model = ChatOpenAI(model=self.model_name, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL)
        model = base_model | PydanticOutputParser(pydantic_object=DialogueNodes)
        nodes = call_llm_api(prompt.format(graph_example_1=graph_example_1), model, temp=0).model_dump()

        last_user = False

        for idx in range(len(nodes['nodes'])):
            nodes['nodes'][idx]['utterances'] = list(set(nodes['nodes'][idx]['utterances']))
        print("NODES: ", nodes)
        embeddings = HuggingFaceEmbeddings(model_name=env_settings.EMBEDDER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})
        try:
            graph_dict = nodes2graph(nodes['nodes'], dialogue, embeddings)
        except Exception as e:
            print(e)
            return Graph({})
        print("RESULT: ", graph_dict, "\n")
        graph_dict = {"nodes": graph_dict['nodes'], "edges": graph_dict['edges'], "reason": ""}

        try:
            if not last_user:
                result_graph = Graph(graph_dict=graph_dict)
                return result_graph

            partial_variables = {}
            prompt_extra = ""
            for idx, dial in enumerate(dialogue):
                partial_variables[f"var_{idx}"] = dial.to_list()
                prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
            prompt = PromptTemplate(template=three_1+"{graph_dict}. "+three_2+prompt_extra, input_variables=["graph_dict"], partial_variables=partial_variables)

            print("PROMPT: ", prompt)

            model = base_model | PydanticOutputParser(pydantic_object=DialogueGraph)

            result = call_llm_api(prompt.format(graph_dict=graph_dict), model, temp=0)
            if result is None:
                return Graph(graph_dict={})
            result.reason = "Fixes: " + result.reason
            graph_dict=result.model_dump()
            if not all([e['target'] for e in graph_dict['edges']]):
                return Graph(graph_dict={})
            result_graph = Graph(graph_dict=graph_dict)
            return result_graph
        except Exception as e:
            print(e)
            return Graph({})

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
    
    async def evaluate(self, dialogues, target_graph, report_type = "dict"):
        graph = self.invoke(dialogues)
        report = {
            "is_same_structure": is_same_structure(graph, target_graph),
            "graph_match": compare_graphs(graph, target_graph),
        }
        if report_type == "dataframe":
            report = pd.DataFrame(report, index=[0])
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")
