import logging
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

from dialog2graph.pipelines.core.algorithms import GraphGenerator
from dialog2graph.metrics.llm_metrics import compare_graphs
from dialog2graph.metrics.no_llm_metrics import is_same_structure
from dialog2graph.pipelines.core.schemas import DialogGraph
from dialog2graph.pipelines.core.dialog import Dialog
from dialog2graph.pipelines.core.graph import BaseGraph, Graph

from embedder import nodes2groups
from utils import call_llm_api, nodes2graph, dialogs2list
from settings import EnvSettings
from missing_edges_prompt import three_1, three_2


env_settings = EnvSettings()
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": env_settings.DEVICE}
)


# @AlgorithmRegistry.register(input_type=list[Dialog], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of dialogs.
    Thee stages:
    1. Algorithmic grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogs ends with user's utterance, ask LLM to add missing edges.
    """

    def invoke(
        self,
        dialogs: list[Dialog] = None,
        graph: DialogGraph = None,
        model_name="chatgpt-4o-latest",
        temp=0,
    ) -> BaseGraph:
        base_model = ChatOpenAI(
            model=model_name,
            api_key=env_settings.OPENAI_API_KEY,
            base_url=env_settings.OPENAI_BASE_URL,
            temperature=temp,
        )
        nexts, nodes, starts, neigbhours, last_user = dialogs2list(dialogs)

        print("LISTS_N: ", [(i, n) for i, n in enumerate(nexts)])
        print("LISTS: ", [(i, n) for i, n in enumerate(nodes)])

        groups = nodes2groups(
            dialogs,
            nodes,
            [" ".join(p) for p in nexts],
            [n + " ".join(p) + " " for p, n in zip(nexts, nodes)],
            neigbhours,
        )

        print("NODES: ", groups)
        # print ("MIX: ", mix)
        nodes = []
        for idx, group in enumerate(groups):
            if any([gr in starts for gr in group]):
                start = True
            else:
                start = False
            nodes.append(
                {"id": idx + 1, "label": "", "is_start": start, "utterances": group}
            )

        print("NODES: ", nodes)
        graph_dict = nodes2graph(nodes, dialogs, embeddings)
        print("RESULT: ", graph_dict, "\n")
        graph_dict = {
            "nodes": graph_dict["nodes"],
            "edges": graph_dict["edges"],
            "reason": "",
        }

        if not last_user:
            result_graph = Graph(graph_dict=graph_dict)
            print("SKIP")
            return result_graph
        partial_variables = {}
        prompt_extra = ""
        for idx, dial in enumerate(dialogs):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialog_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(
            template=three_1 + "{graph_dict}. " + three_2 + prompt_extra,
            input_variables=["graph_dict"],
            partial_variables=partial_variables,
        )

        print("PROMPT: ", prompt)

        model = base_model | PydanticOutputParser(pydantic_object=DialogGraph)

        result = call_llm_api(prompt.format(graph_dict=graph_dict), model, temp=temp)
        print("OUT: ", result)
        if result is None:
            return Graph(graph_dict={})
        result.reason = "Fixes: " + result.reason
        graph_dict = result.model_dump()
        if not all([e["target"] for e in graph_dict["edges"]]):
            return Graph(graph_dict={})
        result_graph = Graph(graph_dict=graph_dict)
        return result_graph

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    async def evaluate(self, dialogs, target_graph, report_type="dict"):
        graph = self.invoke(dialogs)
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
