from langchain.prompts import PromptTemplate
from langchain_openai  import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

from chatsky_llm_autoconfig.algorithms.base import GraphGenerator
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from chatsky_llm_autoconfig.metrics.embedder import nodes2groups
from chatsky_llm_autoconfig.schemas import DialogueGraph
from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.utils import call_llm_api, nodes2graph, dialogues2list
from chatsky_llm_autoconfig.settings import EnvSettings

from chatsky_llm_autoconfig.missing_edges_prompt import three_1, three_2

env_settings = EnvSettings()

embeddings = HuggingFaceEmbeddings(model_name=env_settings.EMBEDDER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})

@AlgorithmRegistry.register(input_type=list[Dialogue], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of diaolgues.
    Thee stages:
    1. Algorithmic grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """
    # prompt_name: str = ""

    # def __init__(self, prompt_name: str=""):
    #     super().__init__()
    #     self.prompt_name = prompt_name

    def invoke(self, dialogues: list[Dialogue] = None, graph: DialogueGraph = None, topic: str = "") -> BaseGraph:

        base_model = ChatOpenAI(model=env_settings.GENERATION_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=1)
        nexts, nodes, starts, neigbhours, last_user = dialogues2list(dialogues)
        
        print("LISTS_N: ",[(i,n) for i,n in enumerate(nexts)])
        print("LISTS: ",[(i,n) for i,n in enumerate(nodes)])

        groups = nodes2groups(dialogues, nodes, [" ".join(p) for p in nexts], [n+ " ".join(p) + " " for p,n in zip(nexts, nodes)], neigbhours)

        print ("NODES: ", groups)
        # print ("MIX: ", mix)
        nodes = []
        for idx, group in enumerate(groups):
            if any([gr in starts for gr in group]):
                start = True
            else:
                start = False
            nodes.append({"id":idx+1, "label": "", "is_start": start, "utterances": group})

        print("NODES: ", nodes)
        # embeddings = HuggingFaceEmbeddings(model_name=env_settings.EMBEDDER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})
        graph_dict = nodes2graph(nodes, dialogues, embeddings)
        print("RESULT: ", graph_dict, "\n")
        graph_dict = {"nodes": graph_dict['nodes'], "edges": graph_dict['edges'], "reason": ""}

        # result_graph = Graph(graph_dict=graph_dict)
        # print("SKIP")
        # return result_graph 

        if not last_user:
            result_graph = Graph(graph_dict=graph_dict)
            print("SKIP")
            return result_graph    
        partial_variables = {}
        prompt_extra = ""
        for idx, dial in enumerate(dialogues):
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

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
