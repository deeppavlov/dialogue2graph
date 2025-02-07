from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import matplotlib.pyplot as plt
import random
import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd
import asyncio
import networkx as nx
load_dotenv()


with open("prompts/prompt_v3.txt", 'r') as f:
    gen_prompt = ChatPromptTemplate.from_template(f.read())

with open("prompts/prompt_augmentation.txt", 'r') as f:
    aug_prompt = ChatPromptTemplate.from_template(f.read())

with open("/home/askatasuna/Документы/DeepPavlov/chatsky-llm-autoconfig/data/data.json") as f:
    graph_templates = {}
    template_data = json.load(f)
    for g in template_data:
        graph_type = g['samping_method']
        if graph_type not in graph_templates:
            graph_templates[graph_type] = {}
            graph_templates[graph_type]['target'] = g['target_graph']
            empty_graph = g['target_graph']
            for node in empty_graph['nodes']:
                node['utterances'] = []  # Set utterances to an empty list
                node['label'] = ""       # Set label to an empty string
            for edge in empty_graph['edges']:
                edge['utterances'] = []
            graph_templates[graph_type]['empty'] = empty_graph


model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)


async def generate_dialogue_graphs_from_templates(graph_templates: dict, retries: int):
    data = {}
    for graph_type in tqdm(graph_templates):
        success = False
        counter = 0
        while not success and counter <= retries:
            try:
                result = await model.ainvoke(gen_prompt.format(SCHEMA=graph_templates[graph_type]['empty'], TARGET=graph_templates[graph_type]['target']))
                res = json.loads(result.content)
                success = True
                data[graph_type] = res
            except json.JSONDecodeError:
                print("Error: Invalid JSON response. Retrying...")
                counter += 1
    
    with open("raw_data.json", 'w') as f:
        f.write(json.dumps(data, indent=2))

    return data


async def augment_data(graph: dict, themes: set, amount: int=5) -> list[dict]:
    res = []
    for theme in themes:
        for _ in tqdm(range(amount)):
            result = await model.ainvoke(aug_prompt.format(THEME=theme, graph=graph))
            try:
                g = json.loads(result.content)
                res.append(g)
            except Exception:
                print("invalid graph")

    with open("augmented_graphs.json", 'w') as f:
        f.write(json.dumps(res, indent=2))
    
    return res

def dialogues_from_graph(graph, include_readable: bool=False):
    # Create a directed graph from the JSON input
    G = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for node in graph['nodes']:
        # Ensure 'is_start' is included in the node attributes
        G.add_node(node['id'], label=node['label'], utterances=node['utterances'], is_start=node.get('is_start', False))
    
    for edge in graph['edges']:
        G.add_edge(edge['source'], edge['target'], utterances=edge['utterances'])
    
    # Start the dialogue from the starting node
    start_nodes = [node for node in G.nodes if G.nodes[node].get('is_start', True)]
    if not start_nodes:
        raise ValueError("No starting node found in the graph.")
    
    current_node = random.choice(start_nodes)  # Randomly select one of the starting nodes
    dialogue = []
    visited_nodes = set()  # Track visited nodes to prevent cycles
    
    # Generate a dialogue by traversing the graph
    while True:
        # Get the current node's utterances
        node_utterances = G.nodes[current_node]['utterances']
        # print("NODE:", node_utterances)
        dialogue.append(random.choice(node_utterances))  # Randomly select an utterance
        
        # Mark the current node as visited
        visited_nodes.add(current_node)
        
        # Get the next possible nodes to traverse
        next_nodes = list(G.successors(current_node))
        if not next_nodes:  # If there are no successors, break the loop
            break
        
        # Filter out already visited nodes to prevent cycles
        next_nodes = [node for node in next_nodes if node not in visited_nodes]
        if not next_nodes:  # If all next nodes have been visited, break the loop
            break
        
        next_node = random.choice(next_nodes)
        
        # Get the edge utterances between the current node and the next node
        edge_utterances = G[current_node][next_node]['utterances']
        # print("EDGE:", edge_utterances)
        dialogue.append(random.choice(edge_utterances)) 
        # Randomly select the next node to traverse
        current_node = next_node
    
    if include_readable:
        out = ""
        for i in range(len(dialogue)):
            if i % 2 == 0:
                out += f"ASSISTANT: {dialogue[i]}\n"
            else:
                out += f"USER: {dialogue[i]}\n"
        return (out, dialogue)
    return dialogue

async def pipeline():
    dataset = {
        "graph_type": [],
        "graph": [],
        "dialogue_str": [],
        "dialogue_list": []
    }

    raw_results = await generate_dialogue_graphs_from_templates(graph_templates, 5)
    for g_type in tqdm(raw_results):
        base_graph = raw_results[g_type]
        variants = await augment_data(base_graph, themes={"shop", "postal services", "cooking", "customer service", "daily life"}, amount=10)
        for i in variants:
            dataset['graph_type'].append(g_type)
            dataset['graph'] = i
            dataset['dialogue_str'], dataset['dialogue_list'] = dialogues_from_graph(i, include_readable=True)

    with open('dataset/dataset_v1.json', 'w') as f:
        f.write(json.dumps(dataset, indent=2, ensure_ascii=False))
    
    dataframe = pd.DataFrame(dataset)
    dataframe.to_parquet("dataset/dataset_v1.parquet")


if __name__=="__main__":
    asyncio.run(pipeline())