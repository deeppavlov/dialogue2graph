from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.algorithms.dialogue_augmentation import DialogAugmentator
from chatsky_llm_autoconfig.algorithms.topic_graph_generation import CycleGraphGenerator
from chatsky_llm_autoconfig.algorithms.dialogue_generation import DialogueSampler
import json
from chatsky_llm_autoconfig.graph import Graph, BaseGraph
from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.metrics.automatic_metrics import *
from chatsky_llm_autoconfig.metrics.llm_metrics import are_triplets_valid, is_theme_valid
import datetime
from colorama import Fore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

with open("dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/autometrics/test_data/data.json") as f:
    test_data = json.load(f)
    graph_to_dialogue = test_data["graph_to_dialogue"]
    # graph_to_graph = test_data["graph_to_graph"]
    dialogue_to_dialogue = test_data["dialogue_to_dialogue"]
    topic_to_graph = test_data["topic_to_graph"]


def run_all_algorithms():
    # Get all registered classes
    algorithms = AlgorithmRegistry.get_all()
    print("Classes to test:", *algorithms.keys(), sep="\n")
    print("------------------\n")
    total_metrics = {}
    for class_ in algorithms:
        class_instance = algorithms[class_]["type"]()
        metrics = {}

        if algorithms[class_]["input_type"] is BaseGraph and algorithms[class_]["output_type"] is Dialogue:
            metrics = {"all_paths_sampled": [], "all_utterances_present": []}
            for case in graph_to_dialogue:
                test_dialogue = Dialogue(dialogue=case["dialogue"])
                test_graph = Graph(graph_dict=case["graph"])
                result = class_instance.invoke(test_graph)

                # metrics["all_paths_sampled"].append(all_paths_sampled(test_graph, result[0]))
                metrics["all_utterances_present"].append(all_utterances_present(test_graph, result))

            # metrics["all_paths_sampled_avg"] = sum(metrics["all_paths_sampled"]) / len(metrics["all_paths_sampled"])
            metrics["all_utterances_present_avg"] = sum(metrics["all_utterances_present"]) / len(metrics["all_utterances_present"])

        elif algorithms[class_]["input_type"] is Dialogue and algorithms[class_]["output_type"] is Dialogue:
            metrics = {
                "all_roles_correct": [],
                "is_correct_lenght": [],
            }
            for case in dialogue_to_dialogue:
                test_dialogue = Dialogue(dialogue=case["dialogue"])
                result = class_instance.invoke(dialogue=test_dialogue)

                metrics["all_roles_correct"].append(all_roles_correct(test_dialogue, result))
                metrics["is_correct_lenght"].append(is_correct_lenght(test_dialogue, result))

            metrics["all_roles_correct_avg"] = sum(metrics["all_roles_correct"]) / len(metrics["all_roles_correct"])
            metrics["is_correct_lenght_avg"] = sum(metrics["is_correct_lenght"]) / len(metrics["is_correct_lenght"])

        elif algorithms[class_]["input_type"] is str and algorithms[class_]["output_type"] is BaseGraph:
            metrics = {"is_theme_valid": [], "are_triplets_valid": []}
            for case in topic_to_graph:
                test_topic = case["topic"]
                result = class_instance.invoke(test_topic)

                metrics["are_triplets_valid"].append(are_triplets_valid(result, model)["value"])
                metrics["is_theme_valid"].append(is_theme_valid(result, model, topic=test_topic)["value"])

            metrics["is_theme_valid_avg"] = sum(metrics["is_theme_valid"]) / len(metrics["is_theme_valid"])
            metrics["are_triplets_valid"] = sum(metrics["are_triplets_valid"]) / len(metrics["are_triplets_valid"])

        elif algorithms[class_]["input_type"] is BaseGraph and algorithms[class_]["output_type"] is BaseGraph:
            metrics = {"is_theme_valid": [], "are_triplets_valid": []}
            for case in graph_to_dialogue:
                test_graph = Graph(graph_dict=case["graph"])
                result = class_instance.invoke(test_graph)

                metrics["are_triplets_valid"].append(are_triplets_valid(result, model, topic="")["value"])
                metrics["is_theme_valid"].append(is_theme_valid(result, model, topic="")["value"])

            metrics["are_triplets_valid"] = sum(metrics["are_triplets_valid"]) / len(metrics["are_triplets_valid"])
            metrics["is_theme_valid_avg"] = sum(metrics["is_theme_valid"]) / len(metrics["is_theme_valid"])
        total_metrics[class_] = metrics

    return total_metrics


def compare_results(date, old_data, compare_to: str = ""):

    current_metrics = old_data.get(date, {})
    if compare_to == "":
        previous_date = list(old_data.keys())[-2]
    else:
        previous_date = compare_to
    previous_metrics = old_data.get(previous_date, {})

    differences = {}
    for algorithm, metrics in current_metrics.items():
        if algorithm in previous_metrics:
            differences[algorithm] = {
                "all_paths_sampled_diff": metrics.get("all_paths_sampled_avg", 0) - previous_metrics[algorithm].get("all_paths_sampled_avg", 0),
                "all_utterances_present_diff": metrics.get("all_utterances_present_avg", 0)
                - previous_metrics[algorithm].get("all_utterances_present_avg", 0),
                "all_roles_correct_diff": metrics.get("all_roles_correct_avg", 0) - previous_metrics[algorithm].get("all_roles_correct_avg", 0),
                "is_correct_length_diff": metrics.get("is_correct_lenght_avg", 0) - previous_metrics[algorithm].get("is_correct_lenght_avg", 0),
                "are_triplets_valid_diff": metrics.get("are_triplets_valid", 0) - previous_metrics[algorithm].get("are_triplets_valid", 0),
                "total_diff": (
                    metrics.get("all_paths_sampled_avg", 0)
                    + metrics.get("all_utterances_present_avg", 0)
                    + metrics.get("all_roles_correct_avg", 0)
                    + metrics.get("is_correct_lenght_avg", 0)
                    + metrics.get("are_triplets_valid", 0)
                )
                - (
                    previous_metrics[algorithm].get("all_paths_sampled_avg", 0)
                    + previous_metrics[algorithm].get("all_utterances_present_avg", 0)
                    + previous_metrics[algorithm].get("all_roles_correct_avg", 0)
                    + previous_metrics[algorithm].get("is_correct_lenght_avg", 0)
                    + previous_metrics[algorithm].get("are_triplets_valid", 0)
                ),
            }
    for algorithm, diff in differences.items():
        print(f"Algorithm: {algorithm}")
        for metric, value in diff.items():
            if value < 0:
                print(f"{metric}: {Fore.RED}{value}{Fore.RESET}")
            elif value > 0:
                print(f"{metric}: {Fore.GREEN}{value}{Fore.RESET}")
            else:
                print(f"{metric}: {Fore.YELLOW}{value}{Fore.RESET}")
    return differences


if __name__ == "__main__":
    with open("dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/autometrics/results/results.json") as f:
        old_data = json.load(f)

    date = str(datetime.datetime.now())
    new_metrics = {date: run_all_algorithms()}
    old_data.update(new_metrics)
    compare_results(date, old_data)

    with open("dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/autometrics/results/results.json", "w") as f:
        f.write(json.dumps(old_data, indent=2, ensure_ascii=False))
