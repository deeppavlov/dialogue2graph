import logging
from pathlib import Path
from chatsky_llm_autoconfig.missing_edges_prompt import prompt_name
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry

# import chatsky_llm_autoconfig.algorithms.dialog_generation
# import chatsky_llm_autoconfig.algorithms.dialog_augmentation
# import chatsky_llm_autoconfig.algorithms.graph_generation
# import chatsky_llm_autoconfig.algorithms.single_graph_generation
# import chatsky_llm_autoconfig.algorithms.multiple_graph_generation
# import chatsky_llm_autoconfig.algorithms.two_stages_graph_generation
# import chatsky_llm_autoconfig.algorithms.three_stages_graph_generation
# import chatsky_llm_autoconfig.algorithms.three_stages_graph_generation_1dialog
# import chatsky_llm_autoconfig.algorithms.topic_graph_generation

# from chatsky_llm_autoconfig.algorithms.dialog_augmentation import DialogAugmentator
# from chatsky_llm_autoconfig.algorithms.topic_graph_generation import CycleGraphGenerator
# from chatsky_llm_autoconfig.algorithms.dialog_generation import DialogSampler
import json

# from datasets import load_dataset
from chatsky_llm_autoconfig.graph import Graph, BaseGraph
from chatsky_llm_autoconfig.dialog import Dialog
from chatsky_llm_autoconfig.metrics.automatic_metrics import (
    # all_paths_sampled,
    is_same_structure,
    compare_graphs,
)
from chatsky_llm_autoconfig.metrics.llm_metrics import (
    are_triplets_valid,
    is_theme_valid,
)
from chatsky_llm_autoconfig.utils import (
    save_json,
    read_json,
    # graph2comparable
)
from chatsky_llm_autoconfig.settings import EnvSettings
from chatsky_llm_autoconfig.metrics.automatic_metrics import *
import datetime
from colorama import Fore
from langchain_community.chat_models import ChatOpenAI

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)

print("modules loaded")
env_settings = EnvSettings()
print("settings")
model = ChatOpenAI(
    model="gpt-4o",
    api_key=env_settings.OPENAI_API_KEY,
    base_url=env_settings.OPENAI_BASE_URL,
    temperature=0,
)
print("model loaded")

generation_model = ChatOpenAI(
    model=env_settings.GENERATION_MODEL_NAME,
    api_key=env_settings.OPENAI_API_KEY,
    base_url=env_settings.OPENAI_BASE_URL,
    temperature=0,
)
validation_model = ChatOpenAI(
    model=env_settings.FORMATTER_MODEL_NAME,
    api_key=env_settings.OPENAI_API_KEY,
    base_url=env_settings.OPENAI_BASE_URL,
)
# generation_model = ChatOpenAI(model=env_settings.GENERATION_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=0.2)
# validation_model = ChatOpenAI(model=env_settings.FORMATTER_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=0)


# dialog_to_graph = read_json(env_settings.TEST_DATA_PATH)["graph_to_dialog"]
dialog_to_graph = read_json(env_settings.TEST_DATA_PATH)
graph_to_graph = read_json(env_settings.GRAPHS_TO_FIX)
print("json read")
# dialog_to_graph = [load_dataset(env_settings.TEST_DATASET, token=env_settings.HUGGINGFACE_TOKEN)['train'][4]]
# graph_to_dialog = test_data["graph_to_dialog"]
# dialog_to_graph = test_data["dialog_to_graph"]

# graph_to_graph = test_data["graph_to_graph"]
# dialog_to_dialog = test_data["dialog_to_dialog"]

# model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

with open(env_settings.TOPICS_DATA) as f:
    test_data = json.load(f)
    # graph_to_dialog = test_data["graph_to_dialog"]
    # graph_to_graph = test_data["graph_to_graph"]
    # dialog_to_dialog = test_data["dialog_to_dialog"]
    topic_to_graph = test_data


def run_all_algorithms():
    print("IN")
    # Get all registered classes
    algorithms = AlgorithmRegistry.get_all()
    print("Classes to test:", *algorithms.keys(), sep="\n")
    print("------------------\n")

    total_metrics = {}
    for class_ in algorithms:
        class_instance = algorithms[class_]["type"]()
        metrics = {}

        if (
            algorithms[class_]["input_type"] is BaseGraph
            and algorithms[class_]["output_type"] is BaseGraph
        ):
            tp = algorithms[class_]["type"]
            class_instance = tp(generation_model)
            result = []
            for case in graph_to_graph:
                test_graph = Graph(graph_dict=case["graph"])
                result.append(class_instance.invoke(test_graph).graph_dict)
                save_json(data=result, filename=env_settings.GRAPH_SAVED)

        elif (
            algorithms[class_]["input_type"] is str
            and algorithms[class_]["output_type"] is BaseGraph
        ):
            metrics = {"is_theme_valid": [], "are_triplets_valid": []}
            for case in topic_to_graph:
                test_topic = case["topic"]
                result = class_instance.invoke(test_topic)

                metrics["are_triplets_valid"].append(
                    are_triplets_valid(result, model)["value"]
                )
                metrics["is_theme_valid"].append(
                    is_theme_valid(result, model, topic=test_topic)["value"]
                )

            metrics["is_theme_valid_avg"] = sum(metrics["is_theme_valid"]) / len(
                metrics["is_theme_valid"]
            )
            metrics["are_triplets_valid"] = sum(metrics["are_triplets_valid"]) / len(
                metrics["are_triplets_valid"]
            )
            metrics["is_theme_valid_avg"] = sum(metrics["is_theme_valid"]) / len(
                metrics["is_theme_valid"]
            )

        elif (
            algorithms[class_]["input_type"] in [Dialog, list[Dialog]]
            and algorithms[class_]["output_type"] is BaseGraph
        ):
            # tp = algorithms[class_]["type"]
            # class_instance = tp(prompt_name="general_graph_generation_prompt")
            # class_instance = tp(prompt_name="specific_graph_generation_prompt")
            # class_instance = tp(prompt_name="fourth_graph_generation_prompt")
            # class_instance = tp(prompt_name="options_graph_generation_prompt")
            # class_instance = tp(prompt_name="list_graph_generation_prompt")
            metrics = {"triplet_match": [], "is_same_structure": [], "graph_match": []}
            saved_data = {}
            result_list = []
            test_list = []
            if algorithms[class_]["path_to_result"] is not None:
                if Path(algorithms[class_]["path_to_result"]).is_file():
                    saved_data = read_json(algorithms[class_]["path_to_result"])

            if (
                saved_data
                and env_settings.GENERATION_MODEL_NAME in saved_data
                and prompt_name in saved_data.get(env_settings.GENERATION_MODEL_NAME)
            ):
                result = saved_data.get(env_settings.GENERATION_MODEL_NAME).get(
                    prompt_name
                )
                if result:
                    test_list = [
                        {
                            next(iter(case)): [
                                Graph(graph_dict=r) for r in case[next(iter(case))]
                            ]
                        }
                        for case in result
                    ]

            # print("LIST: ", test_list)
            if not test_list:
                for case in dialog_to_graph:
                    case_list = []
                    cur_list = []
                    if algorithms[class_]["input_type"] is Dialog:
                        for test_dialog in [
                            Dialog.from_list(c["messages"]) for c in case["dialogs"]
                        ]:
                            result_graph = class_instance.invoke(test_dialog)
                            cur_list.append(result_graph)
                            case_list.append(result_graph.graph_dict)

                    else:
                        result_graph = class_instance.invoke(
                            [
                                Dialog.from_list(c["messages"])
                                for c in case["dialogs"]
                            ]
                        )
                        # result_graph = class_instance.invoke([Dialog.from_list(case["false_dialog"])])
                        cur_list.append(result_graph)
                        case_list.append(result_graph.graph_dict)

                    result_list.append({case["topic"]: case_list})
                    test_list.append({case["topic"]: cur_list})
                new_data = {
                    env_settings.GENERATION_MODEL_NAME: {prompt_name: result_list}
                }
                saved_data.update(new_data)
                save_json(data=saved_data, filename=env_settings.GENERATION_SAVE_PATH)
            save_metrics = []
            for case, dialogs in zip(dialog_to_graph, test_list):
                # test_graph = Graph(graph_dict=graph2comparable(case["graph"]))
                test_graph_orig = Graph(graph_dict=case["graph"])
                for result_graph in dialogs[case["topic"]]:
                    try:
                        # comp_graph=Graph(graph_dict=graph2comparable(result_graph.graph_dict))

                        # metrics["triplet_match"].append(triplet_match(test_graph, comp_graph))
                        # comp = is_same_structure(test_graph, comp_graph)
                        comp = is_same_structure(test_graph_orig, result_graph)
                        metrics["is_same_structure"].append(comp)
                        match = compare_graphs(
                            Graph(graph_dict=result_graph.graph_dict), test_graph_orig
                        )
                        metrics["graph_match"].append(match)
                        print("MATCH: ", case["topic"], metrics["graph_match"][-1])
                        save_metrics.append(
                            {
                                case["topic"]: {
                                    "graph": result_graph.graph_dict,
                                    "is_same_structure": comp,
                                    "graph_match": match,
                                }
                            }
                        )
                    except Exception as e:
                        print("Exception: ", e)
                        # metrics["triplet_match"].append(False)
                        metrics["is_same_structure"].append(False)
                        metrics["graph_match"].append(False)
                        save_metrics.append(
                            {
                                case["topic"]: {
                                    "graph": result_graph.graph_dict,
                                    "is_same_structure": False,
                                    "graph_match": False,
                                }
                            }
                        )

            # metrics["triplet_match"] = sum(metrics["triplet_match"]) / len(metrics["triplet_match"])
            metrics["graph_match"] = sum(metrics["graph_match"]) / len(
                metrics["graph_match"]
            )
            metrics["is_same_structure"] = sum(metrics["is_same_structure"]) / len(
                metrics["is_same_structure"]
            )
            save_json(data=save_metrics, filename=env_settings.METRICS_SAVE_PATH)

        elif (
            algorithms[class_]["input_type"] is str
            and algorithms[class_]["output_type"] is list
        ):
            tp = algorithms[class_]["type"]
            class_instance = tp(generation_model, validation_model)
            result = []
            for case in topic_to_graph:
                test_topic = case["topic"]
                result.extend(class_instance.invoke(test_topic))
                save_json(data=result, filename=env_settings.GRAPH_SAVED)

        # elif algorithms[class_]["input_type"] is BaseGraph and algorithms[class_]["output_type"] is BaseGraph:
        #     metrics = {"is_theme_valid": [], "are_triplets_valid": []}
        #     for case in graph_to_dialog:
        #         test_graph = Graph(graph_dict=case["graph"])
        #         result = class_instance.invoke(test_graph)

        #         metrics["are_triplets_valid"].append(are_triplets_valid(result, model, topic="")["value"])
        #         metrics["is_theme_valid"].append(is_theme_valid(result, model, topic="")["value"])

        #     metrics["are_triplets_valid"] = sum(metrics["are_triplets_valid"]) / len(metrics["are_triplets_valid"])
        #     metrics["is_theme_valid_avg"] = sum(metrics["is_theme_valid"]) / len(metrics["is_theme_valid"])
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
                "all_paths_sampled_diff": metrics.get("all_paths_sampled_avg", 0)
                - previous_metrics[algorithm].get("all_paths_sampled_avg", 0),
                "all_utterances_present_diff": metrics.get(
                    "all_utterances_present_avg", 0
                )
                - previous_metrics[algorithm].get("all_utterances_present_avg", 0),
                "all_roles_correct_diff": metrics.get("all_roles_correct_avg", 0)
                - previous_metrics[algorithm].get("all_roles_correct_avg", 0),
                "is_correct_length_diff": metrics.get("is_correct_lenght_avg", 0)
                - previous_metrics[algorithm].get("is_correct_lenght_avg", 0),
                "are_triplets_valid_diff": metrics.get("are_triplets_valid", 0)
                - previous_metrics[algorithm].get("are_triplets_valid", 0),
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
    with open(env_settings.RESULTS_PATH) as f:
        old_data = json.load(f)

    date = str(datetime.datetime.now())
    new_metrics = {date: run_all_algorithms()}
    old_data.update(new_metrics)
    compare_results(date, old_data)

    print("WRITING")
    save_json(data=old_data, filename=env_settings.RESULTS_PATH)
    # with open(env_settings.RESULTS_PATH, "w") as f:
    #     f.write(json.dumps(old_data, indent=2, ensure_ascii=False))
