# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/graph_generation.py

from typing import Dict, Any, List, Optional
from .model_interactions import generate_dialog_graph_by_keys, fix_json_with_gpt35
from .json_validator import is_json_string_valid, parse_json_string
from .config import FIX_ATTEMPTS


def generate_new_dialog_graph_from_annotation(
    item: Dict[str, Any], selected_keys: List[str], model_name: str, temperature: float
) -> Optional[Dict[str, Any]]:
    if "annotation" not in item or not item["annotation"]:
        print("Error: No annotation found in this item.")
        return None

    ann = item["annotation"]
    if not ann:
        print("Error: 'annotation' key missing or empty inside item['annotation'].")
        return None

    keys_dict = {}
    for k in selected_keys:
        val = ann.get(k, None)
        if val is not None:
            keys_dict[k] = val
        else:
            keys_dict[k] = "unknown"

    raw_json_str = generate_dialog_graph_by_keys(
        keys_dict=keys_dict, model_name=model_name, temperature=temperature
    )

    raw_json_str = raw_json_str.replace("```", "").replace("json", "")

    if not is_json_string_valid(raw_json_str):
        for _ in range(FIX_ATTEMPTS):
            raw_json_str = fix_json_with_gpt35(raw_json_str, model_name="gpt-4o-mini")
            if is_json_string_valid(raw_json_str):
                break

    if not is_json_string_valid(raw_json_str):
        return None

    parsed_graph = parse_json_string(raw_json_str)
    if not isinstance(parsed_graph, dict):
        return None

    return {
        "graph": parsed_graph,
        "model": model_name,
        "temperature": temperature,
        "keys_used": selected_keys,
    }


def generate_new_dialog_graph_from_dict(
    keys_dict: Dict[str, Any], model_name: str, temperature: float
) -> Optional[Dict[str, Any]]:
    raw_json_str = generate_dialog_graph_by_keys(
        keys_dict=keys_dict, model_name=model_name, temperature=temperature
    )

    if not is_json_string_valid(raw_json_str):
        for _ in range(FIX_ATTEMPTS):
            raw_json_str = fix_json_with_gpt35(raw_json_str, model_name="gpt-4o-mini")
            if is_json_string_valid(raw_json_str):
                break

    if not is_json_string_valid(raw_json_str):
        return None

    parsed_graph = parse_json_string(raw_json_str)
    if not isinstance(parsed_graph, dict):
        return None

    return {
        "graph": parsed_graph,
        "model": model_name,
        "temperature": temperature,
        "keys_used": list(keys_dict.keys()),
    }
