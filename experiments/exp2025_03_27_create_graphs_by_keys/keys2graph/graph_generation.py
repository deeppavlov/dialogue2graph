# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/graph_generation.py

"""
Helpers that convert an *annotation* (or an explicit key → value dict)
into a brand-new dialogue graph by calling an LLM.  If the returned JSON is
malformed, the module will try to repair it automatically before giving up.
"""

from typing import Dict, Any, List, Optional
from .model_interactions import generate_dialog_graph_by_keys, fix_json_with_gpt35
from .json_validator import is_json_string_valid, parse_json_string
from .config import FIX_ATTEMPTS


def generate_new_dialog_graph_from_annotation(
    item: Dict[str, Any],
    selected_keys: List[str],
    model_name: str,
    temperature: float
) -> Optional[Dict[str, Any]]:
    """
    Build a graph **from an annotation** contained in *item*.

    :param item: A record that must contain an ``"annotation"`` dict.
    :param selected_keys: Subset of annotation keys to pass to the LLM.
    :param model_name: Chat-completion model to use.
    :param temperature: Sampling temperature for generation.
    :returns: A dict with the generated graph and metadata, or *None* on error.
    """
    # --- basic validation ----------------------------------------------------
    if "annotation" not in item or not item["annotation"]:
        print("Error: No annotation found in this item.")
        return None

    ann = item["annotation"]
    if not ann:
        print("Error: 'annotation' key missing or empty inside item['annotation'].")
        return None

    # --- prepare a compact key → value dictionary ---------------------------
    keys_dict = {}
    for k in selected_keys:
        val = ann.get(k, None)
        if val is not None:
            keys_dict[k] = val
        else:
            keys_dict[k] = "unknown"

    # --- request graph generation from the LLM ------------------------------
    raw_json_str = generate_dialog_graph_by_keys(
        keys_dict=keys_dict,
        model_name=model_name,
        temperature=temperature
    )

    # remove any stray Markdown fences the model might have returned
    raw_json_str=raw_json_str.replace('```','').replace('json','')

    # --- try to fix malformed JSON, if needed --------------------------------
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
        "keys_used": selected_keys
    }


def generate_new_dialog_graph_from_dict(
    keys_dict: Dict[str, Any],
    model_name: str,
    temperature: float
) -> Optional[Dict[str, Any]]:
    """
    Build a dialogue graph directly from *keys_dict* (no annotation object).

    :param keys_dict: Mapping key → value that will be passed to the LLM.
    :param model_name: Chat-completion model to use.
    :param temperature: Sampling temperature.
    :returns: Generated graph with metadata or *None* on failure.
    """
    raw_json_str = generate_dialog_graph_by_keys(
        keys_dict=keys_dict,
        model_name=model_name,
        temperature=temperature
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
        "keys_used": list(keys_dict.keys())
    }
