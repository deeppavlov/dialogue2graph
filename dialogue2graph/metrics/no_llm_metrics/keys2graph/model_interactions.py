# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/model_interactions.py

import os
import json
from typing import Dict, Any
from .prompts import FIX_JSON_INSTRUCTIONS, BASE_GRAPH_INSTRUCTIONS, KEY_DEFINITIONS

# Предположим, openai_client.py лежит рядом (в keys2graph)
# Если он в другом месте, нужно поправить путь:
from openai import OpenAI


def generate_annotation(
    dialog_graph: Dict[str, Any],
    prompt_instructions: str,
    model_name: str,
    temperature: float,
    supports_user_role: bool = True,
) -> str:
    """
    Generates annotation JSON from a dialogue graph by calling LLM.
    Uses environment for API key/base unless explicitly set.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=base_url)

    graph_str = json.dumps(dialog_graph, ensure_ascii=False, indent=2)
    if model_name != "o1-mini":
        if supports_user_role:
            messages = [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": f"DIALOG GRAPHS:\n{graph_str}"},
            ]
        else:
            combined_content = f"{prompt_instructions}\n\nDIALOG GRAPHS:\n{graph_str}"
            messages = [{"role": "user", "content": combined_content}]

        completion = client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature
        )
    else:
        combined_content = f"{prompt_instructions}\n\nDIALOG GRAPHS:\n{graph_str}"
        messages = [{"role": "user", "content": combined_content}]
        completion = client.chat.completions.create(model=model_name, messages=messages)

    return completion.choices[0].message.content


def fix_json_with_gpt35(
    broken_json: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0
) -> str:
    """
    Fix broken JSON using GPT-3.5 with a specialized prompt.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(broken_json)

    fix_prompt = FIX_JSON_INSTRUCTIONS.replace("{{BROKEN_JSON}}", broken_json)
    messages = [
        {"role": "system", "content": "You are a helpful JSON fixer assistant."},
        {"role": "user", "content": fix_prompt},
    ]
    completion = client.chat.completions.create(
        model=model_name, messages=messages, temperature=temperature
    )
    return completion.choices[0].message.content


def regenerate_annotation(
    dialog_graph: Dict[str, Any],
    prompt_instructions: str,
    model_name: str,
    temperature: float,
    api_key: str = "",
    base_url: str = "",
    supports_user_role: bool = True,
) -> str:
    """
    Attempt to regenerate a brand new annotation if repeated fixes fail.
    """
    return generate_annotation(
        dialog_graph=dialog_graph,
        prompt_instructions=prompt_instructions,
        model_name=model_name,
        temperature=temperature,
        supports_user_role=supports_user_role,
    )


def build_dynamic_graph_prompt(keys_dict: Dict[str, Any]) -> str:
    """
    Формирует динамический промпт для генерации диалогового графа
    на основе только тех ключей, которые переданы в keys_dict.
    """
    lines = []

    lines.append(
        "You are given a set of key-value pairs from an annotation. "
        "Use them to create a new dialogue graph in JSON format.\n\n"
        "Here are the definitions of the used keys from the annotation instructions:\n\n"
    )

    # Перечисляем определения только для тех ключей, которые есть в keys_dict
    for k in keys_dict.keys():
        if k in KEY_DEFINITIONS:
            definition_info = KEY_DEFINITIONS[k]
            desc = definition_info["description"]
            ex = definition_info["example"]
            lines.append(f"- {k}\n  Description: {desc}\n  Example: {ex}\n")

    lines.append("\nHere are the actual values for these keys:\n")
    # Выводим реальные значения ключей, которые пользователь передал
    lines.append(json.dumps(keys_dict, indent=2, ensure_ascii=False))

    lines.append("\n\n")
    # Добавляем завершающую часть инструкции (из BASE_GRAPH_INSTRUCTIONS)
    # lines.append(BASE_GRAPH_INSTRUCTIONS)

    # Склеиваем всё в одну строку
    final_prompt = "".join(lines)
    print(final_prompt)
    return final_prompt


def generate_dialog_graph_by_keys(
    keys_dict: Dict[str, Any],
    model_name: str,
    temperature: float,
    api_key: str = "",
    base_url: str = "",
) -> str:
    """
    Generate a new dialogue graph from a dict of keys using LLM.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_API_BASE"]

    client = OpenAI(api_key=api_key, base_url=base_url)

    keys_str = json.dumps(keys_dict, ensure_ascii=False, indent=2)

    # Генерируем динамический промпт
    dynamic_prompt = build_dynamic_graph_prompt(keys_dict)

    if model_name != "o1-mini":
        messages = [
            {"role": "system", "content": BASE_GRAPH_INSTRUCTIONS},
            {"role": "user", "content": dynamic_prompt},
        ]
        completion = client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature
        )
    else:
        combined_prompt = BASE_GRAPH_INSTRUCTIONS + "\n\n" + dynamic_prompt
        messages = [{"role": "user", "content": combined_prompt}]
        completion = client.chat.completions.create(model=model_name, messages=messages)

    print(messages)

    return completion.choices[0].message.content
