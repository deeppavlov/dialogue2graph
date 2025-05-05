from dialog2graph.pipelines.core.dialog import DialogMessage
from typing import List
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from tqdm import tqdm
import pickle
from augmentation_utils import (
    is_correct_length_modified,
    match_roles_modified,
)


class DialogSequence(BaseModel):
    result: List[DialogMessage]


def augment_dialog(dialog, topic, prompt, generation_model, temp=0.7):
    augmentation_prompt = PromptTemplate.from_template(prompt)
    model = ChatOpenAI(
        model=generation_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temp,
    )
    parser = JsonOutputParser(pydantic_object=DialogSequence)
    chain = augmentation_prompt | model | parser

    tries = 0
    while tries < 3:
        try:
            augmented_dialog = chain.invoke({"topic": topic, "dialog": dialog})
            length_comparison = is_correct_length_modified(dialog, augmented_dialog)
            roles_comparison = match_roles_modified(dialog, augmented_dialog)

            if length_comparison and roles_comparison:
                return augmented_dialog
            else:
                tries += 1
                e = f"length comparison: {length_comparison}; roles comparison: {roles_comparison}"

        except Exception:
            tries += 1

    return f"Generation error: {e}"


def augment_dialog_data(data, prompt, generation_model, path_to_save, temp=0.7):
    new_data = []

    for i, example in enumerate(data):
        print(f"Augmenting example {i}:")
        topic = example["topic"]
        all_dialogs = example["dialogs"]

        example["augmented_dialogs"] = []

        for element in tqdm(all_dialogs, total=len(all_dialogs)):
            orig_dialog = element["messages"]
            try:
                aug_dialog = augment_dialog(
                    orig_dialog, topic, prompt, generation_model, temp
                )
            except Exception as e:
                aug_dialog = e

            example["augmented_dialogs"].append(
                {"id": element["id"], "messages": aug_dialog}
            )
        new_data.append(example)
        with open(path_to_save, "wb") as fp:
            pickle.dump(new_data, fp)

    return new_data
