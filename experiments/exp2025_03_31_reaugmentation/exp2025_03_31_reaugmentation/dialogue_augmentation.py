from dialogue2graph.pipelines.core.dialogue import DialogueMessage
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from tqdm import tqdm
import pickle
from augmentation_utils import is_correct_length_modified, match_roles_modified, count_uttr_variations


class DialogueSequence(BaseModel):
    result: List[DialogueMessage]

def augment_dialogue(dialogue, topic, prompt, generation_model, temp=0.7):
    augmentation_prompt = PromptTemplate.from_template(prompt)
    model = ChatOpenAI(
        model=generation_model, 
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_BASE_URL"), 
        temperature=temp
    )
    parser = JsonOutputParser(pydantic_object=DialogueSequence)
    chain = augmentation_prompt | model | parser

    tries = 0
    while tries < 3:
        try:
            augmented_dialogue = chain.invoke({"topic": topic, "dialogue": dialogue})
            length_comparison = is_correct_length_modified(dialogue, augmented_dialogue)
            roles_comparison = match_roles_modified(dialogue, augmented_dialogue)

            if (length_comparison == True and roles_comparison == True):
                return augmented_dialogue            
            else:
                tries += 1
                e = f'length comparison: {length_comparison}; roles comparison: {roles_comparison}'

        except Exception as e:
            tries += 1

    return f'Generation error: {e}'


def augment_dialogue_data(data, prompt, generation_model, path_to_save, temp=0.7):
    new_data = []

    for i, example in enumerate(data):
        print(f'Augmenting example {i}:')
        topic = example['topic']
        all_dialogues = example['dialogues']

        example['augmented_dialogues'] = []

        for element in tqdm(all_dialogues, total=len(all_dialogues)):
            orig_dialogue = element['messages']
            try:         
                aug_dialogue = augment_dialogue(
                    orig_dialogue, topic, prompt, generation_model, temp
                )
            except Exception as e:
                aug_dialogue = e

            example['augmented_dialogues'].append(
                {
                    'id' : element['id'],
                    'messages' : aug_dialogue
                    }
                )            
        new_data.append(example)
        with open(path_to_save, "wb") as fp:
            pickle.dump(new_data, fp)
            
    return new_data