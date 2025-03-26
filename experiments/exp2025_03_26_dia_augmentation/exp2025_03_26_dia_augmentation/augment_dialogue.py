from dialogue2graph.pipelines.core.dialogue import DialogueMessage, Dialogue
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from tqdm import tqdm
import pickle
import numpy as np
import os

# os.environ['PATH_TO_ENV'] = "~/projects/chatsky-llm-autoconfig/.env"

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
            return augmented_dialogue        
        except Exception as e:
            tries += 1

    return f'Generation error: {e}'