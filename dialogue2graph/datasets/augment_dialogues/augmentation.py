import logging
import pandas as pd
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from dialogue2graph.pipelines.core.algorithms import DialogAugmentation
from dialogue2graph.pipelines.core.dialogue import DialogueMessage, Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.metrics.no_llm_metrics.metrics import (
    is_correct_length_multi_utterance,
    match_roles_multi_utterance)


class DialogueSequence(BaseModel):
    result: List[DialogueMessage] = Field(description="Sequence of Dialogue Messages")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class DialogueAugmenter(DialogAugmentation):
    """Dialogue augmenter that rephrases the original dialogue lines
    based on the lines themselves and the topic of the original dialogue.
    """
    
    model_storage: ModelStorage = Field(..., description="Model storage instance")
    generation_llm: str = Field(..., description="Key for generation LLM in storage")
    formatting_llm: str = Field(..., description="Key for formatting LLM in storage")

    def __init__(
        self,
        model_storage: ModelStorage,
        generation_llm: str,
        formatting_llm: str,
    ):

        super().__init__(
            model_storage=model_storage,
            generation_llm=generation_llm,
            formatting_llm=formatting_llm
        )

    def invoke(
        self,
        dialogue: list = None,
        topic: str = None,
        prompt: str = None
    ) -> Union[DialogueSequence, str]:
        """Augments dialogue while preserving conversation structure.
        
        Args:
            dialogue: list of messages to augment, each message is a dictionary 
                with 'participant' and 'text' keys
            topic: topic of the original dialogue
            prompt: prompt template for the LLM chain
            
        Returns:
            augmented dialogue: list of messages, each message is a dictionary 
                with 'participant' and 'text' keys, value of the 'text' key is 
                a list of augmented utterance variations
        """
        try:                   
            augmentation_prompt = PromptTemplate.from_template(prompt)
            parser = JsonOutputParser(pydantic_object=DialogueSequence)
            
            fixed_parser = OutputFixingParser.from_llm(
                parser=parser,
                llm=self.model_storage.storage[self.formatting_llm].model
            )

            chain = (
                augmentation_prompt
                | self.model_storage.storage[self.generation_llm].model
                | fixed_parser
            )

            for attempt in range(3):
                try:
                    return chain.invoke({"topic": topic, "dialogue": dialogue})
                except Exception as e:
                    if attempt == 2:
                        return f"Generation failed after 3 attempts: {str(e)}"
                    
        except Exception as e:
            return f"Critical error: {str(e)}"
        
    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
    
    async def evaluate(self, dialogue, topic, prompt, report_type="dict"):
        augmented_dialogue = self.invoke(dialogue, topic, prompt)

        if isinstance(augmented_dialogue, str):
            return {
                "error": augmented_dialogue
                } if report_type == "dict" else pd.DataFrame([{"error": augmented_dialogue}])

        report = {
            "match_roles": match_roles_multi_utterance(dialogue, augmented_dialogue),
            "is_correct_length": is_correct_length_multi_utterance(dialogue, augmented_dialogue)
        }

        if report_type == "dataframe":
            report = pd.DataFrame(report, index=[0])
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")