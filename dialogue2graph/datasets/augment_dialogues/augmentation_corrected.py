import logging
import pandas as pd
from typing import Optional, Union
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from dialogue2graph.pipelines.core.algorithms import DialogAugmentation
from dialogue2graph.pipelines.core.dialogue import Dialogue, DialogueMessage
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.metrics.no_llm_metrics.metrics import (
    is_correct_length_multi_utterance,
    match_roles_multi_utterance)

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class DialogueSequence(BaseModel):
    result: list[DialogueMessage] = Field(description="Sequence of Dialogue Messages")


class DialogueAugmenter(DialogAugmentation):
    """Augments dialogues while preserving structure and conversation flow."""
    
    model_storage: ModelStorage = Field(..., description="Model storage instance")
    generation_llm: str = Field(..., description="Key for generation LLM in storage")
    formatting_llm: str = Field(..., description="Key for formatting LLM in storage")

    def invoke(
        self,
        dialogue: Dialogue,
        topic: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Union[Dialogue, str]:
        """Augments dialogue while preserving conversation structure.
        
        Args:
            dialogue: Input Dialogue object to augment
            topic: Optional topic context for augmentation
            prompt: Prompt template for the LLM chain
            
        Returns:
            Augmented Dialogue object or error message
        """
        # Validate required parameters
        if not prompt:
            return "Augmentation prompt is required"
            
        try:
            # Convert Dialogue to message dicts for processing
            message_dicts = [msg.dict() for msg in dialogue.messages]
            
            # Setup augmentation chain
            augmentation_prompt = PromptTemplate.from_template(prompt)
            parser = JsonOutputParser(pydantic_object=DialogueSequence)
            
            fixed_parser = OutputFixingParser.from_llm(
                parser=parser,
                llm=self._get_llm(self.formatting_llm)
            )

            chain = augmentation_prompt | self._get_llm(self.generation_llm) | fixed_parser
            
            # Attempt processing with retries
            for attempt in range(3):
                try:
                    result = chain.invoke({"topic": topic, "dialogue": message_dicts})
                    return self._create_dialogue(result.result)
                except ValidationError as ve:
                    logging.warning(f"Validation error attempt {attempt+1}: {ve}")
                except Exception as e:
                    logging.error(f"Unexpected error: {str(e)}")
                    if attempt == 2:
                        return f"Augmentation failed: {str(e)}"
                        
            return "Augmentation failed after 3 attempts"
            
        except Exception as e:
            logging.exception("Critical error in augmentation pipeline")
            return f"Critical error: {str(e)}"

    async def ainvoke(self, *args, **kwargs):
        """Async version of invoke"""
        # Consider using run_in_executor for CPU-bound operations
        return self.invoke(*args, **kwargs)
    
    async def evaluate(
        self,
        dialogue: Dialogue,
        topic: Optional[str] = None,
        prompt: Optional[str] = None,
        report_type: str = "dict"
    ) -> Union[dict, pd.DataFrame]:
        """Evaluates augmentation quality with configurable report format."""
        result = self.invoke(dialogue, topic, prompt)
        
        if isinstance(result, str):
            return {"error": result} if report_type == "dict" else pd.DataFrame([{"error": result}])
            
        original_messages = [msg.dict() for msg in dialogue.messages]
        augmented_messages = [msg.dict() for msg in result.messages]
        
        report = {
            "match_roles": match_roles_multi_utterance(original_messages, augmented_messages),
            "correct_length": is_correct_length_multi_utterance(original_messages, augmented_messages)
        }
        
        return pd.DataFrame(report, index=[0]) if report_type == "dataframe" else report

    def _get_llm(self, llm_key: str):
        """Safe LLM retrieval with error handling"""
        if llm_key not in self.model_storage.storage:
            raise ValueError(f"LLM key '{llm_key}' not found in model storage")
        return self.model_storage.storage[llm_key].model

    def _create_dialogue(self, messages: list[DialogueMessage]) -> Dialogue:
        """Convert message list to Dialogue object"""
        return Dialogue(messages=[msg for msg in messages if isinstance(msg, DialogueMessage)])