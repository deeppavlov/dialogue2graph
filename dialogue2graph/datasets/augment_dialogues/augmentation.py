import logging
from typing import Union
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from dialogue2graph.pipelines.core.algorithms import DialogAugmentation
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.metrics.no_llm_metrics.metrics import is_correct_length, match_roles

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class AugmentedTurn(BaseModel):
    """Dialogue turn to augment"""
    participant: str
    text: list[str] = Field(
        ..., description="List of utterance variations for this turn"
    )


class DialogueSequence(BaseModel):
    """Result as dialogue sequence"""
    result: list[AugmentedTurn] = Field(..., description="Sequence of augmented turns")


class DialogueAugmenter(DialogAugmentation):
    """Class for dialogue augmentation.
    
    Augments dialogues while preserving structure and conversation flow by rephrasing original dialogue lines."""
    
    model_storage: ModelStorage = Field(..., description="Model storage instance")
    generation_llm: str = Field(..., description="Key for generation LLM in storage")
    formatting_llm: str = Field(..., description="Key for formatting LLM in storage")

    def invoke(
        self,
        dialogue: Dialogue,
        prompt: str,
        topic: str = "",
    ) -> Union[list[Dialogue], str]:
        """Augment dialogue while preserving conversation structure.
        
        Args:
            dialogue: Input Dialogue object to augment
            prompt: Required augmentation prompt template
            topic: Contextual topic for augmentation (default: empty)

        Returns:
            List of augmented Dialogue objects or error message
        """
        if prompt == "":
            return "Preprocessing failed: prompt should be a valid instruction for LLM"

        try:
            message_dicts = [msg.model_dump() for msg in dialogue.messages]
            if message_dicts == []:
                return "Preprocessing failed: no messages found in the dialogue"

            augmentation_prompt = PromptTemplate.from_template(prompt)
            parser = JsonOutputParser(pydantic_object=DialogueSequence)

            fixed_parser = OutputFixingParser.from_llm(
                parser=parser, llm=self._get_llm(self.formatting_llm)
            )

            chain = (
                augmentation_prompt | self._get_llm(self.generation_llm) | fixed_parser
            )

            for attempt in range(3):
                try:
                    result = chain.invoke({"topic": topic, "dialogue": message_dicts})
                    try:
                        augmented_dialogues = self._create_dialogues(result)
                        return augmented_dialogues
                    except Exception as e:
                        logging.error(f"Error creating dialogues: {str(e)}")
                        return f"Post-processing failed: {str(e)}"

                except ValidationError as ve:
                    logging.warning(f"Validation error attempt {attempt + 1}: {ve}")

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
        return self.invoke(*args, **kwargs)
    
    async def evaluate(
        self,
        dialogue: Dialogue,
        prompt: str,
        topic: str = ""
    ) -> dict:
        """Evaluate augmentation quality with dictionary report format."""
        result = self.invoke(dialogue, prompt, topic)

        if isinstance(result, str):
            return {"error": result}

        report = {}
        for i, augmented_dialogue in enumerate(result):
            try:
                report[f"augmented_dialogue_{i}"] = {
                    "match_roles": match_roles(dialogue, augmented_dialogue),
                    "correct_length": is_correct_length(dialogue, augmented_dialogue),
                }
            except Exception as e:
                logging.error(f"Error while calculating metrics: {str(e)}")
        return report

    def _get_llm(self, llm_key: str):
        """Get model from model storage safely"""
        if llm_key not in self.model_storage.storage:
            raise ValueError(f"LLM key '{llm_key}' not found in model storage")
        return self.model_storage.storage[llm_key].model
    
    def _combine_one_dialogue(self, augmentation_result: DialogueSequence, i: int) -> dict:
        """Combine new augmented dialogues from utterance variations"""
        new_augmented_dialogue = {}
        new_augmented_dialogue["messages"] = []
        roles_to_add = [turn.participant for turn in augmentation_result.result]
        utterances_to_add = [turn.text[i] for turn in augmentation_result.result]

        for role, uttr in zip(roles_to_add, utterances_to_add):
            dict_messages = {}
            dict_messages["participant"] = role
            dict_messages["text"] = uttr
            new_augmented_dialogue["messages"].append(dict_messages)

        return new_augmented_dialogue

    def _create_dialogues(self, result: dict) -> list[Dialogue]:        
        """Create a list of Dialogue objects"""
        try:
            augmentation_result = DialogueSequence(result=result)
        except Exception as e:
            logging.error(f"Wrong type of augmentation result: {str(e)}")
            return f"Creating a list of Dialogue objects failed: {str(e)}"

        utterances_lists = [turn.text for turn in augmentation_result.result]
        lens = [len(uttr_list) for uttr_list in utterances_lists]

        augmented_dialogues = []
        for i in range(min(lens)):
            new_augmented_dialogue = self._combine_one_dialogue(augmentation_result, i)
            augmented_dialogues.append(new_augmented_dialogue)

        return [
            Dialogue.from_list(new_augmented_dialogue["messages"])
            for new_augmented_dialogue in augmented_dialogues
        ]
