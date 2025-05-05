import logging
from typing import Union
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from dialog2graph.pipelines.core.algorithms import DialogAugmentation
from dialog2graph.pipelines.core.dialog import Dialog
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.metrics.no_llm_metrics.metrics import is_correct_length, match_roles

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class AugmentedTurn(BaseModel):
    """Dialog turn to augment"""

    participant: str
    text: list[str] = Field(
        ..., description="List of utterance variations for this turn"
    )


class DialogSequence(BaseModel):
    """Result as dialog sequence"""

    result: list[AugmentedTurn] = Field(..., description="Sequence of augmented turns")


class DialogAugmenter(DialogAugmentation):
    """Class for dialog augmentation.

    Augments dialogs while preserving structure and conversation flow by rephrasing original dialog lines."""

    model_storage: ModelStorage = Field(..., description="Model storage instance")
    generation_llm: str = Field(..., description="Key for generation LLM in storage")
    formatting_llm: str = Field(..., description="Key for formatting LLM in storage")

    def invoke(
        self,
        dialog: Dialog,
        prompt: str,
        topic: str = "",
    ) -> Union[list[Dialog], str]:
        """Augment dialog while preserving conversation structure.

        Args:
            dialog: Input Dialog object to augment
            prompt: Required augmentation prompt template
            topic: Contextual topic for augmentation (default: empty)

        Returns:
            List of augmented Dialog objects or error message
        """
        if prompt == "":
            return "Preprocessing failed: prompt should be a valid instruction for LLM"

        try:
            message_dicts = [msg.model_dump() for msg in dialog.messages]
            if message_dicts == []:
                return "Preprocessing failed: no messages found in the dialog"

            augmentation_prompt = PromptTemplate.from_template(prompt)
            parser = JsonOutputParser(pydantic_object=DialogSequence)

            fixed_parser = OutputFixingParser.from_llm(
                parser=parser, llm=self._get_llm(self.formatting_llm)
            )

            chain = (
                augmentation_prompt | self._get_llm(self.generation_llm) | fixed_parser
            )

            for attempt in range(3):
                try:
                    result = chain.invoke({"topic": topic, "dialog": message_dicts})
                    try:
                        augmented_dialogs = self._create_dialogs(result)
                        return augmented_dialogs
                    except Exception as e:
                        logging.error(f"Error creating dialogs: {str(e)}")
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

    async def evaluate(self, dialog: Dialog, prompt: str, topic: str = "") -> dict:
        """Evaluate augmentation quality with dictionary report format."""
        result = self.invoke(dialog, prompt, topic)

        if isinstance(result, str):
            return {"error": result}

        report = {}
        for i, augmented_dialog in enumerate(result):
            try:
                report[f"augmented_dialog_{i}"] = {
                    "match_roles": match_roles(dialog, augmented_dialog),
                    "correct_length": is_correct_length(dialog, augmented_dialog),
                }
            except Exception as e:
                logging.error(f"Error while calculating metrics: {str(e)}")
        return report

    def _get_llm(self, llm_key: str):
        """Get model from model storage safely"""
        if llm_key not in self.model_storage.storage:
            raise ValueError(f"LLM key '{llm_key}' not found in model storage")
        return self.model_storage.storage[llm_key].model

    def _combine_one_dialog(self, augmentation_result: DialogSequence, i: int) -> dict:
        """Combine new augmented dialogs from utterance variations"""
        new_augmented_dialog = {}
        new_augmented_dialog["messages"] = []
        roles_to_add = [turn.participant for turn in augmentation_result.result]
        utterances_to_add = [turn.text[i] for turn in augmentation_result.result]

        for role, uttr in zip(roles_to_add, utterances_to_add):
            dict_messages = {}
            dict_messages["participant"] = role
            dict_messages["text"] = uttr
            new_augmented_dialog["messages"].append(dict_messages)

        return new_augmented_dialog

    def _create_dialogs(self, result: dict) -> list[Dialog]:
        """Create a list of Dialog objects"""
        try:
            augmentation_result = DialogSequence(result=result)
        except Exception as e:
            logging.error(f"Wrong type of augmentation result: {str(e)}")
            return f"Creating a list of Dialog objects failed: {str(e)}"

        utterances_lists = [turn.text for turn in augmentation_result.result]
        lens = [len(uttr_list) for uttr_list in utterances_lists]

        augmented_dialogs = []
        for i in range(min(lens)):
            new_augmented_dialog = self._combine_one_dialog(augmentation_result, i)
            augmented_dialogs.append(new_augmented_dialog)

        return [
            Dialog.from_list(new_augmented_dialog["messages"])
            for new_augmented_dialog in augmented_dialogs
        ]
