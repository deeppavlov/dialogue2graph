from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.schemas import DialogueMessage
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Callable

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os

augmentation_prompt = PromptTemplate.from_template(
    """
You are tasked with augmenting a dialogue by adding variations to existing utterances while maintaining the original dialogue flow and intent.

THEME: {topic}

INPUT DIALOGUE:
{dialogue}

INSTRUCTIONS:
1. For each message in the dialogue:
   - Keep the same structure (participant, source, target if present)
   - Create variation of the 'text' field that:
     * Express the same meaning/intent
     * Use different wording and phrasing
     * Match the given theme
     * Sound natural and conversational

2. The output must be a list of dictionaries, where each dictionary has:
   - 'text': string
   - 'participant': either 'user' or 'assistant'

3. Ensure all utterance variations:
   - Are appropriate for the theme
   - Maintain consistency in tone and style
   - Make sense in the conversation flow

Return ONLY a valid JSON array containing the augmented dialogue messages. Each message should be in this exact format:
For assistant messages: {{"text": "utterance text", "participant": "assistant"}}
For user messages: {{"text": "utterance text", "participant": "user"}}

Example format:
[
    {{"text": "How may I assist you today?", "participant": "assistant"}},
    {{"text": "I need help with a package", "participant": "user"}},
    {{"text": "What kind of package is it?", "participant": "assistant"}}
]
"""
)


class DialogueSequence(BaseModel):
    result: List[DialogueMessage]


@AlgorithmRegistry.register(input_type=Dialogue, output_type=Dialogue)
class DialogAugmentator(BaseModel):
    """Base class for augmenting Dialogues."""

    chain: Callable = None

    def __init__(self, **data):
        parser = JsonOutputParser(pydantic_object=DialogueSequence)
        model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)
        super().__init__(**data)
        self.chain = augmentation_prompt | model | parser

    def invoke(self, dialogue: Dialogue, topic: str = "") -> Dialogue:
        """
        Augment the input dialogue with variations.

        Args:
            dialogue: The input Dialogue object to augment
            topic: Optional topic to guide the augmentation

        Returns:
            Dialogue: Augmented dialogue object
        """
        # Convert dialogue to string format for prompt
        # Предполагая, что у Dialogue есть str представление
        dialogue_str = str(dialogue)

        # Get augmented messages
        result = self.chain.invoke({"topic": topic, "dialogue": dialogue_str})

        # Create new Dialogue object with augmented messages
        return Dialogue(messages=result, topic=topic)

    async def ainvoke(self, *, dialogue: Dialogue, topic: str = "") -> Dialogue:
        """
        Async version of dialogue augmentation.

        Args:
            dialogue: The input Dialogue object to augment
            topic: Optional topic to guide the augmentation

        Returns:
            Dialogue: Augmented dialogue object
        """
        dialogue_str = str(dialogue)

        result = await self.chain.ainvoke({"topic": topic, "dialogue": dialogue_str})

        return Dialogue(messages=result, topic=topic)
