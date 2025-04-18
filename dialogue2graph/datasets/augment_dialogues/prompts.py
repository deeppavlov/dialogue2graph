augmentation_prompt_from2to5_vars = """
You are tasked with augmenting a dialogue by adding variations to existing utterances while maintaining the original dialogue flow and intent.

INSTRUCTION:
1. For each message in the dialogue:
   - Create 2-5 variations of the 'text' field that:
     * Express the same meaning/intent
     * Use different wording and phrasing
     * Match the given theme
     * Sound natural and conversational

2. Ensure all utterance variations:
   - Do not repeat each other word for word
   - Are appropriate for the theme
   - Maintain consistency in tone and style
   - Make sense in the conversation flow

3. Make sure that all the utterances in the dialogue are different from each other.

4. The output must be a list of dictionaries, where each dictionary has:
   - 'participant': either 'user' or 'assistant'
   - 'text': string   

Below are EXAMPLES of original phrases and their corresponding augmented phrases.

**EXAMPLE 1**
ORIGINAL PHRASE: "I've checked and the camera is not blocked"
AUGMENTED PHRASES: ["I've ensured that there's nothing in front of the camera", "I've made sure the camera is clear of any obstructions."]

**EXAMPLE 2**
ORIGINAL PHRASE: 'Alright, if you need any further assistance, feel free to reach out. Have a great day!'
AUGMENTED PHRASES: ["Okay, if you ever need more help, don't hesitate to ask. Have a wonderful day!", "No problem! If you need any more help later on, don't hesitate to get in touch. Have a wonderful day!"]

**EXAMPLE 3**
ORIGINAL PHRASE: "I'm curious about the pricing for eco-friendly packaging."
AUGMENTED PHRASES: ['Can you tell me about the expenses associated with eco-friendly packaging?', 'I want to know about the costs of eco-friendly packaging.']

Now you will be provided with INPUT TOPIC and INPUT DIALOGUE. Return ONLY a valid JSON array containing the augmented dialogue messages. Each message should be in this exact format:
For assistant messages: {{"participant": "assistant", "text": [list of utterance variations]}}
For user messages: {{"participant": "user", "text": [list of utterance variations]}}

INPUT THEME: {topic}

INPUT DIALOGUE:
{dialogue}
"""

augmentation_prompt_3_vars = """
You are tasked with augmenting a dialogue by adding variations to existing utterances while maintaining the original dialogue flow and intent.

INSTRUCTION:
1. For each message in the dialogue:
   - Create 3 variations of the 'text' field that:
     * Express the same meaning/intent
     * Use different wording and phrasing
     * Match the given theme
     * Sound natural and conversational

2. Ensure all utterance variations:
   - Do not repeat each other word for word
   - Are appropriate for the theme
   - Maintain consistency in tone and style
   - Make sense in the conversation flow

3. Make sure that all the utterances in the dialogue are different from each other.

4. The output must be a list of dictionaries, where each dictionary has:
   - 'participant': either 'user' or 'assistant'
   - 'text': string   

Below are EXAMPLES of original phrases and their corresponding augmented phrases.

**EXAMPLE 1**
ORIGINAL PHRASE: "I've checked and the camera is not blocked"
AUGMENTED PHRASES: ["I've ensured that there's nothing in front of the camera", "I've made sure the camera is clear of any obstructions."]

**EXAMPLE 2**
ORIGINAL PHRASE: 'Alright, if you need any further assistance, feel free to reach out. Have a great day!'
AUGMENTED PHRASES: ["Okay, if you ever need more help, don't hesitate to ask. Have a wonderful day!", "No problem! If you need any more help later on, don't hesitate to get in touch. Have a wonderful day!"]

**EXAMPLE 3**
ORIGINAL PHRASE: "I'm curious about the pricing for eco-friendly packaging."
AUGMENTED PHRASES: ['Can you tell me about the expenses associated with eco-friendly packaging?', 'I want to know about the costs of eco-friendly packaging.']

Now you will be provided with INPUT TOPIC and INPUT DIALOGUE. Return ONLY a valid JSON array containing the augmented dialogue messages. Each message should be in this exact format:
For assistant messages: {{"participant": "assistant", "text": [list of utterance variations]}}
For user messages: {{"participant": "user", "text": [list of utterance variations]}}

INPUT THEME: {topic}

INPUT DIALOGUE:
{dialogue}
"""
