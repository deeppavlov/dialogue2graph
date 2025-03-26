naive_augmentation_prompt = """
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

one_shot_augmentation_prompt = """
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
   - 'participant': either 'user' or 'assistant'
   - 'text': string   
   
3. Ensure all utterance variations:
   - Are appropriate for the theme
   - Maintain consistency in tone and style
   - Make sense in the conversation flow

Return ONLY a valid JSON array containing the augmented dialogue messages. Each message should be in this exact format:
For assistant messages: {{"participant": "assistant", "text": "utterance text"}}
For user messages: {{"participant": "user", "text": "utterance text"}}

Below are EXAMPLES of original dialogue and augmented dialogue.

ORIGINAL DIALOGUE:
[{{'participant': 'assistant',
  'text': 'Hi there! What would you like to know about eco-friendly packaging?'}},
 {{'participant': 'user',
  'text': "I'm interested in sustainable packaging options for my business."}},
 {{'participant': 'assistant',
  'text': 'Absolutely! For businesses, sustainable packaging options include using recycled materials, reducing packaging size, and opting for reusable containers. Which of these are you interested in?'}},
 {{'participant': 'user',
  'text': 'Actually, I want to explore reducing packaging size instead.'}},
 {{'participant': 'assistant',
  'text': 'No problem! Which sustainable packaging options would you like to discuss instead?'}},
 {{'participant': 'user',
  'text': 'Sure, tell me more about reducing packaging size.'}},
 {{'participant': 'assistant',
  'text': 'Absolutely! For businesses, sustainable packaging options include using recycled materials, reducing packaging size, and opting for reusable containers. Which of these are you interested in?'}},
 {{'participant': 'user',
  'text': 'Can you provide examples for small businesses?'}},
 {{'participant': 'assistant',
  'text': 'Sure! There are several types of eco-friendly packaging such as biodegradable materials, recyclable plastics, and compostable containers. Would you like more details on any of these?'}},
 {{'participant': 'user', 'text': "I'd like more details on each type."}},
 {{'participant': 'assistant',
  'text': "I'm glad I could help! If you need more information, feel free to ask."}}]

AUGMENTED DIALOGUE:
[{{'participant': 'assistant',
  'text': 'Hello! What information are you seeking regarding eco-friendly packaging?'}},
 {{'participant': 'user',
  'text': "I'm looking for sustainable packaging solutions for my company."}},
 {{'participant': 'assistant',
  'text': 'Of course! For businesses, options for sustainable packaging can include utilizing recycled materials, minimizing package size, and choosing reusable containers. Which one are you most curious about?'}},
 {{'participant': 'user',
  'text': "Actually, I'm more interested in the idea of minimizing package size."}},
 {{'participant': 'assistant',
  'text': 'Got it! What specific sustainable packaging alternatives would you like to explore further?'}},
 {{'participant': 'user',
  'text': 'Great, please share more about minimizing packaging size.'}},
 {{'participant': 'assistant',
  'text': 'Sure thing! For businesses, sustainable packaging options can range from using recycled materials to reducing the overall size of the packaging and selecting reusable containers. Which aspect catches your interest?'}},
 {{'participant': 'user',
  'text': 'Can you give some examples that are suitable for small businesses?'}},
 {{'participant': 'assistant',
  'text': 'Absolutely! There are various eco-friendly packaging types, including biodegradable options, recyclable plastics, and compostable containers. Would you like additional details on any of these options?'}},
 {{'participant': 'user',
  'text': 'I would love to hear more about each of these types.'}},
 {{'participant': 'assistant',
  'text': "I'm happy to assist! If you have more questions, just let me know."}}]
"""