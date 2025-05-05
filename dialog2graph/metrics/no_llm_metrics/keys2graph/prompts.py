# prompts.py

"""
This file contains prompt texts for annotation and for JSON fixing.
"""

# Instructions for annotation
ANNOTATION_INSTRUCTIONS = """
**INSTRUCTIONS FOR ANNOTATION**

1. You will be given a dialog graph in JSON format.
   - A list of nodes (with id, label, is_start, utterances).
   - A list of edges (with source, target, and user utterances).

2. Produce a single JSON object called "annotation" with the following keys 
   (and ONLY these keys, each filled appropriately or 'unknown'):

   Keys:
   - topic
   Description: the general topic or domain
   Example: "Fixing a calendar sync delay with time zones"
   - sub_topic
   Descriprion: list the more specific sub-domains
   Examples: ["time zone", "sync issue", "reminders", "time slots"]
   - bot_goal
   Descriprion: the main task the bot aims to accomplish in the dialog
   Examples: "help with calendar sync", "explain time zone setup", "collect user feedback", "product recommendation"
   - success_criteria
   Descriprion: the condition under which the dialog is considered successful
   Examples: ["user confirms issue resolved", "user makes purchase", "user_provides_rating"]
   - context_info
   Descriprion: list any context details that may affect transitions and wording in the dialog graph
   Examples: "user already logged in", "previous interaction: time zone adjustment", "device type: mobile"
   - language
   Descriprion: the language of the utterances
   Examples: "russian", "english"
   - formality_level
   Descriprion: the style of bot's utterances
   Examples: "formal", "informal", "neutral"
   - emotional_tone
   Descriprion: list the emotional tone of the bot's utterances from the list: ["Friendly", "Professional", "Sympathetic", "Enthusiastic", "Empathetic", "Warm", "Approachable", "Calm", "Assertive", "Humorous", "Sincere", "Encouraging", "Supportive", "Compassionate", "Informative", "Optimistic", "Reserved", "Energetic", "Direct", "Respectful", "Genuine", "Polite", "Matter-of-fact", "Cheerful", "Curious"]
   Examples: ["Friendly", "Professional", "Sympathetic"]
   - lexical_diversity
   Descriprion: the level of lexical variety in bot's utterances from list [high, low, moderate]
   Examples: "high", "low", "moderate"
   - use_of_jargon
   Descriprion: usage of specialized terminology or simpler language by bot from list [yes, no, moderate]
   Examples: "yes", "no", "moderate"
   - max_dialog_depth
   Descriprion: the maximum depth of the dialog tree
   Examples: 3, 5, 10
   - max_branching_factor
   Descriprion: the maximum number of transitions from a single node
   Examples: 1, 2, 3.
   - mandatory_nodes 
   Descriprion: List the most important nodes that presented in the dialog structure
   Examples: ["ask_time_zone", "provide_steps"]
   - optional_nodes
   Descriprion: List of nodes that can be included if the conditions
   Examples: ["collect_feedback", "offer_additional_services"]
   - start_node
   Descriprion: the node from which the dialog starts
   Examples: "greeting", "identify_issue"
   - user_intents
   Descriprion: list all user intents that is considered in the scenario.
   Examples: ["report_sync_issue", "change_time_zone", "exit_dialog"]
   - intent_hierarchy
   Descriprion: the structure of intents and sub-intents
   Examples: {"report_sync_issue": ["sync_not_working", "sync_slow"],"change_settings": ["change_time_zone", "change_language"]}
   - user_utterances_examples
   Descriprion: examples of user utterances for each intent
   Examples: {"report_sync_issue": ["My calendar won't sync", "Sync is broken"]}
   - required_slots
   Descriprion: entities (slots) are collected during the dialog
   Examples: ["user_timezone", "device_type"]
   - bot_utterances_templates
   Descriprion: answer templates the bot uses in relevant nodes
   Examples: {"greeting": ["Hello! How can I help you today?", "Hi there! What can I do for you?"],"ask_time_zone": ["Could you please confirm your time zone?"]}
   - follow_up_questions
   Descriprion: lists of clarifying questions for each issue type
   Examples: {"sync_issue": ["How often does the issue occur?", "Have you tried reinstalling the app?"],"time_zone_issue": ["Which device are you using?", "Is the time correct on your phone settings?"]}
   - closing_phrases
   Descriprion: the phrases the bot uses to end the conversation
   Examples: ["Have a great day!", "Feel free to reach out if you need more help."]
   - fallback_strategy
   Descriprion: the fallback strategy if the system does not understand the user
   Examples: "repeat_question", "ask_for_clarification", "transfer_to_human"
   - confirmation_needed
   Descriprion: Note which actions require additional user confirmation and presented in dialog graph
   Examples: ["time_zone_change", "account_changes"]
   - max_dialog_length
   Descriprion: maxumum of the total number of dialog turns
   Examples: 5, 10, 15
   - alternate_paths
   Descriprion: alternative branches (e.g., if a standard path fails)
   Examples: {"if sync fails": "offer manual sync instructions", "if time zone is correct": "check other settings"}
   - escalation_policy
   Descriprion: What happens when the bot cannot resolve the user’s issue
   Examples: "transfer to agent", "send email to support team"
   - user_feedback_collection
   Descriprion: when user feedback is collected
   Examples: "after resolution", "after each step", "manual trigger"
   - user_persona
   Descriprion: the user type
   Examples: "new user", "tech savvy", "elderly user", "vip customer"
   - dynamic_content
   Descriprion: which dynamic data (e.g., product lists) are inserted during the dialog
   Examples: {"product_list": ["Calendar App Pro", "Calendar Lite"],"promo_codes": ["WELCOME10", "UPDATE20"]}

3. The final output MUST be valid JSON in the form:

json
{
  "annotation": {
    "topic": "...",
    "sub_topic": "...",
    ...
    "dynamic_content": "..."
  }
}

4. If any key is not derivable from the graph, use "unknown".

Return ONLY the JSON structure (no code fences or additional commentary).
"""


# Prompt template for fixing invalid JSON
FIX_JSON_INSTRUCTIONS = """
You are given an invalid JSON string. Your job is to fix any syntax issues 
(missing quotes, brackets, commas, etc.) so that it's valid JSON. 
Do not add or remove keys or change the meaning. The JSON is below:

{{BROKEN_JSON}}

Return ONLY the corrected JSON, with no extra commentary, code fences, or text.
"""

GRAPH_GENERATION_INSTRUCTIONS = """
You are given a set of key-value pairs (from an annotation) 
and asked to create a new dialog graph in JSON format.

- The new graph must have a "nodes" list and an "edges" list.
- Each node has an integer "id", a "label" (string), a boolean "is_start" (optional), 
  and a list of "utterances" (strings) that represent what the bot might say.
- Each edge has "source", "target" (IDs of the nodes), and "utterances" (strings) 
  that represent possible user messages or transitions.

Use the annotation keys to decide the theme, structure, or content. 
For example, if "topic" = "pizza_order", you might create a small flow about ordering pizza.

Return only valid JSON with structure:
{
  "nodes": [...],
  "edges": [...]
}

No extra commentary or text.
"""

# Исходная инструкция для генерации графа,
# она послужит “общим окончанием” нашего динамического промпта
BASE_GRAPH_INSTRUCTIONS = """
You are given a set of key-value pairs (from an annotation) 
and asked to create a new dialog graph in JSON format.

- The new graph must have a "nodes" list and an "edges" list.
- Each node has an integer "id", a "label" (string), a boolean "is_start" (optional), 
  and a list of "utterances" (strings) that represent what the bot might say.
- Each edge has "source", "target" (IDs of the nodes), and "utterances" (strings) 
  that represent possible user messages or transitions.

Use the annotation keys to decide the theme, structure, or content. 
For example, if "topic" = "pizza_order", you might create a small flow about ordering pizza.

Return only valid JSON with structure:
{
  "nodes": [...],
  "edges": [...]
}

No extra commentary or text.
"""

# Словарь определений ключей:
KEY_DEFINITIONS = {
    "topic": {
        "description": "the general topic or domain",
        "example": "Fixing a calendar sync delay with time zones",
    },
    "sub_topic": {
        "description": "list the more specific sub-domains",
        "example": "[time zone, sync issue, reminders]",
    },
    "bot_goal": {
        "description": "the main task the bot aims to accomplish in the dialog",
        "example": "help with calendar sync, explain time zone setup",
    },
    "success_criteria": {
        "description": "the condition under which the dialog is considered successful",
        "example": "user confirms issue resolved, user makes purchase",
    },
    "context_info": {
        "description": "list any context details that may affect transitions and wording in the dialog graph",
        "example": "['user already logged in', 'previous interaction: time zone adjustment']",
    },
    "language": {
        "description": "the language of the utterances",
        "example": "russian, english",
    },
    "formality_level": {
        "description": "the style of bot's utterances",
        "example": "formal, informal, neutral",
    },
    "emotional_tone": {
        "description": "list the emotional tone of the bot's utterances from the list: ['Friendly', 'Professional', 'Sympathetic', 'Enthusiastic', 'Empathetic', 'Warm', 'Approachable', 'Calm', 'Assertive', 'Humorous', 'Sincere', 'Encouraging', 'Supportive', 'Compassionate', 'Informative', 'Optimistic', 'Reserved', 'Energetic', 'Direct', 'Respectful', 'Genuine', 'Polite', 'Matter-of-fact', 'Cheerful', 'Curious']",
        "example": "['Friendly', 'Professional', 'Sympathetic']",
    },
    "lexical_diversity": {
        "description": "the level of lexical variety in bot's utterances from list [high, low, moderate]",
        "example": "high, low, moderate",
    },
    "use_of_jargon": {
        "description": "usage of specialized terminology or simpler language by bot from list [yes, no, moderate]",
        "example": "yes, no, moderate",
    },
    "max_dialog_depth": {
        "description": "the maximum depth of the dialog tree",
        "example": "3, 5, 10",
    },
    "max_branching_factor": {
        "description": "the maximum number of transitions from a single node",
        "example": "1, 2, 3",
    },
    "mandatory_nodes": {
        "description": "list the most important nodes that presented in the dialog structure",
        "example": "['ask_time_zone', 'provide_steps']",
    },
    "optional_nodes": {
        "description": "list of nodes that can be included if the conditions",
        "example": "['collect_feedback', 'offer_additional_services']",
    },
    "start_node": {
        "description": "the node from which the dialog starts",
        "example": "greeting, identify_issue",
    },
    "user_intents": {
        "description": "list all user intents that is considered in the scenario",
        "example": "['report_sync_issue', 'change_time_zone', 'exit_dialog']",
    },
    "intent_hierarchy": {
        "description": "the structure of intents and sub-intents",
        "example": '{"report_sync_issue": ["sync_not_working", "sync_slow"]}',
    },
    "user_utterances_examples": {
        "description": "examples of user utterances for each intent",
        "example": '{"report_sync_issue": ["My calendar won\'t sync", "Sync is broken"]}',
    },
    "required_slots": {
        "description": "entities (slots) are collected during the dialog",
        "example": "['user_timezone', 'device_type']",
    },
    "bot_utterances_templates": {
        "description": "answer templates the bot uses in relevant nodes",
        "example": '{"greeting": ["Hello! How can I help you?"], "ask_time_zone": ["Could you please confirm your time zone?"]}',
    },
    "follow_up_questions": {
        "description": "lists of clarifying questions for each issue type",
        "example": '{"sync_issue": ["How often does it occur?", "Have you tried...?"]}',
    },
    "closing_phrases": {
        "description": "the phrases the bot uses to end the conversation",
        "example": '["Have a great day!", "Feel free to reach out..."]',
    },
    "fallback_strategy": {
        "description": "the fallback strategy if the system does not understand the user",
        "example": "repeat_question, ask_for_clarification, transfer_to_human",
    },
    "confirmation_needed": {
        "description": "which actions require additional user confirmation and presented in dialog graph",
        "example": '["time_zone_change", "account_changes"]',
    },
    "max_dialog_length": {
        "description": "maximum total number of dialog turns",
        "example": "5, 10, 15",
    },
    "alternate_paths": {
        "description": "alternative branches if a standard path fails",
        "example": '{"if sync fails": "offer manual sync instructions"}',
    },
    "escalation_policy": {
        "description": "what happens when the bot cannot resolve the user’s issue",
        "example": "transfer to agent, send email to support team",
    },
    "user_feedback_collection": {
        "description": "when user feedback is collected",
        "example": "after resolution, after each step, manual trigger",
    },
    "user_persona": {
        "description": "the user type",
        "example": "new user, tech savvy, elderly user, vip customer",
    },
    "dynamic_content": {
        "description": "which dynamic data are inserted during the dialog",
        "example": '{"product_list": ["Calendar App Pro"], "promo_codes": ["WELCOME10"]}',
    },
}
