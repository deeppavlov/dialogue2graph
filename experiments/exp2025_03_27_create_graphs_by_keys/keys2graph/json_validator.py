# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/json_validator.py

"""
Light-weight helpers for validating and parsing JSON strings as well as
confirming that an ``annotation`` object contains *all* required keys.
"""

import json
from typing import Dict, Any


def is_json_string_valid(json_string: str) -> bool:
    """Return *True* if *json_string* is syntactically valid JSON."""
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False


def parse_json_string(json_string: str) -> Any:
    """Parse *json_string* and return the resulting Python object."""
    return json.loads(json_string)

# --------------------------------------------------------------------------- #
# Annotation validation helpers
# --------------------------------------------------------------------------- #

def has_required_keys(annotation: Dict[str, Any]) -> bool:
    """Check that *annotation* contains **all** required keys."""
    required_keys = [
        "topic",
        "sub_topic",
        "bot_goal",
        "success_criteria",
        "context_info",
        "language",
        "formality_level",
        "emotional_tone",
        "lexical_diversity",
        "use_of_jargon",
        "max_dialog_depth",
        "max_branching_factor",
        "mandatory_nodes",
        "optional_nodes",
        "start_node",
        "user_intents",
        "intent_hierarchy",
        "user_utterances_examples",
        "required_slots",
        "bot_utterances_templates",
        "follow_up_questions",
        "closing_phrases",
        "fallback_strategy",
        "confirmation_needed",
        "max_dialog_length",
        "alternate_paths",
        "escalation_policy",
        "user_feedback_collection",
        "user_persona",
        "dynamic_content"
    ]
    for key in required_keys:
        if key not in annotation:
            return False
    return True


def check_annotation_structure(annotation_object: Dict[str, Any]) -> bool:
    """
    Validate top-level structure::

        {
          "annotation": { ...all required keys... }
        }
    """
    if "annotation" not in annotation_object:
        return False
    annotation = annotation_object["annotation"]
    if not isinstance(annotation, dict):
        return False
    return has_required_keys(annotation)
