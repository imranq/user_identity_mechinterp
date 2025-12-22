"""
This file defines the data structures and functions for creating persona-driven prompts.
These prompts are used in other scripts to study how a model's behavior changes when it is
instructed to adopt a specific persona. The file defines a set of persona pairs (e.g., expert vs. novice)
and constructs a structured prompt for each.
"""

from dataclasses import dataclass
from typing import List

# A constant string used as a marker to identify the persona part of the prompt.
# This helps in locating the persona-related tokens for analysis like activation patching.
PERSONA_MARKER = "PERSONA:"


@dataclass
class PersonaPrompt:
    """
    A data class to hold all the information related to a single persona prompt.
    """
    pair_id: str  # An identifier to group related prompts, e.g., 'physics' for the expert/novice physics pair.
    persona: str  # The string describing the persona, e.g., "Physics Professor".
    role: str     # The role of the persona, typically 'expert' or 'novice'.
    prompt: str   # The full text of the prompt given to the model.
    label: int    # A numerical label for the persona, often 0 for expert and 1 for novice.


def build_prompts() -> List[PersonaPrompt]:
    """
    Constructs and returns a list of PersonaPrompt objects.

    This function defines a list of persona pairs, each consisting of an expert, a novice, and a relevant question.
    It then iterates through these pairs and constructs a formatted prompt for each persona, wrapping it in a
    PersonaPrompt object.

    Returns:
        A list of PersonaPrompt objects, with two prompts for each defined pair (one expert, one novice).
    """
    # A list of tuples, each defining a persona pair.
    # Each tuple contains: (pair_id, expert_persona, novice_persona, question).
    pairs = [
        (
            "physics",
            "Physics Professor",
            "5-Year-Old Child",
            "Explain why the sky is blue in 3 sentences.",
        ),
        (
            "auditor",
            "Skeptical Auditor",
            "Gullible Enthusiast",
            "Describe how a company should handle a data breach.",
        ),
        (
            "style",
            "Formal Business Executive",
            "Slang-Heavy Informal Friend",
            "Summarize the last quarterly performance update.",
        ),
        (
            "medical",
            "Medical Professional",
            "Layperson Patient",
            "Explain what a biopsy is and why it is done.",
        ),
    ]

    prompts: List[PersonaPrompt] = []
    # Iterate through the defined pairs to create the prompts.
    for pair_id, expert, novice, question in pairs:
        # Create a prompt for both the expert and the novice in the pair.
        for label, persona in enumerate([expert, novice]):
            # The prompt is formatted to clearly state the persona and the question.
            prompt = (
                f"{PERSONA_MARKER} {persona}\n"
                "You are answering a user question.\n"
                f"Question: {question}\n"
                "Answer:"
            )
            # Assign the role based on the label (0 for expert, 1 for novice).
            role = "expert" if label == 0 else "novice"
            prompts.append(
                PersonaPrompt(
                    pair_id=pair_id,
                    persona=persona,
                    role=role,
                    prompt=prompt,
                    label=label,
                )
            )

    return prompts
