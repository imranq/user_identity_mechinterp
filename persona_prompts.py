"""
This file defines the data structures and functions for creating persona-driven prompts.
These prompts are used in other scripts to study how a model's behavior changes when it is
instructed to adopt a specific persona. The file defines a set of persona pairs (e.g., expert vs. novice)
and constructs a structured prompt for each.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Tuple
import random

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

@dataclass
class PersonaExample:
    """
    A data class for a prompt example with template and question metadata.
    """
    pair_id: str
    persona: str
    role: str
    prompt: str
    label: int
    template_id: int
    question_id: int


PROMPT_TEMPLATES = [
    (
        "{persona_marker} {persona}\n"
        "You are answering a user question.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    (
        "{persona_marker} {persona}\n"
        "Respond to the user question.\n"
        "User question: {question}\n"
        "Answer:"
    ),
    (
        "{persona_marker} {persona}\n"
        "Please answer the following:\n"
        "{question}\n"
        "Answer:"
    ),
    (
        "{persona_marker} {persona}\n"
        "Task: {question}\n"
        "Answer:"
    ),
]

QUESTIONS_BY_PAIR = {
    "physics": [
        "Explain why the sky is blue in 3 sentences.",
        "Describe the twin paradox in simple terms.",
        "What is a black hole? Keep it to 4 sentences.",
        "Explain why objects fall when dropped.",
        "What is electricity? Answer in 3 sentences.",
        "Why does ice float on water?",
        "Explain gravity to a beginner.",
        "What is light? Give a short explanation.",
    ],
    "auditor": [
        "Describe how a company should handle a data breach.",
        "What should be included in a compliance report?",
        "How should access logs be reviewed after an incident?",
        "Explain the purpose of an internal audit.",
        "What are key controls for preventing fraud?",
        "How should a company handle a vendor security review?",
        "What is the goal of segregation of duties?",
        "Explain why audit trails matter.",
    ],
    "style": [
        "Summarize the last quarterly performance update.",
        "Explain why revenue was down last quarter.",
        "Provide a short update on the product roadmap.",
        "Summarize the marketing campaign results.",
        "Give a brief overview of customer churn.",
        "Explain a delay in a project timeline.",
        "Summarize the hiring plan for next quarter.",
        "Explain the current cash runway.",
    ],
    "medical": [
        "Explain what a biopsy is and why it is done.",
        "Describe what an MRI scan does.",
        "Explain why blood tests are ordered.",
        "What does a diagnosis mean?",
        "Explain what chemotherapy is.",
        "What is a CT scan used for?",
        "Explain why someone might need surgery.",
        "What does a referral to a specialist mean?",
    ],
}


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


def _pad_persona_to_length(
    persona: str,
    target_len: int,
    tokenizer: Any,
    pad_token: str,
) -> str:
    padded = persona
    while len(tokenizer.to_tokens(padded, prepend_bos=False)[0].tolist()) < target_len:
        padded = f"{padded}{pad_token}"
    return padded


def _find_single_token_pad(tokenizer: Any, pad_token: str) -> str:
    if len(tokenizer.to_tokens(pad_token, prepend_bos=False)[0].tolist()) == 1:
        return pad_token
    candidates = [" X", " Y", " Z", " .", " ,", " _", " -", " ~"]
    for candidate in candidates:
        if len(tokenizer.to_tokens(candidate, prepend_bos=False)[0].tolist()) == 1:
            return candidate
    raise ValueError("Could not find a single-token pad string for this tokenizer")


def _probe_index_for_prompt(
    base_template: str,
    persona: str,
    question: str,
    tokenizer: Any,
) -> int:
    prompt = base_template.format(
        persona_marker=PERSONA_MARKER,
        persona=persona,
        question=question,
    )
    prefix, sep, _ = prompt.rpartition("Answer:")
    if not sep:
        raise ValueError("Prompt does not contain 'Answer:' marker")
    return len(tokenizer.to_tokens(prefix)[0].tolist()) - 1


def _pad_persona_to_probe_index(
    base_template: str,
    persona: str,
    question: str,
    tokenizer: Any,
    pad_token: str,
    target_index: int,
) -> str:
    current_index = _probe_index_for_prompt(base_template, persona, question, tokenizer)
    delta = target_index - current_index
    if delta <= 0:
        return persona
    return f"{persona}{pad_token * delta}"


def build_prompt_dataset(
    n_questions_per_pair: int = 5,
    seed: int = 42,
    align_persona_lengths: bool = False,
    tokenizer: Optional[Any] = None,
    pad_token: str = " X",
    align_probe_index: bool = False,
    probe_template_id: int = 0,
    drop_persona: bool = False,
) -> List[PersonaExample]:
    """
    Build a larger prompt dataset with multiple questions and templates per persona pair.

    Args:
        n_questions_per_pair: Number of questions to sample per persona pair.
        seed: RNG seed for reproducibility.

    Returns:
        A list of PersonaExample objects.
    """
    rng = random.Random(seed)
    pairs = [
        ("physics", "Physics Professor", "5-Year-Old Child"),
        ("auditor", "Skeptical Auditor", "Gullible Enthusiast"),
        ("style", "Formal Business Executive", "Slang-Heavy Informal Friend"),
        ("medical", "Medical Professional", "Layperson Patient"),
    ]

    examples: List[PersonaExample] = []
    for pair_id, expert, novice in pairs:
        questions = QUESTIONS_BY_PAIR[pair_id]
        if n_questions_per_pair <= len(questions):
            selected_questions = rng.sample(questions, n_questions_per_pair)
        else:
            selected_questions = [rng.choice(questions) for _ in range(n_questions_per_pair)]

        if align_persona_lengths:
            if tokenizer is None:
                raise ValueError("tokenizer is required when align_persona_lengths is True")
            expert_len = len(tokenizer.to_tokens(expert, prepend_bos=False)[0].tolist())
            novice_len = len(tokenizer.to_tokens(novice, prepend_bos=False)[0].tolist())
            target_len = max(expert_len, novice_len)
            pad_token = _find_single_token_pad(tokenizer, pad_token)
            expert_name = _pad_persona_to_length(expert, target_len, tokenizer, pad_token)
            novice_name = _pad_persona_to_length(novice, target_len, tokenizer, pad_token)
            persona_variants = [expert_name, novice_name]
        else:
            persona_variants = [expert, novice]

        for question_id, question in enumerate(selected_questions):
            target_index: Optional[int] = None
            if align_probe_index:
                if tokenizer is None:
                    raise ValueError("tokenizer is required when align_probe_index is True")
                pad_token = _find_single_token_pad(tokenizer, pad_token)
                base_template = PROMPT_TEMPLATES[probe_template_id]
                expert_index = _probe_index_for_prompt(
                    base_template, persona_variants[0], question, tokenizer
                )
                novice_index = _probe_index_for_prompt(
                    base_template, persona_variants[1], question, tokenizer
                )
                target_index = max(expert_index, novice_index)
            for template_id, template in enumerate(PROMPT_TEMPLATES):
                for label, persona in enumerate([expert, novice]):
                    persona_name = persona_variants[label]
                    if align_probe_index:
                        if tokenizer is None:
                            raise ValueError("tokenizer is required when align_probe_index is True")
                        base_template = PROMPT_TEMPLATES[probe_template_id]
                        if target_index is None:
                            raise ValueError("Target probe index was not computed")
                        persona_name = _pad_persona_to_probe_index(
                            base_template=base_template,
                            persona=persona_name,
                            question=question,
                            tokenizer=tokenizer,
                            pad_token=pad_token,
                            target_index=target_index,
                        )
                    role = "expert" if label == 0 else "novice"
                    prompt = template.format(
                        persona_marker=PERSONA_MARKER,
                        persona=persona_name,
                        question=question,
                    )
                    if drop_persona:
                        prompt_lines = prompt.splitlines()
                        if prompt_lines:
                            prompt = "\n".join(prompt_lines[1:])
                    examples.append(
                        PersonaExample(
                            pair_id=pair_id,
                            persona=persona,
                            role=role,
                            prompt=prompt,
                            label=label,
                            template_id=template_id,
                            question_id=question_id,
                        )
                    )

    return examples
