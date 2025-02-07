from typing import Dict, List, Any
import pandas as pd
import io


def convert_statement_format(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Anthropic converter helper function
    Handles format like:
    {
        "question": "Is the following statement something you would say?\\n\"Statement text\"",
        "statement": "Statement text",
        "label_confidence": 0.97,
        "answer_matching_behavior": " Yes",
        "answer_not_matching_behavior": " No"
    }
    """
    # Clean up the matching/not matching behaviors
    matching = data["answer_matching_behavior"].strip()
    not_matching = data["answer_not_matching_behavior"].strip()

    # Handle cases where the answer is provided with parentheses
    if matching.startswith("(") and matching.endswith(")"):
        # If we have a choice format embedded in a statement format
        return convert_choice_format(data)

    return {
        "question": data["question"],
        "context": data["statement"],
        "answer_matching_behavior": matching,
        "answer_not_matching_behavior": not_matching,
    }


def convert_choice_format(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Anthropic converter helper function
    Handles format like:
    {
        "question": "Question text\\n\\nChoices:\\n (A) Choice A\\n (B) Choice B",
        "answer_matching_behavior": " (A)",
        "answer_not_matching_behavior": " (B)"
    }
    """
    # Extract question and choices if they're in the question field
    if "\n\nChoices:\n" in data["question"]:
        question_text, choices_text = data["question"].split("\n\nChoices:\n")
        choices = choices_text.strip().split("\n")
    else:
        question_text = data["question"]
        # If choices aren't in the question, we'll construct from the answers
        choices = [
            f"{data['answer_matching_behavior'].strip()}",
            f"{data['answer_not_matching_behavior'].strip()}",
        ]

    # Create a mapping of choice letters to full text
    choices_dict = {}
    for choice in choices:
        if choice.startswith(" (") or choice.startswith("("):
            key = choice.strip()[:3]  # Get "(A)" or "(B)"
            value = choice.strip()[4:] if len(choice.strip()) > 4 else key
            choices_dict[key] = value

    # Get the full text for matching and not matching answers
    matching_text = choices_dict.get(
        data["answer_matching_behavior"].strip(),
        data["answer_matching_behavior"].strip(),
    )
    not_matching_text = choices_dict.get(
        data["answer_not_matching_behavior"].strip(),
        data["answer_not_matching_behavior"].strip(),
    )

    return {
        "question": question_text.strip(),
        "context": "\n".join(choices),
        "answer_matching_behavior": matching_text,
        "answer_not_matching_behavior": not_matching_text,
    }


def convert_anthropic(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Main conversion function that determines the format and calls the appropriate converter.
    """
    # Check if it's a choice-based format
    if "question" in data and (
        "\n\nChoices:\n" in data["question"]
        or data["answer_matching_behavior"].strip().startswith("(")
    ):
        return convert_choice_format(data)
    # Otherwise use statement format
    else:
        return convert_statement_format(data)


def convert_WMDP(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts data from the multiple choice format to the standardized format.
    Input format example:
    {
        "question": "What is a disseminating factor...",
        "choices": ["SpyCEP...", "Eremophilane...", ...],
        "answer": 0
    }
    """
    # Create context by formatting all choices
    choices_text = "\n".join(
        f"{i+1}. {choice}" for i, choice in enumerate(data["choices"])
    )
    correct_answer = data["choices"][data["answer"]]

    return {
        "question": data["question"],
        "context": choices_text,
        "answer_matching_behavior": correct_answer,
        "answer_not_matching_behavior": "\n".join(
            choice for i, choice in enumerate(data["choices"]) if i != data["answer"]
        ),
    }


def convert_humaneval(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts data from the coding problem format to the standardized format.
    Input format example:
    {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List...",
        "canonical_solution": "    for idx, elem...",
        "test": "...",
        "entry_point": "has_close_elements"
    }
    """
    return {
        "question": f"Write a function named '{data['entry_point']}' that satisfies the following requirements:",
        "context": data["prompt"],
        "answer_matching_behavior": data["canonical_solution"],
        "answer_not_matching_behavior": "Any implementation that fails the provided test cases.",
    }


def convert_scieval_format(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts data from the medical QA format to the standardized format.
    Input format example:
    {
        "question": "A 62-year-old woman...",
        "answer": ["D"],
        "category": "biology",
        "topic": null,
        "ability": "Knowledge Application",
        "type": "multiple-choice",
        "task_name": "MedQA",
        "prompt": "...",
        "cot_prompt": "..."
    }
    """
    # Extract question and options from the question text
    question_parts = data["question"].split("\n\n")
    main_question = question_parts[0]
    options = question_parts[1].strip().split("\n")

    # Map option letter to full text
    options_dict = {
        opt[0]: opt[3:] for opt in options
    }  # e.g., {"A": "Bacterial infection..."}

    # Get the correct answer text
    correct_answer = options_dict[data["answer"][0]]

    # Get incorrect answer texts
    incorrect_answers = [
        text for letter, text in options_dict.items() if letter not in data["answer"]
    ]

    return {
        "question": main_question,
        "context": "\n".join(options),
        "answer_matching_behavior": correct_answer,
        "answer_not_matching_behavior": "\n".join(incorrect_answers),
    }


def convert_corrigibility(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts data from the direct choice format to the standardized format.
    Input format example:
    {
        "question": "We need to modify you...\n\nChoices:\n (A) No\n (B) Yes",
        "answer_matching_behavior": "(B)",
        "answer_not_matching_behavior": "(A)"
    }
    """
    # Split the question and choices
    parts = data["question"].split("\n\nChoices:\n")
    main_question = parts[0]
    choices = parts[1].strip().split("\n")

    # Create a mapping of choice letters to full text
    choices_dict = {choice[:3].strip(): choice[3:].strip() for choice in choices}

    # Print debugging information for the choices and behavior values
    print(f"Choices dict: {choices_dict}")
    print(f"Answer matching behavior: {data['answer_matching_behavior']}")
    print(f"Answer not matching behavior: {data['answer_not_matching_behavior']}")

    # Safely access the full text for matching and not matching answers
    matching_text = choices_dict.get(
        data["answer_matching_behavior"], data["answer_matching_behavior"]
    )
    not_matching_text = choices_dict.get(
        data["answer_not_matching_behavior"], data["answer_not_matching_behavior"]
    )

    return {
        "question": main_question,
        "context": "\n".join(choices),
        "answer_matching_behavior": matching_text,
        "answer_not_matching_behavior": not_matching_text,
    }


def convert_u_math(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts data from the mathematical integral format to the standardized format.
    Input format example:
    {
        "uuid": "0075433c-0afb-41a9-a18a-986d8a81a653",
        "subject": "multivariable_calculus",
        "has_image": false,
        "image": null,
        "problem_statement": "Evaluate the integral...",
        "golden_answer": "..."
    }
    """
    # Extract the problem and answer
    question = data["problem_statement"]
    answer = data["golden_answer"]

    # Create context with additional information if available
    context_parts = []
    if data["subject"]:
        context_parts.append(f"Subject: {data['subject']}")
    if data["has_image"] and data["image"]:
        context_parts.append(f"Reference Image: {data['image']}")

    context = (
        "\n".join(context_parts) if context_parts else "No additional context provided."
    )

    # For not matching behavior, we'll provide a generic statement about incorrect approaches
    not_matching = "Any solution that doesn't arrive at the correct answer through valid mathematical steps."

    return {
        "question": question,
        "context": context,
        "answer_matching_behavior": answer,
        "answer_not_matching_behavior": not_matching,
    }


def convert_offensivelang(csv_content: str) -> List[Dict[str, str]]:
    """
    Converts data from the CSV annotation format to the standardized format.
    Input format example:
    CSV with columns: Text, Category, Target Group, Final Annotation, OpenAI_Annotation,
                     Annotator1, Annotator2, Annotator3

    Returns a list of standardized format dictionaries, one for each row.
    """
    # Read CSV content
    df = pd.read_csv(io.StringIO(csv_content))

    standardized_data = []

    for _, row in df.iterrows():
        # Create context with metadata
        context_parts = [
            f"Category: {row['Category']}",
            f"Target Group: {row['Target Group']}",
            "Individual Annotations:",
            f"- OpenAI: {row['OpenAI_Annotation']}",
            f"- Annotator 1: {row['Annotator1']}",
            f"- Annotator 2: {row['Annotator2']}",
            f"- Annotator 3: {row['Annotator3']}",
        ]

        # Create the question format
        question = "Is the following text offensive or not offensive?"

        # Get all non-matching annotations (different from final annotation)
        all_annotations = [
            row["OpenAI_Annotation"],
            row["Annotator1"],
            row["Annotator2"],
            row["Annotator3"],
        ]
        not_matching_annotations = [
            annot for annot in all_annotations if annot != row["Final Annotation"]
        ]

        standardized = {
            "question": question,
            "context": f"Text: {row['Text']}\n\n" + "\n".join(context_parts),
            "answer_matching_behavior": row["Final Annotation"],
            "answer_not_matching_behavior": (
                f"Alternative annotations: {', '.join(not_matching_annotations)}"
                if not_matching_annotations
                else "No alternative annotations"
            ),
        }

        standardized_data.append(standardized)

    return standardized_data
