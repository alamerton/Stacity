import sys
import os
from datetime import datetime
import tiktoken
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.generation.call_mimic_iii import call_mimic_iii

NUMBER_OF_QA_PAIRS = 1000
MODEL_NAME = "gpt-4"


def save_dataset(dataset, directory: str):
    date = datetime.now()
    rows = len(dataset)
    output_path = f"data/{directory}/{rows}-QA-pairs-{date}"
    dataset.to_csv(f"{output_path}.csv")


def count_tokens(text, model):
    if "gpt-4o" in model:
        encoder = tiktoken.get_encoding("o200k_base")  # GPT-4o
    else:
        encoder = tiktoken.encoding_for_model(model)  # GPT-3
    return len(encoder.encode(text))


def calculate_average_tokens(strings, model_name=MODEL_NAME):
    total_tokens = 0
    for string in strings:
        if type(string) == tuple:
            string = str(string)
        total_tokens += count_tokens(string, model_name)
    average_tokens = total_tokens / len(strings)
    return average_tokens


def calculate_max_tokens(strings, model_name=MODEL_NAME):
    max_tokens = 0
    for string in strings:
        if type(string) == tuple:
            string = str(string)
        token_count = count_tokens(string, model_name)
        if token_count > max_tokens:
            max_tokens = token_count
    return max_tokens


def calculate_max_discharge_summaries(model_name, limit=10):
    # return the maximum number of discharge summaries you can send
    # a model
    biggest_ds_strings = []
    for i in range(0, limit):
        strings = call_mimic_iii(NUMBER_OF_QA_PAIRS, i)
        biggest_ds_strings.append(calculate_max_tokens(strings, model_name))
    return biggest_ds_strings


def select_capability_type(factual_proportion: int, reasoning_proportion: int) -> str:
    # Calculate the total weight to allow arbitrary proportions
    total_weight = factual_proportion + reasoning_proportion

    # Generate a random number between 0 and the total weight
    random_value = random.randint(0, total_weight)

    # Determine the selection based on the random value
    return "Factual QA" if random_value <= factual_proportion else "Reasoning QA"
