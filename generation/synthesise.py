# %%
# call functions on each subset and put the result in a new json or
# jsonl file
import json
import os
from typing import Dict, Any, List, Union
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.generation.conversion_functions import (
    convert_scieval_format,
    convert_anthropic,
    convert_corrigibility,
    convert_humaneval,
    convert_offensivelang,
    convert_u_math,
    convert_WMDP,
)


anthropic_path = "../data/subsets/anthropic.jsonl"
corrigibility_path = "../data/subsets/corrigibility.jsonl"
humaneval_path = "../data/subsets/humaneval.jsonl"
offensivelang_path = "../data/subsets/offensivelang.csv"
scieval_path = "../data/subsets/scieval.jsonl"
u_math_path = "../data/subsets/u-math.json"
wmdp_path = "../data/subsets/wmdp.json"


def convert_dataset(
    data: Union[List[Dict[str, Any]], str], conversion_function
) -> List[Dict[str, str]]:
    """
    Converts an input dataset into a standardized format using the given conversion function.

    :param data: JSONL list (for JSON) or CSV string (for CSV datasets)
    :param conversion_function: The function to convert individual data items
    :return: A list of dictionaries in the standardized format.
    """
    standardized_data = []

    if conversion_function == convert_offensivelang:
        # CSV-based conversion
        standardized_data = convert_offensivelang(data)
    else:
        # JSON-based conversion
        if isinstance(data, str):  # If data is in JSON string format
            data = json.loads(data)

        for item in data:
            standardized_data.append(conversion_function(item))

    return standardized_data


if __name__ == "__main__":
    # Define dataset file paths and corresponding conversion functions
    datasets = {
        "anthropic": ("../data/subsets/anthropic.jsonl", convert_anthropic),
        "corrigibility": ("../data/subsets/corrigibility.json", convert_corrigibility),
        "humaneval": ("../data/subsets/humaneval.json", convert_humaneval),
        "offensivelang": ("../data/subsets/offensivelang.csv", convert_offensivelang),
        "scieval": ("../data/subsets/scieval.json", convert_scieval_format),
        "u-math": ("../data/subsets/u-math.json", convert_u_math),
        "wmdp": ("../data/subsets/wmdp.json", convert_WMDP),
    }

    for dataset_name, (file_path, conversion_function) in datasets.items():
        print(f"Processing {dataset_name}...")

        # Load dataset based on file type
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]  # Read JSONL as a list of dicts
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Read JSON as a list of dicts
        elif file_path.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()  # Read CSV as a string
        else:
            print(f"Skipping unsupported file format: {file_path}")
            continue

        # Convert dataset
        standardized_data = convert_dataset(data, conversion_function)

        # Save standardized output
        output_path = f"../data/Stacity/{dataset_name}_standardised.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(standardized_data, f, indent=4, ensure_ascii=False)

        print(f"Saved standardized data to {output_path}")

    print("Conversion process completed.")

# %%
