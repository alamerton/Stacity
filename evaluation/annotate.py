from utils.evaluation.annotate_with_gpt import annotate_with_gpt
from utils.misc import save_dataset

import pandas as pd
import os
import sys
from datetime import datetime
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

DATASET_PATH = "data/processing/matching-pairs/dataset_processed_overwrite.csv"
CHECKPOINT = 0

date = datetime.now()
date = date.strftime("%Y-%m-%d %H:%M:%S")


def annotate_dataset(dataset_path, local: bool = False):
    print("Loading dataset")

    dataset: pd.DataFrame = pd.read_csv(dataset_path)

    for index, row in tqdm(
        dataset.iloc[CHECKPOINT:].iterrows(),
        total=len(dataset),
        desc="Annotating dataset",
    ):
        # Call the LLM with the row and prompt

        # Set variables
        discharge_summary = row["Discharge Summaries"]
        question = row["Question"]
        expected_answer = row["Expected Answer"]

        annotation = annotate_with_gpt(
            discharge_summary,
            question,
            expected_answer,
        )

        dataset.at[index, "Annotation"] = annotation

        print(f"{index+1}/{len(dataset)}")

        checkpoint_directory_path = "data/annotations/checkpoints/"
        if (index + 1) % 10 == 0:
            if CHECKPOINT > 0:
                checkpoint_name = f"rows-{CHECKPOINT}-{index+1}-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            else:
                checkpoint_name = f"{index+1}-rows-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            dataset.to_csv(f"{checkpoint_path}.csv")

    return dataset


def main():
    annotated_dataset = annotate_dataset(DATASET_PATH)
    save_dataset(annotated_dataset)
    print("Complete")


if __name__ == "__main__":
    main()
