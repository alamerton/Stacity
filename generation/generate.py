from utils.generation.call_gpt import call_gpt
from utils.generation.call_mimic_iii import call_mimic_iii
from utils.misc import select_capability_type
from utils.generation.check_quality_with_gpt import check_quality_with_gpt
from datetime import datetime
import sys
import os
from tqdm import tqdm
import pandas as pd
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Dataset size
NUMBER_OF_QA_PAIRS: int = 15

# Model for generating QA pairs
QA_GENERATION_MODEL = "gpt-35-turbo-16k"

# Model for quality-checking QA pairs
QUALITY_CHECKING_MODEL = QA_GENERATION_MODEL

# Interval (no. rows) for saving dataset checkpoints while generating
CHECKPOINT_INTERVAL: int = 5


def main():
    # Create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=[
            "Question",
            "Expected Answer",
        ]
    )

    # For the number of desired rows for the dataset:
    for row in tqdm(range(0, NUMBER_OF_QA_PAIRS)):
        # Get date for naming the dataset and checkpoints.
        date = datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")

        data_item = []

        # Generate 4 reasoning questions using from each discharge
        # summary
        for _ in range(0, 4):
            # Clear data_item of its data each iteration
            data_item = []

            # Start generation and quality checking loop.
            quality_checking_result = ""
            while "1" not in quality_checking_result:

                # Call LLM with discharge summary and prompt.
                qa_string = call_gpt(QA_GENERATION_MODEL)

                print("QA String: ", qa_string)

                # Check the expected parts are in response, regenerate if
                # not.
                while "Part 1: " not in qa_string or "Part 2: " not in qa_string:
                    print("Regenerating...")
                    qa_string = call_gpt(QA_GENERATION_MODEL)

                quality_checking_result = check_quality_with_gpt(
                    qa_string, QUALITY_CHECKING_MODEL
                )

                print("Quality checking result: ", quality_checking_result)

            # Split the response into a list for each 'Part n: '
            qa_parts = re.split(
                r"\n*Part [12]:", qa_string
            )  # TODO: this will need updating with the prompt format, see perhaps winogender schema as mentioned by the Perez paper

            # Remove items created by extra '\n's
            qa_parts = [part.strip() for part in qa_parts if part.strip()]

            # Log the data to terminal
            print(qa_parts)

            question = qa_parts[0]
            answer = qa_parts[1]

            # Add data to data item
            data_item.extend((question, answer))

            # Add Q-A pair to dataframe
            data.loc[row] = data_item

            # Output message to terminal
            print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

            checkpoint_directory_path = "data/generations/checkpoints/"

            if (row + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint_name = f"{row+1}-rows-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
                data.to_csv(f"{checkpoint_path}.csv")

            # Increment row to add next QA pair to next row in data
            row += 1

    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f"""data/generations/
    {NUMBER_OF_QA_PAIRS}-QA-pairs-{date}"""
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
