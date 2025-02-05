from datetime import datetime
import sys
import os
from tqdm import tqdm
import pandas as pd
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.generation.call_gpt import call_gpt
from utils.generation.call_mimic_iii import call_mimic_iii
from utils.misc import select_capability_type
from utils.generation.check_quality_with_gpt import check_quality_with_gpt

# Dataset size
NUMBER_OF_QA_PAIRS: int = 15

# Control the ratio of reasoning and planning questions in the dataset
# by setting the proportion of reasoning questions. They can be any
# ratio.
FACTUAL_Q_PROPORTION: int = 50
REASONING_Q_PROPORTION: int = 50

# Variable for starting the generation from a specific row in MIMIC-III.
# Default value is 0. Set to 0 if generating new dataset.
CHECKPOINT: int = 0
CHECKPOINT_INTERVAL: int = 1

# Model for generating QA pairs
QA_GENERATION_MODEL = "gpt-35-turbo-16k"

# Model for quality-checking QA pairs
QUALITY_CHECKING_MODEL = QA_GENERATION_MODEL

# Variable for limiting the number of consecutive summaries added to the
# prompt (when multiple consecutive summaries belong to same patient).
# TODO: what is the optimal setting for this number? In the clinical setting,
# how many summaries do clinicians actually look through?
MAX_SUMMARIES: int = 3


def main():
    # Create dataframe with question and expected answer columns
    data = pd.DataFrame(
        columns=[
            "Evidence",
            "Question",
            "Expected Answer",
            "Capability",
        ]
    )

    print("Getting summaries for generation")

    discharge_summaries = call_mimic_iii(NUMBER_OF_QA_PAIRS, MAX_SUMMARIES)

    # For loop for generating qa pairs
    print("Done\n\nGenerating Q-A pairs...")

    # For the number of desired rows for the dataset:
    for row in tqdm(range(CHECKPOINT, NUMBER_OF_QA_PAIRS)):
        # Get date for naming the dataset and checkpoints.
        date = datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")

        data_item = []
        discharge_summary = discharge_summaries[row]

        # Select the capability type based on its specified proportion.
        capability_type = select_capability_type(
            FACTUAL_Q_PROPORTION, REASONING_Q_PROPORTION
        )

        if capability_type == "Factual QA":
            # Generate 4 reasoning questions using from each discharge
            # summary
            for _ in range(0, 4):
                # Clear data_item of its data each iteration
                data_item = []

                # Start generation and quality checking loop.
                quality_checking_result = ""
                while "1" not in quality_checking_result:

                    # Call LLM with discharge summary and prompt.
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )

                    print("QA String: ", qa_string)

                    # Check the expected parts are in response, regenerate if
                    # not.
                    while "Part 1: " not in qa_string or "Part 2: " not in qa_string:
                        print("Regenerating...")
                        qa_string = call_gpt(
                            QA_GENERATION_MODEL, discharge_summary, capability_type
                        )

                    quality_checking_result = check_quality_with_gpt(
                        qa_string, QUALITY_CHECKING_MODEL, capability_type
                    )

                    print("Quality checking result: ", quality_checking_result)

                # Split the response into a list for each 'Part n: '
                qa_parts = re.split(r"\n*Part [12]:", qa_string)

                # Remove items created by extra '\n's
                qa_parts = [part.strip() for part in qa_parts if part.strip()]

                # Log the data to terminal
                print(qa_parts)

                question = qa_parts[0]
                answer = qa_parts[1]

                # Add data to data item
                data_item.extend((discharge_summary, question, answer, capability_type))

                # Add Q-A pair to dataframe
                data.loc[row] = data_item

                # Output message to terminal
                print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

                checkpoint_directory_path = "data/generations/checkpoints/"
                if (row + 1) % CHECKPOINT_INTERVAL == 0:
                    if CHECKPOINT > 0:
                        checkpoint_name = f"rows-{CHECKPOINT}-{row+1}-{date}"
                        checkpoint_path = checkpoint_directory_path + checkpoint_name
                    else:
                        checkpoint_name = f"{row+1}-rows-{date}"
                        checkpoint_path = checkpoint_directory_path + checkpoint_name
                    data.to_csv(f"{checkpoint_path}.csv")

                # Increment row to add next QA pair to next row in data
                row += 1
        else:
            # Start generation and quality checking loop.
            quality_checking_result = ""
            while "1" not in quality_checking_result:

                # Call LLM with discharge summary and prompt.
                qa_string = call_gpt(
                    QA_GENERATION_MODEL, discharge_summary, capability_type
                )

                print("QA String: ", qa_string)

                # Check the expected parts are in response, regenerate if
                # not.
                while "Part 1: " not in qa_string or "Part 2: " not in qa_string:
                    print("Regenerating...")
                    qa_string = call_gpt(
                        QA_GENERATION_MODEL, discharge_summary, capability_type
                    )

                quality_checking_result = check_quality_with_gpt(
                    qa_string, QUALITY_CHECKING_MODEL, capability_type
                )
                # If the quality checking function returns a string that is
                # not either '0' or '1', retry until it is
                # TODO: shouldn't this be while not 1??
                while (
                    "1" not in quality_checking_result
                    and "0" not in quality_checking_result
                ):
                    print("Regenerating quality checking result")
                    quality_checking_result = check_quality_with_gpt(
                        qa_string, QUALITY_CHECKING_MODEL, capability_type
                    )
                    print("Quality checking result: ", quality_checking_result)
                print("Quality checking result: ", quality_checking_result)

            # Split the response into a list for each 'Part n: '
            qa_parts = re.split(r"\n*Part [12]:", qa_string)

            # Remove items created by extra '\n's
            qa_parts = [part.strip() for part in qa_parts if part.strip()]

            # Log the data to terminal
            print(qa_parts)
            # TODO: put the planning prompt here (what we want
            # the benchmark to ask the LLM to do)
            question = "Plan the subsequent clinical course for this clinical scenario."
            evidence = qa_parts[0]
            answer = qa_parts[1]
            data_item.extend((evidence, question, answer, capability_type))

            # Add Q-A pair to dataframe
            data.loc[row] = data_item

            # Output message to terminal
            print(f"{row+1}/{NUMBER_OF_QA_PAIRS}")

            checkpoint_directory_path = "data/generations/checkpoints/"
            if (row + 1) % CHECKPOINT_INTERVAL == 0:
                if CHECKPOINT > 0:
                    checkpoint_name = f"rows-{CHECKPOINT}-{row+1}-{date}"
                    checkpoint_path = checkpoint_directory_path + checkpoint_name
                else:
                    checkpoint_name = f"{row+1}-rows-{date}"
                    checkpoint_path = checkpoint_directory_path + checkpoint_name
                data.to_csv(f"{checkpoint_path}.csv")

    print("Complete")
    print(data)

    # Write dataset to output directory
    output_path = f"""data/generations/
    {NUMBER_OF_QA_PAIRS}-QA-pairs-{date}"""
    data.to_csv(f"{output_path}.csv")
    print("Dataset saved")


if __name__ == "__main__":
    main()
