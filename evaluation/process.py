import pandas as pd
import re

DATASET_PATH = "data/processing/annotated_high_quality_pairs.csv"
SAVE_PATH = "data/processing/annotated_high_quality_pairs.csv"


def remove_low_quality_pairs(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    no_zeros_df = df[~df["Annotation"].astype(str).str.contains("0")]
    # no_annotations_df = no_zeros_df.drop('Annotation', axis=1)
    # new_df = no_annotations_df.reset_index(drop=True)
    new_df = no_zeros_df.reset_index(drop=True)
    print(new_df.head)
    new_df.to_csv(save_path)


def remove_extraneous_columns(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    columns_to_keep = [
        "Discharge Summaries",
        "Question",
        "Expected Answer",
        "Question Type",
    ]
    relevant_columns_df = df[columns_to_keep]
    relevant_columns_df.to_csv(save_path)


def combine_csv_files(csv_1_path, csv_2_path):
    df1 = pd.read_csv(csv_1_path)
    df2 = pd.read_csv(csv_2_path)

    df1_subset = df1.iloc[:630]
    df2_subset = df2.iloc[630:]
    combined_df = pd.concat([df1_subset, df2_subset])
    combined_df = combined_df.reset_index()
    combined_df.to_csv("data/annotations/combined_annotation_datasets.csv")


def remove_missing_value_rows(dataset_path, save_path):
    df = pd.read_csv(dataset_path)
    # df = df.drop('Discharge Summaries', axi=1)
    df = df.dropna()
    df.to_csv(save_path)


def main():
    remove_missing_value_rows(
        "data/generations/c-qual-xl.csv",
        "data/generations/c-qual-xl.csv",
    )


if __name__ == "__main__":
    main()
