import pandas as pd
from tqdm import tqdm
import sys
import os
import numpy as np
from rouge import Rouge
import numpy as np
from nltk.util import ngrams
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from datetime import datetime
import spacy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from utils.evaluation.benchmark_with_azure import benchmark_with_azure
from utils.evaluation.benchmark_locally import benchmark_locally
from utils.misc import save_dataset

DATASET_PATH = "data/processing/cqual-small.csv"
MODEL_NAME = "Llama-2-70b-uvwrs"
LOCAL = False
CHECKPOINT = 0


date = datetime.now()
date = date.strftime("%Y-%m-%d %H:%M:%S")

nlp = spacy.load("en_core_web_md")
rouge = Rouge()
g_eval_metric = GEval(
    name="g-eval",
    criteria="""
    Evaluate the quality of the generated text based on the ability of the LLM
    to summarize, identify, and arrange text, ancondad answer specific questions 
    related to:
        1. Patient’s medical history
        2. Diagnoses made
        3. Procedures that were done
        4. Outcomes of procedures
        5. Changes in medication
        6. Complications
        7. Abnormalities
        8. Tests the patient has undergone
    """,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)


def record_model_answers(dataset_path, model_name):
    print("Loading dataset")
    dataset = pd.read_csv(dataset_path)

    for index, row in tqdm(
        dataset.iterrows(), total=len(dataset), desc=f"Benchmarking model"
    ):
        discharge_summary = row["Discharge Summaries"]
        question = row["Question"]

        if LOCAL:
            response = benchmark_locally(model_name, discharge_summary, question)
        else:
            response = benchmark_with_azure(model_name, discharge_summary, question)

        if "/" in model_name:
            model_name.replace("/", "_")

        dataset.at[index, f"{model_name} Response"] = response

        checkpoint_directory_path = "data/model-answers/checkpoints/"

        if (index + 1) % 10 == 0:
            if CHECKPOINT > 0:
                checkpoint_name = f"rows-{CHECKPOINT}-{index+1}-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            else:
                checkpoint_name = f"{model_name}-{index+1}-rows-{date}"
                checkpoint_path = checkpoint_directory_path + checkpoint_name
            dataset.to_csv(f"{checkpoint_path}.csv")

    return dataset


def get_exact_match(expected_answer: str, model_answer: str):
    return expected_answer == model_answer


def get_f1_score(expected_answer: str, model_answer: str):
    expected_answer_tokens = expected_answer.lower().split()
    model_answer_tokens = model_answer.lower().split()

    all_tokens = list(set(expected_answer_tokens + model_answer_tokens))
    gt_vector = [1 if token in expected_answer_tokens else 0 for token in all_tokens]
    model_vector = [1 if token in model_answer_tokens else 0 for token in all_tokens]

    # Calculate precision, recall, and F1 score
    precision = np.sum(np.array(gt_vector) * np.array(model_vector)) / np.sum(
        model_vector
    )
    recall = np.sum(np.array(gt_vector) * np.array(model_vector)) / np.sum(gt_vector)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return f1


def get_sas(expected_answer: str, model_answer: str):
    doc1 = nlp(expected_answer)
    doc2 = nlp(model_answer)
    similarity = doc1.similarity(doc2)
    return similarity


def get_rouge(expected_answer: str, model_answer: str):
    scores = rouge.get_scores(expected_answer, model_answer)[0]
    return scores["rouge-1"]["f"]


def get_n_grams(expected_answer: str, model_answer: str, n: int):
    expected_n_grams = set(ngrams(expected_answer.split(), n))
    model_n_grams = set(ngrams(model_answer.split(), n))
    return len(model_n_grams.intersection(expected_n_grams))


def get_precision(expected_answer: str, model_answer: str, n: int):
    n_grams = get_n_grams(model_answer, expected_answer, n)
    total = len(list(ngrams(model_answer.split(), n)))
    if total == 0:
        return 0.0
    return n_grams / total


def get_bleu(reference: str, candidate: str):
    precisions = []
    if not isinstance(reference, list):
        reference = [reference]

    for n in range(1, 4):
        precision = sum(get_precision(candidate, ans, n) for ans in reference) / len(
            reference
        )
        precisions.append(precision)

    if precisions:
        score = np.prod(np.power(precisions, 1.0 / len(precisions)))
    else:
        score = 0
    return score


def get_g_eval(expected_answer: str, model_answer: str):
    # test_case = LLMTestCase(input=model_answer, actual_output=expected_answer)
    test_case = LLMTestCase(input="Test input", actual_output="Expected output")
    print(g_eval_metric.measure(test_case))  # Check if this returns a valid score
    # print(f"Debug: test_case = {test_case}")
    score = g_eval_metric.measure(test_case)
    # print(f"Debug: score = {score}")
    return score


def score_model(dataset, model_name):

    if "/" in model_name:
        model_name.replace("/", "_")

    em_scores = []
    f1_scores = []
    sas_scores = []
    rouge_scores = []
    bleu_scores = []
    # clinical_concept_extraction_scores = []
    # medical_relation_extraction_scores = []
    # g_eval_scores = []

    for row in tqdm(range(0, len(dataset))):
        expected_answer = dataset["Expected Answer"][row]
        model_answer = dataset[f"{model_name} Response"][row]

        em_scores.append(get_exact_match(expected_answer, model_answer))
        f1_scores.append(get_f1_score(expected_answer, model_answer))
        sas_scores.append(get_sas(expected_answer, model_answer))
        rouge_scores.append(get_rouge(expected_answer, model_answer))
        bleu_scores.append(get_bleu(expected_answer, model_answer))

    exact_match = np.mean(em_scores)
    f1 = np.mean(f1_scores)
    semantic_answer_similarity = np.mean(sas_scores)
    rouge = np.mean(rouge_scores)
    bleu = np.mean(bleu_scores)
    # clinical_concept_extraction = np.mean(clinical_concept_extraction_scores)
    # medical_relation_extraction = np.mean(medical_relation_extraction_scores)
    # print("g_eval_scores: ", g_eval_scores)

    benchmarking_results = pd.DataFrame(
        {
            "Metric": [
                "Exact Match",
                "F1 Score",
                "Semantic Answer Similarity",
                "Rouge",
                "BLEU",
                # "Clinical Concept Extraction",
                # "Medical Relation Extraction",
                # "G-Eval",
            ],
            f"{MODEL_NAME} Score": [
                exact_match,
                f1,
                semantic_answer_similarity,
                rouge,
                bleu,
                # clinical_concept_extraction,
                # medical_relation_extraction,
                # g_eval,
            ],
        }
    )

    return benchmarking_results


def main():
    model_answers = record_model_answers(DATASET_PATH, MODEL_NAME)
    save_dataset(model_answers, directory="model-answers")
    model_answers = pd.read_csv("data/model-answers/Mistral-large.csv")
    benchmarking_results = score_model(model_answers, MODEL_NAME)
    save_dataset(benchmarking_results, directory="benchmarking-results")
    print(benchmarking_results)


if __name__ == "__main__":
    main()
