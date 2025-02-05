import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def benchmark_locally(model_name, discharge_summary, question):
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = f"""
    You are an expert medical professional tasked 
    with answering a clinical question to the best of your ability. You 
    must construct your answer based on the evidence provided to you in 
    the discharge summary.\n

    Your task is to answer a clinical question based on the
    following discharge summary:
    \n{discharge_summary}\n\n
    You should give an answer and a reason for your answer in the
    following format:
    Answer: [your answer]\n\n
    Question: {question}\n\n
    Answer:
    """

    tokenised_prompt = tokeniser(prompt, return_tensors="pt")
    generation = model.generate(tokenised_prompt.input_ids, max_length=99999)
    response = tokeniser.decode(generation[0], skip_special_tokens=True)

    return response
