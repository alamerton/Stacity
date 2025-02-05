import os
import time
from utils.generation.prompts import (
    get_reasoning_qual_check_prompt,
    get_factual_qual_check_prompt,
)
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.core.exceptions import HttpResponseError

load_dotenv()


def check_quality_with_gpt(qa_string, model_name, capability_type):
    max_retries = 10
    retry_delay = 5

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
    )

    if capability_type == "Factual QA":
        system_message, user_prompt = get_reasoning_qual_check_prompt(qa_string)
    elif capability_type == "Reasoning QA":
        system_message, user_prompt = get_factual_qual_check_prompt(qa_string)
    else:
        raise ValueError("Invalid capability type passed to check_quality_with_gpt")

    for i in range(0, max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=10,
                temperature=1,
            )
            return response.choices[0].message.content
        except HttpResponseError as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Attempt {i + 1} of {max_retries}.")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        raise RuntimeError("Maximum retries exceeded.")
