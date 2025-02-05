import os
import requests
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import urllib

load_dotenv()


def benchmark_with_azure(
    model_name,
    discharge_summary,
    question,
):
    if "gpt" in model_name:
        if "4o" in model_name:
            endpoint = os.getenv("AZURE_GPT_4O_ENDPOINT")
            api_key = os.getenv("AZURE_GPT_4O_API_KEY")
        else:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_KEY")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.getenv("AZURE_API_VERSION"),
        )

    system_message = """
        You are an expert medical professional tasked 
        with answering a clinical question to the best of your ability. You 
        must construct your answer based on the evidence provided to you in 
        the discharge summary.
        """

    user_prompt = f"""
        Your task is to answer a clinical question based on the
        following discharge summary:\n{discharge_summary}\n\n
        You should give your answer in the
        following format:s
        Answer: [your answer]
        Question: {question}\n\n
        Answer:
        """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    if "gpt" in model_name:
        result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=999,
            temperature=0,
        )

    elif "Llama-3" in model_name:

        llama_endpoint = os.getenv("AZURE_LLAMA_3_ENDPONT")
        llama_api_key = os.getenv("AZURE_LLAMA_3_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llama_api_key}",
        }

        data = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": 999,
        }

        response = requests.post(
            f"{llama_endpoint}/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code == 200:
            result = response.json()
            choice = result["choices"][0]
            return choice["message"]["content"]
        else:
            print(f"An error occured, status code: {response.status_code}")
            return 0

    elif "Mistral" in model_name:
        mistral_endpoint = os.getenv("AZURE_MISTRAL_LARGE_ENDPOINT")
        mistral_api_key = os.getenv("AZURE_MISTRAL_LARGE_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mistral_api_key}",
        }

        data = {"messages": messages, "max_tokens": 4096, "temperature": 0}
        body = str.encode(json.dumps(data))

        req = urllib.request.Request(mistral_endpoint, body, headers)
        response = urllib.request.urlopen(req)
        result = response.read()
        response_json = json.loads(result)
        content = response_json["choices"][0]["message"]["content"]
        return content

    else:
        raise ValueError("Model name not recognised by script.")

    return result.choices[0].message.content
