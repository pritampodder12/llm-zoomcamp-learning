import json
import os
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv
from minsearch import Index

load_dotenv()

# Prepareing Dataset
CACHE_FILE = Path("data/documents.json")
DOCS_URL = "https://datatalks.club/faq/json/courses.json"
URL_PREFIX = "https://datatalks.club/faq"


def fetch_documents() -> list:
    response = requests.get(DOCS_URL)
    response.raise_for_status()
    courses_raw = response.json()

    documents = []
    for course in courses_raw:
        course_url = f"""{URL_PREFIX}{course["path"]}"""
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()
        documents.extend(course_data)
    return documents


def load_documents(refresh: bool = False) -> list:
    if CACHE_FILE.exists() and not refresh:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    documents = fetch_documents()

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    return documents


# Indexing and Searching
documents = load_documents()
index = Index(text_fields=["question", "section", "answer"], keyword_fields=["course"])
index.fit(documents)


def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question, boost_dict=boost_dict, filter_dict=filter_dict, num_results=5
    )


# Building Context
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_context(search_results: list) -> str:
    lines = []
    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")
    return "\n".join(lines).strip()


# Build Prompt
def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt.strip()


# Call LLM
client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))


def call_llm(instructions: str, prompt: str) -> str:
    response = client.converse(
        modelId=os.getenv("MODEL_ID"),
        system=[{"text": instructions}],
        messages=[
            {"role": "user", "content": [{"text": prompt}]},
        ],
        inferenceConfig={"maxTokens": 256, "temperature": 0.5},
    )
    print(response["usage"])
    # print(response)
    return response["output"]["message"]["content"][0]["text"]


# Rag
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = call_llm(INSTRUCTIONS, prompt)
    return answer


answer = rag("How do I get a certificate?")
print(answer)
