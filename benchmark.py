# implemting benchmark to compare various llm performance on user requests
import os

import splitter as splitter
import streamlit as lit
import pickle
import time
from langchain import OpenAI, text_splitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader






import time
import requests
import concurrent.futures
from tabulate import tabulate

# Define API keys (Replace with actual API keys)
API_KEYS = {
    "openai": "your-openai-api-key",
    "anthropic": "your-anthropic-api-key",
    "google": "your-google-api-key"
}

# Define API endpoints
API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "google": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
}

# Define prompt
PROMPT = "Explain quantum entanglement in simple terms."

# Function to call OpenAI API
def call_openai(prompt):
    headers = {"Authorization": f"Bearer {API_KEYS['openai']}", "Content-Type": "application/json"}
    data = {"model": "gpt-4", "prompt": prompt, "max_tokens": 200}
    start_time = time.time()
    response = requests.post(API_ENDPOINTS['openai'], json=data, headers=headers)
    elapsed_time = time.time() - start_time
    return response.json().get("choices", [{}])[0].get("text", ""), elapsed_time

# Function to call Anthropic API
def call_anthropic(prompt):
    headers = {"x-api-key": API_KEYS['anthropic'], "Content-Type": "application/json"}
    data = {"model": "claude-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    start_time = time.time()
    response = requests.post(API_ENDPOINTS['anthropic'], json=data, headers=headers)
    elapsed_time = time.time() - start_time
    return response.json().get("content", ""), elapsed_time

# Function to call Google API
def call_google(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEYS['google']}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    start_time = time.time()
    response = requests.post(API_ENDPOINTS['google'], json=data, headers=headers, params=params)
    elapsed_time = time.time() - start_time
    return response.json().get("candidates", [{}])[0].get("content", ""), elapsed_time

# Function to evaluate response length
def evaluate_response_length(response):
    return len(response.split())

# Function to run all models concurrently
def query_all_models(prompt):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "OpenAI GPT-4": executor.submit(call_openai, prompt),
            "Anthropic Claude-3": executor.submit(call_anthropic, prompt),
            "Google Gemini-Pro": executor.submit(call_google, prompt)
        }
        results = {}
        for model, future in futures.items():
            try:
                response, response_time = future.result()
                word_count = evaluate_response_length(response)
                results[model] = (response_time, word_count, response)
            except Exception as e:
                results[model] = (None, None, str(e))
        return results

# Run the benchmark
results = query_all_models(PROMPT)

# Display results in a table
table_data = []
for model, (response_time, word_count, response) in results.items():
    table_data.append([model, f"{response_time:.2f} sec" if response_time else "Error", word_count, response[:100] + "..."])

print(tabulate(table_data, headers=["Model", "Response Time", "Word Count", "Response Preview"], tablefmt="grid"))
