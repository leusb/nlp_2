#!/usr/bin/env python3
"""
Task 2.2: Multiple-Choice Question Answering with output restriction (logit_bias)

This script supports:
- Local LLaMA model via Ollama HTTP API (http://localhost:11434/api/chat)
- OpenAI-compatible API hosted at http://llm.lehre.texttechnologylab.org/v1 (add --remote)

Runs zero-shot MC-QA on oLMpics tasks: Property_Conjunction and Taxonomy_Conjunction.
Uses logit_bias and max_tokens to restrict the model output to answer keys (A, B, C, ...).
"""

import sys
import requests
from datasets import load_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# === Configuration ===
USE_REMOTE_API = "--remote" in sys.argv
MODEL_NAME = "llama3.1:8b"

# === API Setup ===
if USE_REMOTE_API:
    from openai import OpenAI
    client = OpenAI(
        base_url="http://llm.lehre.texttechnologylab.org/v1",
        api_key="some_string"  # any string
    )
    print(" Using REMOTE API (TextTechnology Server)")
else:
    API_URL = "http://localhost:11434/api/chat"
    HEADERS = {"Content-Type": "application/json"}
    print("Using LOCAL API")

# === System prompt ===
SYSTEM_PROMPT = (
    "You are a multiple-choice assistant for oLMpics tasks "
    "(Property_Conjunction, Taxonomy_Conjunction). "
    "Only reply with a single uppercase letter (A, B, C, ...). "
    "The client will only accept one of those letters."
)

# === Format the request payload with output restriction ===
def format_payload_constrained(stem, choices):
    question = stem.replace("[MASK]", "_____")
    options = [f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)]
    user_input = question + "\n\n" + "\n".join(options) + "\nAnswer:"
    # print(user_input)

    allowed = [chr(ord("A") + i) for i in range(len(choices))]
    logit_bias = {str(ord(c)): 100 for c in allowed}  # ASCII-based token IDs

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "stop": ["\n"],
        "logit_bias": logit_bias,
        "stream": False
    }

# === Send request to the model ===
def query_llama3(payload):
    if USE_REMOTE_API:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            max_tokens=payload["max_tokens"],
            stop=payload["stop"],
            logit_bias=payload["logit_bias"]
        )
        print (response)
        print (response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    else:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(f"HTTP Error {response.status_code}: {response.text}")
            raise
        data = response.json()
        if "message" in data:
            return data["message"]["content"].strip()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

# === Extract answer key from model response ===
def extract_key(reply: str) -> str:
    import re
    match = re.search(r"\b([A-Z])\b", reply)
    return match.group(1) if match else ""

# === Main evaluation logic ===
def evaluate_mc_qa(task_name: str, num_examples=10) -> float:
    ds = load_dataset("KevinZ/oLMpics", task_name, split="test")
    ds = ds.select(range(num_examples))  # limit for quick test runs
    correct = 0
    for ex in tqdm(ds, desc=f"Evaluating {task_name}"):
        payload = format_payload_constrained(ex["stem"], ex["choices"])
        reply = query_llama3(payload)
        key = extract_key(reply)
        pred_idx = ord(key) - ord("A") if key else -1
        true_idx = ex.get("label", ex.get("answerKey"))
        if pred_idx == true_idx:
            correct += 1
    accuracy = correct / len(ds)
    print(f"{task_name:25s} Accuracy: {accuracy:.2%}")
    return accuracy

# === Run tasks and visualize results ===
if __name__ == "__main__":
    tasks = ["Property_Conjunction", "Taxonomy_Conjunction"]
    scores = {}
    for task in tasks:
        scores[task] = evaluate_mc_qa(task)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values(), color='mediumseagreen')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("oLMpics Task Accuracy (Task 2.2 with logit_bias)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
