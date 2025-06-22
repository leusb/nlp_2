#!/usr/bin/env python3
"""
Task 2.1: Multiple-Choice Question Answering with zero-shot prompting (no constraints)

This script supports:
- Local LLaMA model via Ollama HTTP API (http://localhost:11434/api/chat)
- OpenAI-compatible API hosted at http://llm.lehre.texttechnologylab.org/v1 (add --remote)

Runs zero-shot MC-QA on oLMpics tasks: Property_Conjunction and Taxonomy_Conjunction.
This version uses prompt engineering only – no logit_bias, no max_tokens.
"""

import sys
import requests
from datasets import load_dataset
from tqdm.auto import tqdm # for progress bar...
import matplotlib.pyplot as plt

# === Configuration ===
USE_REMOTE_API = "--remote" in sys.argv
MODEL_NAME = "llama3.1:8b"

# === API Setup ===
if USE_REMOTE_API:
    from openai import OpenAI
    client = OpenAI(
        base_url="http://llm.lehre.texttechnologylab.org/v1",
        api_key="demo"  # any string is accepted
    )
    print("Using REMOTE API (TextTechnology Server)")
else:
    API_URL = "http://localhost:11434/api/chat"
    HEADERS = {"Content-Type": "application/json"}
    print("Using LOCAL API (Ollama)")

# === System prompts ===
SYSTEM_PROMPT_improved = (
    "You are a helpful assistant for multiple-choice questions from the oLMpics benchmark. "
    "Answer each question by choosing the correct option and reply only with the corresponding letter "
    "(A, B, C, ...). Do not explain your answer."
)
SYSTEM_PROMPT=(
    "You are a helpful assistant for multiple-choice questions from the oLMpics benchmark. "
    "Answer each question"
)

# === Format the request payload (no restrictions for Task 2.1) ===
def format_payload(stem, choices):
    question = stem.replace("[MASK]", "_____")
    options = [f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)]
    user_input = question + "\n\n" + "\n".join(options) + "\nAnswer:"
    print(user_input)

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0,
        "top_p": 1,
        "stream": False
    }

# === Send request to the model ===
def query_llama3(payload):
    if USE_REMOTE_API:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload.get("temperature", 0),
            top_p=payload.get("top_p", 1)
        )
        print(response)
        print(response.choices[0].message.content.strip())
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
    ds = ds.select(range(num_examples))  # limit for testing
    correct = 0
    for ex in tqdm(ds, desc=f"Evaluating {task_name}"):
        payload = format_payload(ex["stem"], ex["choices"])
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
    plt.bar(scores.keys(), scores.values(), color='cornflowerblue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("oLMpics Task Accuracy (Task 2.1 - Zero-Shot Prompting)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
