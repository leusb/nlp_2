"""Exercise 2.1 with modified datasets

Supported: 
- Local LLaMA model via Ollama HTTP API (http://localhost:11434/api/chat)
- OpenAI-compatible API hosted at http://llm.lehre.texttechnologylab.org/v1 (add --remote)
"""

import sys
import requests
from datasets import load_dataset
from tqdm.auto import tqdm # for progress bar...
import matplotlib.pyplot as plt
import os
import csv

########### CONSTANTS ###########
USE_REMOTE_API = "--remote" in sys.argv
MODEL_NAME = "llama3.1:8b"


# API:
if USE_REMOTE_API:
    from openai import OpenAI
    client = OpenAI(
        base_url="http://llm.lehre.texttechnologylab.org/v1",
        api_key="demo"  # any string is accepted
    )
    print("Using REMOTE API (TextTechnology Server): ")
else:
    API_URL = "http://localhost:11434/api/chat"
    HEADERS = {"Content-Type": "application/json"}
    print("Using LOCAL API ")

#improved
SYSTEM_PROMPT_improved = (
    "You are a helpful assistant for multiple-choice questions from the oLMpics benchmark. "
    "Answer each question by choosing the correct option and reply only with the corresponding letter "
    "(A, B, C, ...). Do not explain your answer."
)

#old
SYSTEM_PROMPT=(
    "You are a helpful assistant for multiple-choice questions from the oLMpics benchmark. "
    "Answer each question"
)

########### CONSTANTS ###########

# format querry
def format_querry(stem, choices):
    question = stem.replace("[MASK]", "___") # create placeholder in string
    options = [f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)]
    user_input = question + "\n\n" + "\n".join(options) + "\nAnswer:"

    #print(user_input)

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_improved},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0,
        "stream": False
    }

# request
def query_llama3(payload):
    if USE_REMOTE_API:
        response = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload.get("temperature", 0),
        )
        # print(response)
        # print(response.choices[0].message.content.strip())
        ## Reminder: ChatCompletion(id='chatcmpl-VOsWvdIFd9QdPz8zCQG2ZkIhHj2cj1Gd', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='A) mammal', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1750703284, model='llama3.1:8b', object='chat.completion', service_tier=None, system_fingerprint='b5700-8d947136', usage=CompletionUsage(completion_tokens=5, prompt_tokens=92, total_tokens=97, completion_tokens_details=None, prompt_tokens_details=None), timings={'prompt_n': 35, 'prompt_ms': 20.233, 'prompt_per_token_ms': 0.5780857142857143, 'prompt_per_second': 1729.8472791973509, 'predicted_n': 5, 'predicted_ms': 55.373, 'predicted_per_token_ms': 11.0746, 'predicted_per_second': 90.2967150055081})
        return response.choices[0].message.content.strip()
    else:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        data = response.json()
        print(data)
        # Reminder: {'model': 'llama3.1:8b', 'created_at': '2025-06-23T18:34:00.563771Z', 'message': {'role': 'assistant', 'content': 'C'}, 'done_reason': 'stop', 'done': True, 'total_duration': 188020416, 'load_duration': 11735208, 'prompt_eval_count': 90, 'prompt_eval_duration': 154065583, 'eval_count': 2, 'eval_duration': 21581083}
        return data["message"]["content"].strip()

# extract key from answr
def extract_key(reply):
    import re
    match = re.search(r"\b([A-C])\b", reply) # get capital single character and use word boundray
    return match.group(1) if match else ""



# main loop for accuarcy
def evaluate_mc_qa(task_name,csv_flag=False, num_examples=10):
    dataset = load_dataset("KevinZ/oLMpics", task_name, split="test")
    #dataset = dataset.select(range(num_examples))  # limit for testing

    if csv_flag:
        # Creating csv-output:
        os.makedirs("results_2_1_new", exist_ok=True)
        csv_path = f"results_2_1_new/{task_name}_llama3.csv"
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["index", "stem", "choices", "true_label", "model_reply", "extracted_key", "predicted_label", "correct"])

    correct = 0
    for id, exercise in enumerate(tqdm(dataset, desc=f"Evaluating {task_name}")):
        payload = format_querry(exercise["stem"], exercise["choices"]) #build paylod
        reply = query_llama3(payload)
        key = extract_key(reply)
        prediction = ord(key) - ord("A") if key else -1 # convert to matching nubmer from ds: A->0, B->1, C-> 2
        label = exercise.get("label", exercise.get("answerKey")) # answer key
        if prediction == label:
            correct += 1

        if csv_flag:
            # Write to CSV
            csv_writer.writerow([
                id,
                exercise["stem"],
                " | ".join(exercise["choices"]),
                label,
                reply,
                key,
                prediction,
                int(prediction == label)
            ])


    accuracy = correct / len(dataset)
    print(f"{task_name} Accuracy: {accuracy:.2%}")

    if csv_flag:
        csv_file.close()
        print(f"CSV results written to {csv_path}")

    return accuracy

if __name__ == "__main__":
    # changed:
    tasks = ["Property_Conjunction", "Encyclopedic_Composition"]
    scores = {}
    for task in tasks:
        scores[task] = evaluate_mc_qa(task, True)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values(), color='blue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("oLMpics Task Accuracy (Task 2.1 - Zero-Shot Prompting)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
