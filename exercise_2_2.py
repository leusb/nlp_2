"""Exercise 2.2

Supported: 
- Local LLaMA model via Ollama HTTP API (http://localhost:11434/api/chat)
"""

import sys
import requests
from datasets import load_dataset
from tqdm.auto import tqdm # for progress bar...
import matplotlib.pyplot as plt
from openai import OpenAI
import tiktoken 

########### CONSTANTS ###########
MODEL_NAME     = "llama3.1:8b"
BASE_URL       = "http://llm.lehre.texttechnologylab.org/v1"
API_KEY        = "demo"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
print("Using REMOTE API only")

enc = tiktoken.get_encoding("cl100k_base")  # passt für viele OpenAI-Models
id_A = enc.encode("A")[0]
id_B = enc.encode("B")[0]
id_C = enc.encode("C")[0]

logit_bias = {
    str(id_A): 100, 
    str(id_B): 100,
    str(id_C): 100 
}




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
        "temperature": 0,            # deterministic
        "max_tokens": 1,             # one token only
        "stop": ["\n"],              # stop
        "logit_bias": logit_bias     # bias towards A,B and C
    }

# request
def query_llama3(payload):
    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        temperature=payload["temperature"],
        max_tokens=payload["max_tokens"],
        stop=payload["stop"],
        logit_bias=payload["logit_bias"],
        logprobs=5
    )
    #testing
    #print(response.choices[0].logprobs)  
    # print(response)
    #print(response.choices[0].message.content.strip())
    ## Reminder: ChatCompletion(id='chatcmpl-VOsWvdIFd9QdPz8zCQG2ZkIhHj2cj1Gd', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='A) mammal', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1750703284, model='llama3.1:8b', object='chat.completion', service_tier=None, system_fingerprint='b5700-8d947136', usage=CompletionUsage(completion_tokens=5, prompt_tokens=92, total_tokens=97, completion_tokens_details=None, prompt_tokens_details=None), timings={'prompt_n': 35, 'prompt_ms': 20.233, 'prompt_per_token_ms': 0.5780857142857143, 'prompt_per_second': 1729.8472791973509, 'predicted_n': 5, 'predicted_ms': 55.373, 'predicted_per_token_ms': 11.0746, 'predicted_per_second': 90.2967150055081})
    return response.choices[0].message.content.strip()
    
# extract key from answr
def extract_key(reply):
    import re
    match = re.search(r"\b([A-C])\b", reply) # get capital single character and use word boundray
    return match.group(1) if match else ""

# main loop for accuarcy
def evaluate_mc_qa(task_name, num_examples=10):
    dataset = load_dataset("KevinZ/oLMpics", task_name, split="test")
    dataset = dataset.select(range(num_examples))  # limit for testing
    correct = 0
    for exercise in tqdm(dataset, desc=f"Evaluating {task_name}"):
        payload = format_querry(exercise["stem"], exercise["choices"]) #build paylod
        reply = query_llama3(payload)
        key = extract_key(reply)
        prediction = ord(key) - ord("A") if key else -1 # convert to matching nubmer from ds: A->0, B->1, C-> 2
        label = exercise.get("label", exercise.get("answerKey")) # answer key
        if prediction == label:
            correct += 1
    accuracy = correct / len(dataset)
    print(f"{task_name} Accuracy: {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":
    tasks = ["Property_Conjunction", "Taxonomy_Conjunction"]
    scores = {}
    for task in tasks:
        scores[task] = evaluate_mc_qa(task)

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values(), color='blue')
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("oLMpics Task Accuracy (Task 2.1 - Zero-Shot Prompting)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
