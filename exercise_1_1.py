"""Exercise 1.1"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
from tqdm.auto import tqdm # progress bar
import matplotlib.pyplot as plt

#remove warning: Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForMaskedLM:...
logging.set_verbosity_error() 

########### CONSTANTS ###########
# List of task variants for ex 1-1
VARIANTS = [
    "Age_Comparison",
    "Always_Never",
    "Object_Comparison",
    "Antonym_Negation",
    "Taxonomy_Conjunction",
    "Multihop_Composition"
]

MODELS_DICT = {
    "bert": "bert-base-cased",
    "roberta": "roberta-base",
}

########### CONSTANTS ###########

#Load tokenizers and models
print("Loading models and tokenisers: ")
models = {}
for name, checkpoint in MODELS_DICT.items():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) # use autotokenzier
    model  = AutoModelForMaskedLM.from_pretrained(checkpoint)
    model.eval() # use evaulation mode -> inference only
    models[name] = {"tokenizer": tokenizer, "model": model} # store tokenizer and model
print("Loading done.\n")



# Evaluation function: simple accuracy
def evaluate_mc_mlm(model, tokenizer, examples):
    """
    Evaluates a masked language model.

    For each example:
    - Replaces [MASK] in the prompt with the model's actual mask token
    - Feeds the prompt into the model
    - Scores each multiple-choice option based on model confidence at the masked position
    - Selects the highest-scoring option as the prediction
    - Compares it to the ground-truth label to compute accuracy

    Args:
        model: The Hugging Face model for masked language modeling.
        tokenizer: The corresponding tokenizer for the model.
        examples: List of examples from the oLMpics dataset.
            Each example does have a "stem", "choices", and "label" or "answerKey".

    Returns:
        float: Accuracy over the examples (correct / total).
    """
    correct = 0
    for example in tqdm(examples, desc="Evaluating", leave=False):
        prompt = example["stem"].replace("[MASK]", tokenizer.mask_token) # [mask] for bert and <mask> for roberta
        options = example["choices"] # answers
        true_index = example.get("label", example.get("answerKey")) #retrievce correct answer

        inputs = tokenizer(prompt, return_tensors="pt") # tokenize
        mask_position = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero() # get mask-token position
        mask_position = mask_position[0].item() # convert tensor

        with torch.no_grad(): # just forward passes (for inference only tasks )
            logits = model(**inputs).logits[0, mask_position] # get logits at mask positon

        # find token id for each option and look up the score for that token at mask position
        scores = [] 
        for option in options:
            opt_id = tokenizer(option, add_special_tokens=False).input_ids[0]
            scores.append(logits[opt_id].item())

        pred = int(torch.tensor(scores).argmax()) # pick models preferred answer
        if pred == true_index: # check if its correct 
            correct += 1 # update accuracy
    return correct / len(examples) 

# Zero-shot evaluation
results = {m: {} for m in models} # result dict for models
for variant in VARIANTS: # loop datasets
    print(f"Loading dataset {variant}: ")
    dataset = load_dataset("KevinZ/oLMpics", variant, split="test")
    for mname, md in models.items():
        acc = evaluate_mc_mlm(md["model"], md["tokenizer"], dataset)
        results[mname][variant] = acc
        print(f"  {mname.upper()} accuracy on {variant}: {acc:.2%}")
    print()

# Plotting results

x = range(len(VARIANTS))

bert_scores    = [results["bert"][v]    for v in VARIANTS]
roberta_scores = [results["roberta"][v] for v in VARIANTS]

plt.figure(figsize=(10, 6))
plt.bar([i - 0.2 for i in x], bert_scores,    width=0.4, label="BERT")
plt.bar([i + 0.2 for i in x], roberta_scores, width=0.4, label="RoBERTa")
plt.xticks(x, VARIANTS, rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Zero-Shot MC-MLM Accuracy on oLMpics")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Print summary
print("\nSummary of results:")
for variant in VARIANTS:
    print(f"{variant}: BERT={results['bert'][variant]:.2%}, RoBERTa={results['roberta'][variant]:.2%}")
