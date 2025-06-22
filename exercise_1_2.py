import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, logging
from tqdm.auto import tqdm  # progress bar
import matplotlib.pyplot as plt

# remove warnings
logging.set_verbosity_error()

########### CONSTANTS ###########
# List of task variants for ex 1-2
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

# Load tokenizers and models
print("Loading models and tokenizers: ")
loaded_models = {}
for model_name, checkpoint in MODELS_DICT.items():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  # use AutoTokenizer
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)  # load pretrained model
    model.eval()  # use evaluation mode (no gradients)
    loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}  # store both in dictionary
print("Loading done.\n")

# Evaluation function: simple accuracy with optional perturbation for Age_Comparison
def evaluate_mc_mlm(model, tokenizer, examples, perturb=False):
    """
    Evaluates a masked language model on a multiple-choice dataset.

    Optionally perturbs prompts (only for Age_Comparison) to test shortcut learning.
    
    Args:
        model: The MLM model
        tokenizer: The corresponding tokenizer
        examples: List of examples from the dataset
        perturb: Whether to apply input perturbation

    Returns:
        Accuracy (float) over all examples
    """
    correct_predictions = 0
    for example in tqdm(examples, desc="Evaluating", leave=False):
        prompt = example["stem"].replace("[MASK]", tokenizer.mask_token)  # replace with real mask token
        if perturb:
            # replace 'age' -> 'blah', 'than' -> 'da'
            prompt = prompt.replace("age", "blah").replace("than", "da")

        inputs = tokenizer(prompt, return_tensors="pt")  # tokenize prompt
        mask_position = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero()  # locate mask token
        if mask_position.size(0) == 0:
            raise ValueError(f"No mask token found: {prompt}")
        mask_position = mask_position[0].item()  # convert to int

        with torch.no_grad():  # inference only
            logits = model(**inputs).logits[0, mask_position]  # get logits at mask position

        option_scores = []
        for option in example["choices"]:  # loop answer options
            option_token_id = tokenizer(option, add_special_tokens=False).input_ids[0]  # get token ID
            option_scores.append(logits[option_token_id].item())  # score from logits

        predicted_index = int(torch.tensor(option_scores).argmax())  # model prediction (index of highest score)
        true_index = example.get("label", example.get("answerKey"))  # correct answer index
        if predicted_index == true_index:
            correct_predictions += 1
    return correct_predictions / len(examples)  # accuracy

# Zero-shot evaluation (original prompts)
original_results = {m: {} for m in loaded_models}  # result dict for models
for task in VARIANTS:
    print(f"Loading dataset {task}...")
    dataset = load_dataset("KevinZ/oLMpics", task, split="test")  # load test split
    for model_name, model_data in loaded_models.items():
        accuracy = evaluate_mc_mlm(model_data["model"], model_data["tokenizer"], dataset, perturb=False)  # no perturbation
        original_results[model_name][task] = accuracy
        print(f"{model_name.upper():7} accuracy on {task}: {accuracy:.2%}")
    print()

# Zero-shot evaluation (perturbed) for Age_Comparison
perturbed_results = {m: {} for m in loaded_models}  # result dict for perturbed inputs
task = "Age_Comparison"
print(f"Loading dataset {task} (perturbed)...")
dataset = load_dataset("KevinZ/oLMpics", task, split="test")
for model_name, model_data in loaded_models.items():
    accuracy = evaluate_mc_mlm(model_data["model"], model_data["tokenizer"], dataset, perturb=True)  # with perturbation
    perturbed_results[model_name][task] = accuracy
    print(f" {model_name.upper()} accuracy on {task} (perturbed): {accuracy:.2%}")
print()

#  Plotting results (original + perturbed) 
all_tasks = VARIANTS + [f"{task}_perturbed"]  # include perturbation
x = range(len(all_tasks))

bert_scores = [original_results["bert"][v] for v in VARIANTS] + [perturbed_results["bert"][task]]
roberta_scores = [original_results["roberta"][v] for v in VARIANTS] + [perturbed_results["roberta"][task]]

plt.figure(figsize=(12, 6))
plt.bar([i - 0.3 for i in x], bert_scores,    width=0.3, label="BERT-Base")
plt.bar([i for i in x], roberta_scores, width=0.3, label="RoBERTa-Base")
plt.xticks(x, all_tasks, rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Zero-Shot MC-MLM Accuracy on oLMpics (incl. perturbed Age_Comparison)")
plt.legend()
plt.tight_layout()
plt.show()

# Print summary
print("Summary of results (original):")
for task in VARIANTS:
    print(f"{task}: BERT={original_results['bert'][task]:.2%}, RoBERTa={original_results['roberta'][task]:.2%}")

print("\nSummary of results (perturbed Age_Comparison):")
print(f"Age_Comparison (perturbed): "
      f"BERT={perturbed_results['bert']['Age_Comparison']:.2%}, "
      f"RoBERTa={perturbed_results['roberta']['Age_Comparison']:.2%}")
