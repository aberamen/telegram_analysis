from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report
from datasets import load_dataset
from time import time

# Load the dataset
dataset = load_dataset("conll2003", split="train[:10%]")  # Replace with your dataset
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Preprocess the data
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# List of models to compare
models = [
    {"name": "XLM-Roberta", "model": "xlm-roberta-base"},
    {"name": "DistilBERT", "model": "distilbert-base-uncased"},
    {"name": "mBERT", "model": "bert-base-multilingual-cased"}
]

def evaluate_model(model_name, model_path):
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=9)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Start timing the evaluation
    start_time = time()
    eval_results = trainer.evaluate(tokenized_dataset)
    end_time = time()
    
    evaluation_time = end_time - start_time
    print(f"Evaluation time for {model_name}: {evaluation_time:.2f} seconds")
    
    return eval_results

# Compare models
for model_info in models:
    print(f"Evaluating {model_info['name']}...")
    eval_results = evaluate_model(model_info["name"], model_info["model"])
    print(classification_report(eval_results["labels"], eval_results["predictions"]))

