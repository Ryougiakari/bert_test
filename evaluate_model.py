# evaluate_model.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import os

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    model_dir = "sentiment_bert_model"
    test_data_dir = "processed_data/test"
    results_dir = "./eval_results" # Directory to save evaluation results/logs

    print(f"Loading tokenizer from {model_dir}...")
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print(f"Error: Model directory {model_dir} is empty or does not exist.")
        print("Please ensure the model has been trained and saved correctly using train_model.py.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading model from {model_dir}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading test dataset from {test_data_dir}...")
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory {test_data_dir} does not exist.")
        print("Please ensure the dataset has been processed using prepare_dataset.py.")
        return
    try:
        test_dataset = load_from_disk(test_data_dir)
        # Ensure columns are correctly named for the Trainer
        if 'label' in test_dataset.column_names and 'labels' not in test_dataset.column_names:
            test_dataset = test_dataset.rename_column("label", "labels")

        required_cols = {'input_ids', 'attention_mask', 'labels'}
        if not required_cols.issubset(test_dataset.column_names):
            print(f"Error: Test dataset is missing required columns. Found: {test_dataset.column_names}. Required: {required_cols}")
            return

    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    print("Defining training arguments for evaluation...")
    # Ensure the output directory for evaluation results exists
    os.makedirs(results_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=results_dir,
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        report_to="none" # Disable reporting to wandb/tensorboard if not configured
    )

    print("Initializing Trainer for evaluation...")
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    print("Starting evaluation...")
    try:
        eval_results = trainer.evaluate()
        print("Evaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
