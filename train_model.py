import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk

# 1. Define Constants
MODEL_NAME = "bert-base-uncased"
PROCESSED_TRAIN_DIR = "processed_data/train"
PROCESSED_TEST_DIR = "processed_data/test"
MODEL_OUTPUT_DIR = "sentiment_bert_model"

# 2. Load Processed Datasets
print(f"Loading processed datasets from {PROCESSED_TRAIN_DIR} and {PROCESSED_TEST_DIR}...")
if not os.path.exists(PROCESSED_TRAIN_DIR) or not os.path.exists(PROCESSED_TEST_DIR):
    print("Error: Processed data not found. Please run prepare_dataset.py first.")
    exit()

train_dataset = load_from_disk(PROCESSED_TRAIN_DIR)
test_dataset = load_from_disk(PROCESSED_TEST_DIR)
print("Datasets loaded.")

# Ensure datasets have the 'torch' format; this should have been set in prepare_dataset.py
# but it's good to be sure or re-apply if necessary.
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# 3. Initialize Tokenizer
print(f"Loading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# 4. Initialize Model
print(f"Initializing model: {MODEL_NAME} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("Model initialized.")

# 5. Define TrainingArguments
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=1,  # Small number for initial run
    per_device_train_batch_size=4, # Reduced batch size
    per_device_eval_batch_size=4,  # Reduced batch size
    warmup_steps=50, # Further reduced warmup steps for smaller dataset/epochs
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch", # Corrected argument name
    # Ensure the model is saved at the end of training
    save_strategy="epoch", # Corrected argument name (assuming it's the same pattern)
    load_best_model_at_end=True, # Loads the best model found during training (based on eval)
)
print("Training arguments defined.")

# 6. Instantiate Trainer
print("Instantiating Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print("Trainer instantiated.")

# 7. Start Training
print("Starting model training...")
trainer.train()
print("Training finished.")

# 8. Save Model and Tokenizer
print(f"Saving model and tokenizer to {MODEL_OUTPUT_DIR}...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
trainer.save_model(MODEL_OUTPUT_DIR) # Saves both model and training_args
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"Model and tokenizer saved to {MODEL_OUTPUT_DIR}.")

print("\nScript finished successfully!")
