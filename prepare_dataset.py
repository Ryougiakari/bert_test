import os
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Define Model Name
MODEL_NAME = "bert-base-uncased"

# 2. Load Dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
print("Dataset loaded.")

# 3. Initialize Tokenizer
print(f"Loading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# 4. Define Tokenization Function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 5. Apply Tokenization
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized.")

# 6. Select Subsets and Shuffle
# Shuffle the datasets first
shuffled_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
shuffled_test_dataset = tokenized_datasets["test"].shuffle(seed=42)

# Select subsets
# It's important to select after shuffling to get a random subset
# Also, it's good practice to ensure the subset size is not larger than the dataset itself.
train_subset_size = 1000
test_subset_size = 500

print(f"Selecting {train_subset_size} samples for training and {test_subset_size} for testing...")
train_dataset_subset = shuffled_train_dataset.select(range(min(train_subset_size, len(shuffled_train_dataset))))
test_dataset_subset = shuffled_test_dataset.select(range(min(test_subset_size, len(shuffled_test_dataset))))
print("Subsets selected.")

# 7. Set Format to PyTorch
print("Setting dataset format to 'torch'...")
train_dataset_subset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset_subset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print("Dataset format set.")

# 8. Print a Few Examples
print("\nProcessed Training Data Examples:")
for i in range(min(3, len(train_dataset_subset))): # Print at most 3 examples
    print(f"Example {i+1}:")
    print({
        "input_ids": train_dataset_subset[i]["input_ids"].shape,
        "attention_mask": train_dataset_subset[i]["attention_mask"].shape,
        "label": train_dataset_subset[i]["label"]
    })
    # If you want to see the decoded tokens:
    # print(f"Decoded input: {tokenizer.decode(train_dataset_subset[i]['input_ids'])}")


print("\nProcessed Test Data Examples:")
for i in range(min(3, len(test_dataset_subset))): # Print at most 3 examples
    print(f"Example {i+1}:")
    print({
        "input_ids": test_dataset_subset[i]["input_ids"].shape,
        "attention_mask": test_dataset_subset[i]["attention_mask"].shape,
        "label": test_dataset_subset[i]["label"]
    })
    # If you want to see the decoded tokens:
    # print(f"Decoded input: {tokenizer.decode(test_dataset_subset[i]['input_ids'])}")

# 9. Save Processed Datasets
train_output_dir = "processed_data/train"
test_output_dir = "processed_data/test"

print(f"\nSaving processed training dataset to {train_output_dir}...")
os.makedirs(train_output_dir, exist_ok=True)
train_dataset_subset.save_to_disk(train_output_dir)
print("Training dataset saved.")

print(f"Saving processed test dataset to {test_output_dir}...")
os.makedirs(test_output_dir, exist_ok=True)
test_dataset_subset.save_to_disk(test_output_dir)
print("Test dataset saved.")

print("\nScript finished successfully!")
