import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

MODEL_DIR = "sentiment_bert_model"
TEST_DATA_DIR = "processed_data/test"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(TEST_DATA_DIR):
    raise FileNotFoundError(
        f"Processed test dataset not found at {TEST_DATA_DIR}. Run prepare_dataset.py first.")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_DIR}. Run train_model.py first.")

print(f"Loading test dataset from {TEST_DATA_DIR}...")

test_dataset = load_from_disk(TEST_DATA_DIR)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Loading model and tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)

model.eval()
all_preds = []
all_labels = []

print("Evaluating model...")
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
