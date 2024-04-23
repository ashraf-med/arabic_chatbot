import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from data.preprocessing import clean_df


df = pd.read_csv('data/data.csv')
df = clean_df(df)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

class RequestDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_length):
        self.encodings = tokenizer(data_df['request'].tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = data_df['encoded_intent'].tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



num_labels = len(df['encoded_intent'].unique())

model_name = "aubmindlab/bert-base-arabertv01"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = RequestDataset(train_df, tokenizer, max_length=128)
val_dataset = RequestDataset(val_df, tokenizer, max_length=128)
test_dataset = RequestDataset(test_df, tokenizer, max_length=128)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# optimizer = AdamW(model.parameters(), lr=5e-5)
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Test the model on the test set
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs.logits, dim=1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')


# Save the fine-tuned model
model_path = "model/fine_tuned_model" 
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
