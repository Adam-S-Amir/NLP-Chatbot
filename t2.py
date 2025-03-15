import numpy as np
import random
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the intents dataset
Path = "data/raw/"
FileName = "salon"
FullPath = Path + FileName + ".json"
with open(FullPath, 'r') as f:
    intents = json.load(f)

# Prepare data for intent classification
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Convert patterns to training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(input_size, output_size)

# Create PyTorch dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train intent classifier model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save intent classification model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "data/processed/salon.pth")
print(f'Intent classification training complete.')

# =============================
# GPT-2 Fine-Tuning
# =============================

# Convert intent data into conversation-style text for GPT-2
train_text = ""

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        response = random.choice(intent["responses"])
        train_text += f"User: {pattern}\nBot: {response}\n\n"

# Save the training data for GPT-2
with open("data/processed/salon_gpt2.txt", "w") as f:
    f.write(train_text)

print("Training data prepared for GPT-2 fine-tuning.")

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset for training
dataset = load_dataset("text", data_files={"train": "data/processed/salon_gpt2.txt"})

def tokenize_function(examples):
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training parameters for GPT-2
training_args = TrainingArguments(
    output_dir="data/models/gpt2_salon",
    evaluation_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],  # Only use train dataset
    # Remove eval_dataset if validation set is missing
)


# Train GPT-2
trainer.train()

# Save fine-tuned GPT-2 model
model.save_pretrained("data/models/gpt2_salon")
tokenizer.save_pretrained("data/models/gpt2_salon")

print("GPT-2 fine-tuning complete.")
