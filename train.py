import numpy as np
import random
import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load the intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# List to store all unique words from the patterns
all_words = []
# List to store tags
tags = []
# List to store pairs of patterns and their tags
xy = []

# Loop each intent in the intents file
for intent in intents['intents']:
    # Retrieve tag
    tag = intent['tag']
    # Add the tag to tags list
    tags.append(tag)
    # Loop through each pattern under the current tag
    for pattern in intent['patterns']:
        # Tokenize the pattern into words
        w = tokenize(pattern)
        # Add the tokenized words to all_words list
        all_words.extend(w)
        # Add the pair of words and tag to xy list
        xy.append((w, tag))

# Remove punctuation marks and stem the words; reduce to basic form
# List of words to ignore (punctuation marks)
ignore_words = ['?', '.', '!']
# Stem and filter out ignored words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort list of words
all_words = sorted(set(all_words))
# Remove duplicate tags and sort tags list
tags = sorted(set(tags))

# Print out statistics about the data
# Print the number of pattern-tag pairs
print(len(xy), "patterns")
# Print the number of tags
print(len(tags), "tags:", tags)
# Print the number of unique stemmed words
print(len(all_words), "unique stemmed words:", all_words)

# Prepare training data
# Store the bag of words for each sentence
X_train = []
# Store the corresponding tags for each sentence
y_train = []

# Convert each pattern into a bag of words and store it in the training data
for (pattern_sentence, tag) in xy:
    # Create a bag of words representation for pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    # Add the bag of words to the training input list
    X_train.append(bag)
    # Tag is converted to an integer index
    label = tags.index(tag)
    # Add the tag's index ton the training output list
    y_train.append(label)

# Convert the training data into numpy arrays
# Convert input features to numpy array
X_train = np.array(X_train)
# Convert labels to numpy array
y_train = np.array(y_train)

# Hyperparameters for model training
# Number of epochs to train the model
num_epochs = 1000
# Number of samples per batch
batch_size = 8
# Learning rate for optimizer
learning_rate = 0.001
# Number of features in the input (length of bag of words)
input_size = len(X_train[0])
# Number of hidden units in the neural network
hidden_size = 8
# Number of output classes (equal to the number of tags)
output_size = len(tags)
# Print input and output size
print(input_size, output_size)

# Create Dataset class for PyTorch
class ChatDataset(Dataset):
    def __init__(self):
        # Total number of samples
        self.n_samples = len(X_train)
        # Input data (bag of words)
        self.x_data = X_train
        # Output data (tags)
        self.y_data = y_train

    def __getitem__(self, index):
        # Return the sample at the given index
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.n_samples

# Create a DataLoader to load data in batches during training
# Init dataset
dataset = ChatDataset()
# Create the DataLoader
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init neural network model
# Move model to GPU if available
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
# Cross entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()
# Adam optimizer with specified learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# Loop through each epoch
for epoch in range(num_epochs):
    # Loop through each batch in the DataLoader
    for (words, labels) in train_loader:
        # Move words to the correct device (GPU or CPU)
        words = words.to(device)
        # Convert labels to long tensor and move to device
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass: calculate model outputs
        # Get model predictions for current batch
        outputs = model(words)

        # Compute loss
        # Calculate loss between predictions and true labels
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients and optimize model
        # Zero out previous gradients
        optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()

    # Print the loss at every 100th epoch
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the final loss after all epochs
print(f'final loss: {loss.item():.4f}')

# Save the trained model and related data
data = {
    # Save model parameters
    "model_state": model.state_dict(),
    # Save input size
    "input_size": input_size,
    # Save hidden layer size
    "hidden_size": hidden_size,
    # Save output size
    "output_size": output_size,
    # Save list of all words
    "all_words": all_words,
    # Save list of tags
    "tags": tags
}

# Save the model data
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete.')
