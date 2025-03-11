import random
import json
import torch
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spellchecker import SpellChecker

# Set the device to use for model inference: CUDA (GPU) if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents data to understand expected inputs and outputs
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Define trained data location
FILE = "data.pth"

# Load trained model data
data = torch.load(FILE)

# Extract information from data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model with appropriate size of layers, move to selected device (CPU or GPU)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Load pre-trained data into the model from the model_state dictionary
model.load_state_dict(model_state)
# Set the model to evaluation mode (prevents things like dropout from being active)
model.eval()
# Define bot name
bot_name = "VEGA"
# Initialize spell checker to correct user input
spell = SpellChecker()

# Correct the spelling of words in a sentence
def correct_spelling(sentence):
    # Split the sentence into words
    words = sentence.split()
    # For each word, if it's misspelled, correct it using the spell checker
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    # Join the corrected words back into a full sentence and return it
    return ' '.join(corrected_words)

# Load GPT-2 model and tokenizer from Hugging Face
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Function to generate response using GPT-2
def generate_gpt2_response(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, top_k=50)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


os.system('cls' if os.name == 'nt' else 'clear')

# Loop where the bot listens for user input
while True:
    # Get user input
    sentence = input("User: ")
    # Correct the spelling of the sentence before processing
    sentence = correct_spelling(sentence)
    # Tokenize the sentence (split it into individual words)
    sentence = tokenize(sentence)
    # Convert the tokenized sentence into a bag-of-words format (numerical representation)
    X = bag_of_words(sentence, all_words)
    # Reshape input to fit the model's expected input format (batch size of 1)
    X = X.reshape(1, X.shape[0])
    # Convert input into a PyTorch tensor and move to the appropriate device (CPU or GPU)
    X = torch.from_numpy(X).to(device)
    # Pass input through the model to get predictions
    output = model(X)
    # Get the predicted tag with the highest probability
    _, predicted = torch.max(output, dim=1)
    # Map predicted index to the corresponding tag (category of intent)
    tag = tags[predicted.item()]
    # Apply softmax to the output to get probabilities for each tag
    probs = torch.softmax(output, dim=1)
    # Get probability of the predicted tag
    prob = probs[0][predicted.item()]
    # If the probability of the tag is greater than 75%, respond with an appropriate message
    if prob.item() > 0.75:
        # Loop through all intents and find the corresponding output
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Choose a random response from the list of possible responses for this tag
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
                # If the tag is "goodbye", quit
                if tag == "goodbye":
                    exit()
    else:
        # If the model is unsure (probability too low), output for user clarity
        print(f"{bot_name}: I do not understand. Let me try to generate a response...")

        # If the model is unsure, fall back to GPT-2 to generate a response
        # gpt2_response = generate_gpt2_response(sentence)
        # print(f"{bot_name}: {gpt2_response}")