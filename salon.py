import random
import json
import torch
import os
import dateparser

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spellchecker import SpellChecker

# Set the device to use for model inference: CUDA (GPU) if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents data to understand expected inputs and outputs
with open('data/raw/salon.json', 'r') as json_data:
    intents = json.load(json_data)

# Define trained data location
FILE = "./data/processed/salon.pth"

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

# Use dateparser to try and extract a date from the sentence
def extract_date_time(sentence):
    date_time = dateparser.parse(sentence)
    if date_time:
        return date_time
    else:
        return None

# Load GPT-2 model and tokenizer from Hugging Face
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Function to generate response using GPT-2
def generate_gpt2_response(prompt):
    # Encode the prompt into tokens
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Create attention mask (1s for real tokens, 0s for padding tokens)
    attention_mask = torch.ones(inputs.shape, device=device)  # All tokens are real in this case
    # Generate output using GPT-2
    outputs = gpt2_model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,  # Allow for sampling (more varied responses)
        attention_mask=attention_mask,  # Pass the attention mask
        pad_token_id=gpt2_tokenizer.eos_token_id  # Set pad token to eos_token_id
    )
    # Decode the generated tokens and return the response
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()


# Define conversation state to manage conversation flow
conversation_state = {
    "greeting": False,
    "appointment": False,
    "provide_name": False,
    "provide_phone": False,
    "confirm_appointment": False
}

os.system('cls' if os.name == 'nt' else 'clear')

# Loop where the bot listens for user input
while True:
    # Get user input
    sentence = input("User: ")
    # Correct the spelling of the sentence before processing
    sentence = correct_spelling(sentence)
    # Tokenize the sentence (split it into individual words)
    sentence_tokens = tokenize(sentence)
    # Convert the tokenized sentence into a bag-of-words format (numerical representation)
    X = bag_of_words(sentence_tokens, all_words)
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
    if prob.item() > 0.75:
        # Handle conversation flow based on the current state
        if tag == "greeting" and not conversation_state["greeting"]:
            response = random.choice(intents['intents'][0]['responses'])
            conversation_state["greeting"] = True
        elif tag == "book_appointment" and conversation_state["greeting"] and not conversation_state["appointment"]:
            # If appointment is mentioned, check if day and time are present
            date_time = extract_date_time(sentence)
            if date_time:
                response = f"Let's schedule for{date_time.strftime('%A, %B %d at %I:%M %p')}."
            # response = random.choice(intents['intents'][1]['responses'])
            conversation_state["appointment"] = True
        elif tag == "provide_name" and conversation_state["appointment"] and not conversation_state["provide_name"]:
            response = random.choice(intents['intents'][2]['responses'])
            conversation_state["provide_name"] = True
        elif tag == "provide_phone" and conversation_state["provide_name"] and not conversation_state["provide_phone"]:
            response = random.choice(intents['intents'][3]['responses'])
            conversation_state["provide_phone"] = True
        elif tag == "confirm_appointment" and conversation_state["provide_phone"] and not conversation_state["confirm_appointment"]:
            response = random.choice(intents['intents'][4]['responses'])
            conversation_state["confirm_appointment"] = True
        elif tag == "goodbye" and conversation_state["confirm_appointment"]:
            response = random.choice(intents['intents'][5]['responses'])
            print(f"{bot_name}: {response}")
            break  # End conversation
        else:
                # Use GPT-2 to generate a fallback response
                gpt2_response = generate_gpt2_response(f"User: {sentence}\nBot:")
                response = gpt2_response
        print(f"{bot_name}: {response}")
    else:
        # If the model is not confident enough, use GPT-2 as fallback
        print(f"{bot_name}: I do not understand. Let me try to generate a response...")
        gpt2_response = generate_gpt2_response(f"User: {sentence}\nBot:")
        print(f"{bot_name}: {gpt2_response}")
