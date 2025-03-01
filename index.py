import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from convokit import Corpus, download

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load Cornell Movie Dialogs Corpus using ConvoKit
cornell_corpus = Corpus(download('cornell_movie_dialogs_corpus'))
cornell_data = pd.DataFrame([utt.text for utt in cornell_corpus.iter_utterances()], columns=['dialogue'])

# Load Persona-Chat Dataset using ConvoKit
persona_corpus = Corpus(download('persona_chat'))
persona_data = pd.DataFrame([utt.text for utt in persona_corpus.iter_utterances()], columns=['text'])

# Save raw datasets
cornell_data.to_csv('data/raw/cornell_movie_dialogs_corpus.csv', index=False)
persona_data.to_csv('data/raw/persona_chat.csv', index=False)

# Tokenization
nltk.download('punkt')
cornell_data['tokens'] = cornell_data['dialogue'].apply(word_tokenize)
persona_data['tokens'] = persona_data['text'].apply(word_tokenize)

# Cleaning and Normalization
cornell_data['tokens'] = cornell_data['tokens'].apply(lambda x: [word.lower() for word in x if word.isalnum()])
persona_data['tokens'] = persona_data['tokens'].apply(lambda x: [word.lower() for word in x if word.isalnum()])

# Splitting (example for Cornell data)
train_data = cornell_data.sample(frac=0.8, random_state=42)
test_data = cornell_data.drop(train_data.index)

# Save preprocessed data
train_data.to_csv('data/processed/train_data.csv', index=False)
test_data.to_csv('data/processed/test_data.csv', index=False)