import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Download punkt (needed for tokenizing sentences)
nltk.download("punkt")

# Creating instance of PorterStemmer
stems = PorterStemmer()

# Tokenize a sentence (split it into words/tokens)
def tokenize(sentence):
    # Using NLTK's word_tokenize to split sentence into tokens
    return nltk.word_tokenize(sentence)
    """
    Tokenizes a sentence into an array of words/tokens.
    A token can be a word, punctuation, or number.
    Args: sentence (str): The sentence that needs to be tokenized.
    Returns: list: A list of tokens (words, punctuation, etc.)
    """

# Perform stemming (find the root form of a word)
def stem(word):
    # Convert word to lowercase and stem it using PorterStemmer
    return stems.stem(word.lower())
    """
    Stemming process: Convert a word into its root form.
    Args: word (str): The word to be stemmed.
    Returns: str: The stemmed (root) form of the word.
    """

# Create a "bag of words" from a tokenized sentence
def bag_of_words(tokenized_sentence, words):
    """
    Creates a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Args: tokenized_sentence (list): List of tokens (words) from the sentence.
    words (list): List of all known words (vocabulary).
    Returns: np.ndarray: Array representing the presence (1) or absence (0) of known words in the sentence.
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Stem each word in the tokenized sentence
    # Create a list of stemmed words
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Init 'bag' (array) with zeros, the same length as the list of known words
    bag = np.zeros(len(words), dtype=np.float32)
    # Loop through each known word, check if it's in the stemmed tokenized sentence
    for idx, w in enumerate(words):
        # If the word exists in the tokenized sentence, set the corresponding index in the 'bag' to 1
        if w in sentence_words:
            bag[idx] = 1
    # Return the final bag of words array (binary representation)
    return bag
