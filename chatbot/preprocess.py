import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize

# Download tokenizer data
nltk.download("punkt")

# Load our conversation dataset
data = pd.read_csv("data/conversation_dataset.csv")

# Combine user messages and bot responses
texts = data["user_message"].tolist() + data["bot_response"].tolist()


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text


# Clean and tokenize the conversations
tokenized_texts = []

for text in texts:
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    tokenized_texts.append(tokens)


# Display some examples
for i in range(5):
    print("Original:", texts[i])
    print("Tokens:", tokenized_texts[i])
    print()