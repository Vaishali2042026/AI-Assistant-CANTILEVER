import pickle
import re
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("data/conversation_dataset.csv")

data = data.dropna()

user_messages = data["user_message"].astype(str).tolist()
bot_responses = data["bot_response"].astype(str).tolist()


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


normalized_messages = [
    normalize_text(message)
    for message in user_messages
]


# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = Tokenizer(
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(normalized_messages)

sequences = tokenizer.texts_to_sequences(normalized_messages)

max_length = max(len(sequence) for sequence in sequences)

input_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post"
)


# ============================================================
# 4. WORD2VEC
# ============================================================

sentences = [
    message.split()
    for message in normalized_messages
]

word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)

print("Word2Vec model trained successfully!")


# ============================================================
# 5. CREATE EMBEDDING MATRIX
# ============================================================

vocab_size = len(tokenizer.word_index) + 1

embedding_matrix = np.zeros(
    (vocab_size, 100)
)

for word, index in tokenizer.word_index.items():

    if word in word2vec_model.wv:

        embedding_matrix[index] = (
            word2vec_model.wv[word]
        )


# ============================================================
# 6. CREATE LABELS
# ============================================================

# Each CSV question has its own response.
labels = np.arange(len(normalized_messages))

number_of_classes = len(labels)

target_data = to_categorical(
    labels,
    num_classes=number_of_classes
)


# ============================================================
# 7. BUILD LSTM CLASSIFIER
# ============================================================

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=100,
        weights=[embedding_matrix],
        trainable=True
    )
)

model.add(
    LSTM(128)
)

model.add(
    Dense(
        number_of_classes,
        activation="softmax"
    )
)


# ============================================================
# 8. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. TRAIN
# ============================================================

print("\nTraining LSTM chatbot...\n")

model.fit(
    input_sequences,
    target_data,
    epochs=150,
    batch_size=8,
    verbose=1
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

model.save(
    "chatbot/chatbot_model.keras"
)

with open(
    "chatbot/tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


with open(
    "chatbot/word2vec.pkl",
    "wb"
) as file:

    pickle.dump(
        word2vec_model,
        file
    )


# ============================================================
# 11. SAVE CHATBOT INFORMATION
# ============================================================

with open(
    "chatbot/model_info.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "max_length": max_length,
            "vocab_size": vocab_size,
            "number_of_classes": number_of_classes,
            "user_messages": user_messages,
            "normalized_messages": normalized_messages,
            "bot_responses": bot_responses
        },
        file
    )


print("\n===================================")
print("CHATBOT TRAINING COMPLETE!")
print("===================================")
print("Word2Vec: YES")
print("LSTM: YES")
print("Responses: CSV")
print("Model saved successfully!")