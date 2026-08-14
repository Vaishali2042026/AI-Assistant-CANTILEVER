import os
import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    Dropout,
    Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence


# ==========================================
# 1. File paths
# ==========================================

FEATURE_FILE = "image_captioning/models/image_features.pkl"
CAPTION_FILE = "image_captioning/dataset/captions.txt"
MODEL_FOLDER = "image_captioning/models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


# ==========================================
# 2. Load image features
# ==========================================

print("Loading image features...")

with open(FEATURE_FILE, "rb") as file:
    features = pickle.load(file)

print(f"Loaded features for {len(features)} images")


# ==========================================
# 3. Load captions
# ==========================================

print("Loading captions...")

captions = {}

with open(CAPTION_FILE, "r", encoding="utf-8") as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        # Skip header
        if line.lower().startswith("image"):
            continue

        parts = line.split(",", 1)

        if len(parts) != 2:
            continue

        image_name = parts[0].split("#")[0]

        caption = parts[1].strip().lower()

        caption = "startseq " + caption + " endseq"

        if image_name not in captions:
            captions[image_name] = []

        captions[image_name].append(caption)


print(f"Images with captions: {len(captions)}")


# ==========================================
# 4. Match captions with image features
# ==========================================

valid_captions = {}

for image_name in captions:

    if image_name in features:

        valid_captions[image_name] = captions[image_name]


print(
    f"Images available for training: "
    f"{len(valid_captions)}"
)


# ==========================================
# 5. Create tokenizer
# ==========================================

all_captions = []

for image_name in valid_captions:

    all_captions.extend(
        valid_captions[image_name]
    )


print("Creating tokenizer...")

tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<unk>"
)

tokenizer.fit_on_texts(all_captions)

vocab_size = min(
    5000,
    len(tokenizer.word_index) + 1
)

print(f"Vocabulary size: {vocab_size}")


# ==========================================
# 6. Maximum caption length
# ==========================================

max_length = max(
    len(caption.split())
    for caption in all_captions
)

print(f"Maximum caption length: {max_length}")


# ==========================================
# 7. Create training samples
# ==========================================

training_samples = []

print("Creating training sample index...")


for image_name in valid_captions:

    feature = np.asarray(
        features[image_name],
        dtype=np.float32
    ).reshape(-1)

    # Make sure image feature has 2048 values
    if feature.shape[0] != 2048:

        print(
            f"Skipping {image_name}: "
            f"feature size = {feature.shape[0]}"
        )

        continue

    for caption in valid_captions[image_name]:

        sequence = tokenizer.texts_to_sequences(
            [caption]
        )[0]

        for i in range(1, len(sequence)):

            input_sequence = sequence[:i]

            output_word = sequence[i]

            training_samples.append(
                (
                    feature,
                    input_sequence,
                    output_word
                )
            )


print(
    f"Training samples: "
    f"{len(training_samples)}"
)


# ==========================================
# 8. Memory-efficient generator
# ==========================================

class CaptionDataGenerator(Sequence):

    def __init__(
        self,
        samples,
        max_length,
        batch_size=32,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.samples = samples
        self.max_length = max_length
        self.batch_size = batch_size

    def __len__(self):

        return int(
            np.ceil(
                len(self.samples)
                / self.batch_size
            )
        )

    def __getitem__(self, index):

        batch = self.samples[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        X_image = []
        X_sequence = []
        y = []

        for feature, sequence, output_word in batch:

            padded_sequence = pad_sequences(
                [sequence],
                maxlen=self.max_length,
                padding="post"
            )[0]

            X_image.append(feature)

            X_sequence.append(
                padded_sequence
            )

            y.append(output_word)

        return (
            (
                np.asarray(
                    X_image,
                    dtype=np.float32
                ),

                np.asarray(
                    X_sequence,
                    dtype=np.int32
                )
            ),

            np.asarray(
                y,
                dtype=np.int32
            )
        )


# ==========================================
# 9. Create generator
# ==========================================

print("Creating data generator...")

train_generator = CaptionDataGenerator(
    training_samples,
    max_length,
    batch_size=32
)

print(
    f"Training batches: "
    f"{len(train_generator)}"
)


# ==========================================
# 10. Build improved CNN-LSTM model
# ==========================================

print("Building improved CNN-LSTM model...")


# ------------------------------------------
# Image branch
# ------------------------------------------

image_input = Input(
    shape=(2048,),
    name="image_features"
)

image_dense = Dense(
    512,
    activation="relu"
)(image_input)

image_dense = Dropout(
    0.4
)(image_dense)


image_dense = Dense(
    256,
    activation="relu"
)(image_dense)


# ------------------------------------------
# Caption branch
# ------------------------------------------

caption_input = Input(
    shape=(max_length,),
    name="caption_input"
)

caption_embedding = Embedding(
    input_dim=vocab_size,
    output_dim=256,
    mask_zero=True
)(caption_input)


caption_lstm = LSTM(
    256,
    dropout=0.3,
    recurrent_dropout=0.2
)(caption_embedding)


# ------------------------------------------
# Combine image + language information
# ------------------------------------------

merged = Concatenate()([
    image_dense,
    caption_lstm
])


merged = Dense(
    512,
    activation="relu"
)(merged)

merged = Dropout(
    0.4
)(merged)


merged = Dense(
    256,
    activation="relu"
)(merged)


merged = Dropout(
    0.3
)(merged)


# ------------------------------------------
# Predict next word
# ------------------------------------------

output = Dense(
    vocab_size,
    activation="softmax"
)(merged)


# ------------------------------------------
# Create model
# ------------------------------------------

model = Model(
    inputs=[
        image_input,
        caption_input
    ],
    outputs=output
)


# ==========================================
# 11. Compile model
# ==========================================

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


model.summary()


# ==========================================
# 12. Callbacks
# ==========================================

checkpoint = ModelCheckpoint(
    "image_captioning/models/image_captioning_lstm.keras",
    monitor="loss",
    save_best_only=True,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ==========================================
# 13. Train model
# ==========================================

print("\n========================================")
print("STARTING IMPROVED LSTM TRAINING")
print("========================================")

print("Epochs: 15")
print("Batch size: 32")
print("Images: 8091")
print("Using existing InceptionV3 features")
print("========================================\n")


model.fit(
    train_generator,
    epochs=15,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)


# ==========================================
# 14. Save tokenizer
# ==========================================

with open(
    "image_captioning/models/tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


# ==========================================
# 15. Save model information
# ==========================================

with open(
    "image_captioning/models/model_info.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "max_length": max_length,
            "vocab_size": vocab_size,
            "feature_size": 2048
        },
        file
    )


# ==========================================
# 16. Finished
# ==========================================

print("\n========================================")
print("IMPROVED LSTM TRAINING COMPLETED!")
print("========================================")

print(
    "Model saved to:"
)

print(
    "image_captioning/models/"
    "image_captioning_lstm.keras"
)

print("Tokenizer saved successfully!")

print("========================================")