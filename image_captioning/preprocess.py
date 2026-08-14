import os
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Dataset location
IMAGE_FOLDER = "image_captioning/dataset/images"

# Load pretrained InceptionV3
model = InceptionV3(weights="imagenet")

# Remove final classification layer
model = Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

# Store image features
features = {}

print("Extracting image features...")

for filename in os.listdir(IMAGE_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(IMAGE_FOLDER, filename)

        image = load_img(image_path, target_size=(299, 299))

        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        features[filename] = feature

print("Image feature extraction completed!")

# Save features
os.makedirs("image_captioning/models", exist_ok=True)

with open("image_captioning/models/image_features.pkl", "wb") as file:
    pickle.dump(features, file)

print("Image features saved successfully!")