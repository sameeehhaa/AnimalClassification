import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# --------------------------
# Load the trained model
# --------------------------
model = tf.keras.models.load_model("animal_classifier_model.h5")

# Define class names (same order as during training)
class_names = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
    "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

# --------------------------
# Streamlit Web Interface
# --------------------------
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🐾 Animal Image Classifier")
st.markdown("Upload an animal image and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)


    # Preprocess image
    img = image_pil.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Animal: **{predicted_class}**")
