import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

MODEL_PATH = "models/plant_disease_mobilenetv2.keras"
IMG_SIZE = (160, 160)

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("Plant Disease Detection")
st.write(
    "Upload an image of a plant leaf to predict its disease."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.success("Model loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize(IMG_SIZE)

    # Convert image to array
    image_array = np.array(image_resized)

    # Add batch dimension
    input_image = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(input_image, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.subheader("Prediction")

    st.write("**Disease:**", predicted_class)
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Top 5 predictions
    top_5_indices = np.argsort(prediction[0])[-5:][::-1]

    st.subheader("Top 5 Predictions")

    for rank, index in enumerate(top_5_indices, start=1):
        class_name = class_names[index]
        probability = prediction[0][index] * 100

        st.write(
            f"{rank}. **{class_name}** — {probability:.2f}%"
        )
        
        
