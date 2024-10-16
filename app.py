import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained model (ensure it's properly compiled)
def load_model():
    model = tf.keras.models.load_model('brain_stroke_model.keras')
    return model

# Function to preprocess the image step-by-step
def preprocess_image(image):
    # Stage 1: Grayscale Conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Stage 2: Blurring (Gaussian Blur)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Stage 3: Edge Detection (Canny)
    edges_image = cv2.Canny(blurred_image, 100, 200)

    return gray_image, blurred_image, edges_image

# Function to make predictions
def predict(image):
    # Preprocess the image as needed
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0  # Rescale image
    prediction = model.predict(image)
    return prediction

# Load model
model = load_model()

# Streamlit logic
st.title('Brain Stroke Detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image step-by-step
    gray_image, blurred_image, edges_image = preprocess_image(image)

    # Display the process in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image, channels="BGR", caption="Original Image")

    with col2:
        st.image(gray_image, caption="Grayscale Image", use_column_width=True)

    with col3:
        st.image(blurred_image, caption="Blurred Image", use_column_width=True)

    with col4:
        st.image(edges_image, caption="Edge Detection", use_column_width=True)

    # Final Prediction (using original image processed for the model)
    prediction = predict(image)
    st.write(f"Prediction: {'Stroke' if prediction[0] > 0.5 else 'No Stroke'}")
    st.write(prediction[0])