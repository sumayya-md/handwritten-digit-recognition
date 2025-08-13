import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import os
from tensorflow.keras.models import load_model

# Path to saved model
MODEL_PATH = os.path.join("models", "mnist_cnn.h5")

# Load the model
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = None
if os.path.exists(MODEL_PATH):
    model = load_my_model()
else:
    st.error("❌ Model not found! Run train.py first to generate mnist_cnn.h5")

st.title("✏️ Handwritten Digit Recognition")
st.write("Draw a digit or upload an image to recognize it.")

# Radio buttons for input method
mode = st.radio("Choose input method:", ("Draw", "Upload"))

# Preprocessing function
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img)
    # Invert if background is white
    if np.mean(arr) > 127:
        arr = 255 - arr
    arr = arr.astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# Draw mode
if mode == "Draw":
    st.write("🖌 Draw your digit below (black on white background).")
    canvas_result = st_canvas(
        fill_color="",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = img.convert("RGB")
        st.image(img, caption="Your Drawing", width=150)

        if model:
            x = preprocess_pil(img)
            preds = model.predict(x)
            digit = np.argmax(preds)
            conf = float(np.max(preds))
            st.success(f"Prediction: **{digit}** (Confidence: {conf:.2%})")

# Upload mode
else:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=150)

        if model:
            x = preprocess_pil(img)
            preds = model.predict(x)
            digit = np.argmax(preds)
            conf = float(np.max(preds))
            st.success(f"Prediction: **{digit}** (Confidence: {conf:.2%})")

st.markdown("---")
st.write("💡 Tip: Draw clear, centered digits for best accuracy.")
