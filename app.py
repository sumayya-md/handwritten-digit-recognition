
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import os
from tensorflow.keras.models import load_model
import datetime

# Paths
MODEL_PATH = os.path.join("models", "mnist_cnn.h5")
CUSTOM_DATA_DIR = "custom_digits"

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

# Preprocessing function (MNIST-style centering & padding)
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # white digit on black
    arr = np.array(img)

    # Find bounding box
    coords = np.column_stack(np.where(arr > 0))
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    arr = arr[y_min:y_max+1, x_min:x_max+1]

    # Make square and center
    size = max(arr.shape)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - arr.shape[0]) // 2
    x_off = (size - arr.shape[1]) // 2
    square[y_off:y_off+arr.shape[0], x_off:x_off+arr.shape[1]] = arr

    # Resize to 20x20, pad to 28x28
    img = Image.fromarray(square).resize((20, 20), Image.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (4, 4))

    # Normalize
    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# Input method selection
mode = st.radio("Choose input method:", ("Draw", "Upload"))

# DRAW MODE
if mode == "Draw":
    st.markdown(
        """
        <style>
        canvas {
            border: none !important;
        }
        div[data-testid="stCanvasToolbar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None and model:
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"), "RGB")
        x = preprocess_pil(img)
        preds = model.predict(x)
        digit = int(np.argmax(preds))
        conf = float(np.max(preds))
        st.success(f"Prediction: **{digit}** (Confidence: {conf:.2%})")

        # Save drawing button
        save_label = st.number_input("Label this drawing for retraining:", min_value=0, max_value=9, step=1, value=digit)
        if st.button("💾 Save Drawing"):
            save_path = os.path.join(CUSTOM_DATA_DIR, str(save_label))
            os.makedirs(save_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img.save(os.path.join(save_path, f"{timestamp}.png"))
            st.success(f"✅ Saved to {save_path}")

# UPLOAD MODE
else:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and model:
        img = Image.open(uploaded_file)
        x = preprocess_pil(img)
        preds = model.predict(x)
        digit = int(np.argmax(preds))
        conf = float(np.max(preds))
        st.success(f"Prediction: **{digit}** (Confidence: {conf:.2%})")

st.markdown("---")
st.write("💡 Tip: Draw clear, centered digits for best accuracy.")
