import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps

# ========================
# CONFIG
# ========================
CUSTOM_DATA_DIR = "custom_digits"  # Folder where your digit images go
MODEL_SAVE_PATH = "models/mnist_cnn.h5"
EPOCHS = 5
BATCH_SIZE = 128
# ========================

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# ========================
# Load MNIST
# ========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ========================
# Load Custom Drawings
# Expected folder structure:
# custom_digits/
#     0/
#         img1.png
#         img2.png
#     1/
#         img1.png
#     ...
#     9/
#         imgX.png
# ========================
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # white digit on black
    arr = np.array(img)

    # Find bounding box of the digit
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

    # Resize to 20x20, pad to 28x28 (MNIST format)
    img = Image.fromarray(square).resize((20, 20), Image.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (4, 4))

    # Normalize
    arr = np.array(new_img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ========================
# Build Model
# ========================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========================
# Train Model
# ========================
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"✅ Model trained and saved at {MODEL_SAVE_PATH}")
