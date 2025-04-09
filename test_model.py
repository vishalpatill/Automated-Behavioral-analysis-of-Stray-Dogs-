import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("dog_behavior_model.h5")

# Define the class labels in the same order as training
class_labels = ['angry', 'happy', 'relaxed', 'sad']

# Path to the test images folder
test_folder = "test_images"

# Loop through all files in the folder
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_folder, filename)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to training input shape
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize like training data

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        print(f"{filename} ➤ Predicted: {predicted_class}")
