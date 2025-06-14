import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os

# Load all models (assuming files are uploaded to the Hugging Face Space manually)
model_paths = {
    "ResNet50": "./resnet50_skin_cancer_model.h5",
    "MobileNetV2": "./mobilenetv2_ham10000_model.h5",
    "EfficientNetB0": "./efficientnetb0_skin_cancer_model.h5",
    "InceptionV3": "./inceptionv3_skin_cancer_model.h5",
    "ConvNeXtTiny": "./convnext_tiny_notop.h5"
}

models = {name: load_model(path) for name, path in model_paths.items()}

# Load condition descriptions
with open("prompts/condition_info.json", "r") as f:
    condition_info = json.load(f)

def preprocess(img, target_size=(224, 224)):
    img = img.astype("float32") / 255.0
    img = tf.image.resize(img, target_size)
    img = img_to_array(img)
    return np.expand_dims(img, axis=0)

def predict_condition(img):
    preprocessed = preprocess(img)
    results = {}

    for name, model in models.items():
        preds = model.predict(preprocessed)
        class_index = np.argmax(preds)
        confidence = float(np.max(preds))
        label = list(condition_info.keys())[class_index]
        desc = condition_info.get(label, "No description available.")
        results[name] = f"{label} ({confidence:.2f} confidence):\n{desc}\n"

    output = "\n".join([f"\u2728 **{model}**:\n{info}" for model, info in results.items()])
    return output
