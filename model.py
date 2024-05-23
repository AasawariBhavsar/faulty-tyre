import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def prepare_image(image, target_size):
    image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0
