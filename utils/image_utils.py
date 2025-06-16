from PIL import Image
import os

def load_image_from_path(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist.")
    return Image.open(image_path).convert("RGB")
