from PIL import Image
import requests
from io import BytesIO

def get_image(image_path: str):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def display_image(image_path: str):
    image = get_image(image_path)
    display(image)