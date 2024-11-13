from typing import Literal

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from environment import CLAUDE_API_KEY
import anthropic, base64
from PIL import Image
from helper import retry_if_fail, read_json

def ensure_jpeg(image_path):
    try:
        with Image.open(image_path) as img:
            if img.format != 'JPEG':
                img = img.convert('RGB')  # Convert to RGB if not
                
                new_image_path = image_path.replace('.jpg', '.jpeg').replace('datasets', 'archive/datasets_jpeg')
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok = True)
                
                img.save(new_image_path, 'JPEG')  # Save as JPEG
                image_path = new_image_path
    except IOError:
        print("Unable to open the image. Ensure the file is a valid image.")
        
    return image_path

# Function to encode the image
def encode_image(image_path):
    image_path = ensure_jpeg(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image(image_input):
    image_content = {
        'type': 'image',
        'source': {
            'type': 'base64',
            'media_type': 'image/jpeg',
            'data': encode_image(image_input),
        }
    }
    return image_content

def process_text(text_input):
    if text_input: 
        text_content = {
            "type": "text",
            "text": text_input,
        }
    else:
        text_content = None
    return text_content

def load_claude(
    model: str = "claude-3-haiku-20240307",
    api_key: str = 'yz',
):
    client = anthropic.Anthropic(
        api_key = CLAUDE_API_KEY[api_key],
    )
    return {
        'client': client,
        'model': model,
    }

def call_claude(
    claude, 
    prompt, 
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    description = '',
    context = "",
    temperature = 0,
    max_new_tokens = 1000,
    **kwargs,
):
    model, client = claude['model'], claude['client']

    # make messages
    contents = []
    for i, image_path in enumerate(image_paths):
        if description:
            contents.append(process_text(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
        elif context:
            contents.append(process_text(f"Meme {i+1}: {read_json(image_path['description_path'])['description']}\n"))
            contents.append(process_image(image_path['image_path']))
        else:
            contents.append(process_image(image_path))
            
    contents.append(process_text(prompt))
    messages = [{
        "role": "user",
        "content": contents,
    }]

    output_dict = {}

    if history is not None: messages = history + messages

    response = client.messages.create(
        model = model,
        messages = messages,
        max_tokens = max_new_tokens,
        temperature = temperature,
    )
    output = response.content[0].text


    if save_history: 
        output_dict['history'] = messages + [{
            "role": "assistant", 
            "content": output,
        }]

    output_dict['output'] = output
    return output_dict


