from typing import Literal

from environment import OPENAI_API_KEY
from openai import OpenAI
import os, base64, requests, sys
from typing import Literal
from time import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from helper import retry_if_fail, read_json
import pdb

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def process_image(
    image_input, 
    image_input_detail,
    image_mode: Literal['url', 'path'] = 'url',
):
    if image_mode == 'url':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_input,
                "detail": image_input_detail,
            },
        }
    elif image_mode == 'path':
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpg;base64,{encode_image(image_input)}",
                "detail": image_input_detail,
            },
        }
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {image_mode}.")
    
    return image_content

def process_text(text_input):
    text_content = {
        "type": "text",
        "text": text_input,
    }
    
    return text_content

def load_gpt(
    model: str = 'gpt-4o-mini',
    api_key: str = 'yz',
):
    client = OpenAI(api_key = OPENAI_API_KEY[api_key])
    return {
        'client': client,
        'model': model,
        'api_key': OPENAI_API_KEY[api_key],
    }

@retry_if_fail
def call_gpt(
    gpt,
    prompt,
    image_paths: list[str] = [],
    max_new_tokens: int = 200,
    history = None,
    save_history = False,
    seed: int = 42,
    image_input_detail: Literal['low', 'high'] = 'low',
    image_mode: Literal['url', 'path'] = 'path',
    description = '',
):
    model, client, api_key = gpt['model'], gpt['client'], gpt['api_key']
    contents = []
    for i, image_path in enumerate(image_paths):
        if description:
            contents.append(process_text(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
        else:
            contents.append(process_image(image_path, image_input_detail, image_mode))
    contents.append(process_text(prompt))
    messages = [{
        "role": "user",
        "content": contents,
    }]

    output_dict = {}

    if history is not None: messages = history + messages

    payload = {
        'model': model,
        'messages':messages,
        'max_tokens':max_new_tokens,
        'seed': seed,
    }

    if image_mode == 'url':
        response = client.chat.completions.create(**payload)
        output = response.choices[0].message.content
    elif image_mode == 'path':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload,
        )
        output = response.json()['choices'][0]['message']['content']
    else:
        raise ValueError("The image_mode must be either 'url' or 'path', not {mode}.")  

    if save_history: 
        output_dict['history'] = messages + [{
            "role": "assistant", 
            "content": output,
        }]

    output_dict['output'] = output
    return output_dict
