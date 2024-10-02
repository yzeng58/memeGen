from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch, re
from typing import Literal
from huggingface_hub import login


import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import huggingface_api_key
from helper import get_image
login(token = huggingface_api_key)


def load_llama(
    model_id: Literal[
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-90B-Vision",
        "meta-llama/Llama-3.1-405B",
    ] = "meta-llama/Llama-3.2-11B-Vision"
):

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    llama = {
        'model': model,
        'processor': processor,
    }
    return llama

def call_llama(
    llama, 
    prompt, 
    image_paths: list[str] = [],
    max_new_tokens: int = 200,
):
    model, processor = llama['model'], llama['processor']
    images = [get_image(image_path) for image_path in image_paths]
    inputs = processor(images, f'<|image|>'*len(images) + f'<|begin_of_text|>{prompt}', return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.decode(output[0])
    cleaned_response = re.sub(r'<\|.*?\|>', '', response)
    return cleaned_response
