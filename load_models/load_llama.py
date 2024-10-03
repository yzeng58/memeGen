from transformers import MllamaForConditionalGeneration, AutoProcessor, pipeline
import torch, re
from typing import Literal
from huggingface_hub import login
from configs import support_models


import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import HUGGINGFACE_API_KEY
from helper import get_image


def load_llama(
    model: str = "Llama-3.2-11B-Vision",
    api_key: str = 'yz',
):
    login(token = HUGGINGFACE_API_KEY[api_key])
    model = f'meta-llama/{model}'

    if 'Llama-3.2' in model:
        llama_model = MllamaForConditionalGeneration.from_pretrained(
            model,
        torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        llama_processor = AutoProcessor.from_pretrained(model)
        llama = {
            'model_id': model,
            'model': llama_model,
            'processor': llama_processor,
        }
    elif 'Llama-3.1' in model:
        llama_pipeline = pipeline(
            "text-generation", 
            model=model, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto"
        )
        llama = {
            'model_id': model,
            'model': llama_pipeline,
        }

    return llama

def call_llama(
    llama, 
    prompt, 
    image_paths: list[str] = [],
    max_new_tokens: int = 200,
):
    if 'Llama-3.2' in llama['model_id']:
        model, processor = llama['model'], llama['processor']
        images = [get_image(image_path) for image_path in image_paths]
        inputs = processor(images, f'<|image|>'*len(images) + f'<|begin_of_text|>{prompt}', return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(output[0])
        cleaned_response = re.sub(r'<\|.*?\|>', '', response)
        return cleaned_response
    elif 'Llama-3.1' in llama['model_id']:
        llama_pipeline = llama['model']
        response = llama_pipeline(prompt, max_new_tokens=max_new_tokens)
        return response
