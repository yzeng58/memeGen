from transformers import MllamaForConditionalGeneration, AutoProcessor, pipeline
import torch, re
from huggingface_hub import login


import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import HUGGINGFACE_API_KEY
from helper import get_image, read_json


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
    history = None,
    save_history = False,
    description = '',
):
    if 'Llama-3.2' in llama['model_id']:
        model, processor = llama['model'], llama['processor']
        if description:
            images = [f"Meme {i+1}: {read_json(image_path)['description']}\n" for i, image_path in enumerate(image_paths)]
            begin_of_text = '<|begin_of_text|>' if history is None else ''
            prompt = "".join(images) + f"{begin_of_text}Question: {prompt}\nAnswer: "
        else:
            images = [get_image(image_path) for image_path in image_paths]
            begin_of_text = '<|begin_of_text|>' if history is None else ''
            prompt = f'<|image|>'*len(images) + f'{begin_of_text}Question: {prompt}\nAnswer: '
        if history: 
            prompt = history['prompt'] + '\n\n' + prompt
            images = history['images'] + images

        inputs = processor(images, prompt, return_tensors="pt").to(model.device)

        output_dict = {}

        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(output[0])
        response = re.sub(r'<\|.*?\|>', '', response)
        # Remove the prompt from the response
        response = response.replace(prompt, '').strip()

        output_dict['output'] = response
        if save_history: 
            output_dict['history'] = {
                'prompt': prompt + ' ' + response,
                'images': images,
            }
        
        return output_dict
    elif 'Llama-3.1' in llama['model_id']:
        llama_pipeline = llama['model']
        response = llama_pipeline(prompt, max_new_tokens=max_new_tokens)
        # Remove the prompt from the response
        response_without_prompt = response[0]['generated_text'].replace(prompt, '').strip()
        return response_without_prompt
