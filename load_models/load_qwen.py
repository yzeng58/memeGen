# Update qwen/finetune.py

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from configs import system_prompts
from helper import read_json
def load_qwen(
    model_name: str,
):  
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}", device_map='auto', trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
    qwen = {
        'model': model,
        'tokenizer': tokenizer,
    }
    return qwen

def process_text(text_input):
    return {'text': text_input}

def process_image(image_path):
    return {'image': image_path}

    
def call_qwen(
    qwen, 
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    system_prompt = 'evaluator',
    description = '',
    **kwargs,
):
    model, tokenizer = qwen['model'], qwen['tokenizer']
    
    # make messages
    contents = []
    for i, image_path in enumerate(image_paths):
        if description:
            contents.append(process_text(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
        else:
            contents.append(process_image(image_path))
    contents.append(process_text(prompt))
    
    query = tokenizer.from_list_format(contents)
    output_dict = {}
    output_dict['output'], output_dict['history'] = model.chat(
        tokenizer, 
        query=query, 
        history=history,
        system = system_prompts['qwen'][system_prompt],
    )

    if not save_history: output_dict.pop('history')

    return output_dict