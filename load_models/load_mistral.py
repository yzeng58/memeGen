import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from configs import system_prompts
from helper import read_json, set_seed
import pdb

def load_mistral(
    model_name: str,
):
    model_id = f"mistralai/{model_name}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    mistral = {
        'model': model,
        'tokenizer': tokenizer,
    }

    return mistral

def call_mistral(
    mistral,
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    system_prompt = 'evaluator',
    description = '',
    max_new_tokens = 500,
    seed = 42,  
    temperature = 0.1,
    **kwargs,
):
    set_seed(seed)
    model, tokenizer = mistral['model'], mistral['tokenizer']

    if description == '': raise ValueError("Description is required for Mistral models!")

    if history:
        conversation = history
    else:
        conversation = [{"role": "system", "content": system_prompts['mistral'][system_prompt]}]

    user_prompt = ""
    for i, image_path in enumerate(image_paths):
        user_prompt += f"Meme {i+1}: {read_json(image_path)['description']}\n"
    user_prompt += prompt
    conversation.append({"role": "user", "content": user_prompt})
    
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens+2,
        temperature=temperature,
    )

    input_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):].strip()

    output_dict = {}

    if save_history: 
        conversation.append({"role": "assistant", "content": output_text})
        output_dict['history'] = conversation

    output_dict['output'] = output_text

    return output_dict
    