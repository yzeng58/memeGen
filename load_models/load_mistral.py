import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from configs import system_prompts
from helper import read_json, set_seed
import pdb

def load_mistral(
    model_path: str,
):
    model_name = model_path.split("/")[0]
    if model_path.endswith('/pretrained'):
        model_path = f"mistralai/{model_name}"
    else:
        model_path = f"{root_dir}/models/{model_path}"
    model_id = f"mistralai/{model_name}"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    mistral = {
        'model': model,
        'tokenizer': tokenizer,
    }

    return mistral

def process_sample_feature(
    image_paths,
):
    user_prompt = ""
    for i, image_path in enumerate(image_paths):
        idx_str = f" {i+1}" if len(image_paths) > 1 else ""
        user_prompt += f"Meme{idx_str}: {read_json(image_path)['description']['output']}\n"
    return user_prompt

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
    demonstrations = [],
    **kwargs,
):
    set_seed(seed)
    model, tokenizer = mistral['model'], mistral['tokenizer']

    if description == '': raise ValueError("Description is required for Mistral models!")

    if history:
        conversation = history
    else:
        conversation = [{"role": "system", "content": system_prompts['mistral'][system_prompt]}]

    if demonstrations:
        for sample_idx, sample in enumerate(demonstrations):
            user_prompt = process_sample_feature(sample['image_paths'])
            if sample_idx == 0: user_prompt = prompt + user_prompt
            conversation.append({"role": "user", "content": user_prompt})

            if not 'label' in sample:
                raise ValueError("Label is required for non-test samples!")
            conversation.append({"role": "assistant", "content": sample['label']})

    user_prompt = process_sample_feature(image_paths)
    if not demonstrations: user_prompt = user_prompt + prompt
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
    