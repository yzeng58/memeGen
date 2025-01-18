from transformers import MllamaForConditionalGeneration, AutoProcessor, pipeline
import torch, re
from huggingface_hub import login


import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import HUGGINGFACE_API_KEY
from helper import get_image, read_json, set_seed, retry_if_fail
from configs import system_prompts
import pdb

def load_llama(
    model_path: str = "Llama-3.2-11B-Vision",
    api_key: str = 'yz',
):
    login(token = HUGGINGFACE_API_KEY[api_key])
    model_name = model_path.split("/")[0]
    if model_path.endswith('/pretrained'):
        model_path = f'meta-llama/{model_name}'
    else:
        model_path = f"{root_dir}/models/{model_path}"
    model_id = f'meta-llama/{model_name}'

    if 'Llama-3.2' in model_name:
        llama_model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        llama_processor = AutoProcessor.from_pretrained(model_id)
        llama = {
            'model_id': model_id,
            'model': llama_model,
            'processor': llama_processor,
        }
    elif 'Llama-3.1' in model_name:
        llama_pipeline = pipeline(
            "text-generation", 
            model=model_path, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto"
        )
        llama = {
            'model_id': model_id,
            'model': llama_pipeline,
        }
    llama['model_name'] = model_name
    return llama

def process_sample_feature(
    image_paths,
    context,
    llama,
):
    if 'Llama-3.2' in llama['model_id']:
        content, images = [], []
        if context:
            for i, image_path in enumerate(image_paths):
                idx_str = f" {i+1}" if len(image_paths) > 1 else ""
                content.append({"type": "text", "text": f"Meme{idx_str}: {read_json(image_path['description_path'])['description']['output']}\n"})
                content.append({"type": "image"})
                images.append(get_image(image_path['image_path']))
        else:
            for i, image_path in enumerate(image_paths):
                images.append(get_image(image_path))
                content.append({"type": "image"})
        return content, images
    elif 'Llama-3.1' in llama['model_id']:
        text_prompt = ''
        for i, image_path in enumerate(image_paths):
            idx_str = f" {i+1}" if len(image_paths) > 1 else ""
            text_prompt += f"Meme{idx_str}: {read_json(image_path)['description']['output']}\n"
        return text_prompt


def call_llama(
    llama, 
    prompt, 
    image_paths: list[str] = [],
    max_new_tokens: int = 200,
    history = None,
    save_history = False,
    description = '',
    system_prompt = 'evaluator',
    seed = 42,
    context = "",
    demonstrations = [],
    **kwargs,
):
    set_seed(seed)
    if 'Llama-3.2' in llama['model_id']:
        if description: raise ValueError('Description is not supported for Llama-3.2 models.')

        model, processor = llama['model'], llama['processor']
        if history: 
            messages = history['messages']
            images = history['images']
        else:
            messages, images = [], []

        if demonstrations:
            messages.append({"role": "user", "content": {"type": "text", "text": prompt}})
            for sample in demonstrations:
                content_idx, images_idx = process_sample_feature(sample['image_paths'], context, llama)
                images.extend(images_idx)
                messages.append({"role": "user", "content": content_idx})

                if not 'label' in sample:
                    raise ValueError("Label is required for non-test samples!")
                messages.append({"role": "assistant", "content": sample['label']})

        content_idx, images_idx = process_sample_feature(image_paths, context, llama)
        messages.append({"role": "user", "content": content_idx})
        images.extend(images_idx)
    
        if not demonstrations: messages.append({"role": "user", "content": {"type": "text", "text": prompt}})
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = processor.decode(output[0])[len(input_text):]

        output_dict = {}
        if save_history: 
            messages.append({"role": "assistant", "content": output})
            output_dict['history'] = {
                'messages': messages,
                'images': images,
            }
        output_dict['output'] = output
        return output_dict
    
    elif 'Llama-3.1' in llama['model_id']:
        pipeline = llama['model']
        if description == '':
            raise ValueError('Description is required for Llama-3.1 since it is text-only model.')
        
        if history:
            messages = history['messages']
        else:
            messages = [{"role": "system", "content": system_prompts[llama['model_name']][system_prompt]}]

        if demonstrations:
            messages.append({"role": "user", "content": prompt})
            for sample in demonstrations:
                text_prompt = process_sample_feature(sample['image_paths'], context, llama)
                messages.append({"role": "user", "content": text_prompt})

                if not 'label' in sample:
                    raise ValueError("Label is required for non-test samples!")
                messages.append({"role": "assistant", "content": sample['label']})
        
        text_prompt = process_sample_feature(image_paths, context, llama)
        if not demonstrations: text_prompt += prompt
        messages.append({"role": "user", "content": text_prompt})

        outputs = pipeline(messages, max_new_tokens=max_new_tokens)
        output = outputs[0]['generated_text']

        output_dict = {}
        if save_history: 
            output_dict['history'] = {
                'messages': output,
            }
        output_dict['output'] = output[-1]['content']
        return output_dict