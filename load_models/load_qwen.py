import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from configs import system_prompts
from helper import read_json, set_seed
from qwen_vl_utils import process_vision_info
import pdb

def load_qwen(
    model_path: str,
):  
    model_name = model_path.split("/")[0]
    if model_path.endswith('/pretrained'):
        model_path = f"Qwen/{model_name}"
    else:
        model_path = f"{root_dir}/models/{model_path}"
        
    if 'qwen2.5' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            f"Qwen/{model_name}",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")
        qwen = {
            'model': model,
            'tokenizer': tokenizer,
            'type': 'qwen2.5',
        }
    elif 'qwen2-vl' in model_name.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")
        qwen = {
            'model': model,
            'processor': processor,
            'type': 'qwen2-vl',
            "model_name": model_name,
        }
    else:
        raise ValueError(f"Model {model_name} not found")
    return qwen

def process_text_qwen(text_input):
    return {'text': text_input}

def process_image_qwen(image_path):
    return {'image': image_path}

def process_text_qwen2(text_input):
    return {
        "type": "text",
        "text": text_input,
    }

def process_image_qwen2(image_path):
    return {
        "type": "image",
        "image": image_path,
    }

def process_sample_feature(
    image_paths,
    qwen,
    description,
    context,
):
    if qwen['type'] in ['qwen2-vl']:
        contents = []
        for i, image_path in enumerate(image_paths):
            idx_str = f" {i+1}" if len(image_paths) > 1 else ""
            if description:
                contents.append(process_text_qwen2(f"Meme{idx_str}: {read_json(image_path)['description']}\n"))
            elif context:
                contents.append(process_text_qwen2(f"Meme{idx_str}: {read_json(image_path)['description']}\n"))
                contents.append(process_image_qwen2(image_path))
            else:
                contents.append(process_image_qwen2(image_path))
        return contents
    elif qwen['type'] in ['qwen2.5']:
        user_prompt = ""
        for i, image_path in enumerate(image_paths):
            idx_str = f" {i+1}" if len(image_paths) > 1 else ""
            user_prompt += f"Meme{idx_str}: {read_json(image_path)['description']}\n"
        return user_prompt
    
def call_qwen(
    qwen, 
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    system_prompt = 'evaluator',
    description = '',
    max_new_tokens = 500,
    seed = 42,  
    temperature = 0.1,
    context = "",
    demonstrations = None,
    **kwargs,
):
    set_seed(seed)

    if qwen['type'] in ['qwen2-vl']:
        model, processor, model_name = qwen['model'], qwen['processor'], qwen['model_name']
        
        if history:
            messages = history
        else:
            messages = [{"role": "system", "content": system_prompts[model_name][system_prompt]}]

        if demonstrations:
            messages.append({"role": "user", "content": [process_text_qwen2(prompt)]})
            for sample in demonstrations:
                contents = []
                image_paths = sample['image_paths']
                contents.extend(process_sample_feature(
                    image_paths=image_paths, 
                    qwen=qwen,
                    description=description,
                    context=context,
                ))
                
                messages.append({"role": "user", "content": contents})

                if not 'label' in sample:
                    raise ValueError("Label is required for non-test samples!")
                messages.append({"role": "assistant", "content": [process_text_qwen2(sample['label'])]})

        contents = process_sample_feature(
            image_paths=image_paths, 
            qwen=qwen, 
            description=description,
            context=context,
        )
        if not demonstrations: contents.append(process_text_qwen2(prompt))
        messages.append({"role": "user", "content": contents})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_timestamp=True) + 'assistant\n'
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text = [text],
            images = image_inputs,
            videos = video_inputs,
            return_tensors = 'pt',
            padding = True,
        ).to(model.device)
        output_dict = {}
        
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens = max_new_tokens+10,
            temperature = temperature,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_dict['output'] = output_texts[0].replace('system\n', '').replace('assistant: ', '').replace('assistant\n', '')

        if save_history:
            messages.append({"role": "assistant", "content": output_dict['output']})
            output_dict['history'] = messages
            
    elif qwen['type'] in ['qwen2.5']:
        if description == "" and image_paths: raise ValueError("Description is required for qwen2.5 series model!")

        model, tokenizer = qwen['model'], qwen['tokenizer']
        if history:
            messages = history
        else:
            messages = [{"role": "system", "content": system_prompts['qwen'][system_prompt]}]

        if demonstrations:
            messages.append({"role": "user", "content": prompt})
            for sample in demonstrations:
                image_paths = sample['image_paths']
                user_prompt = process_sample_feature(
                    image_paths=image_paths, 
                    qwen=qwen,
                    description=description,
                    context=context,
                )
                messages.append({"role": "user", "content": user_prompt})

                if not 'label' in sample:
                    raise ValueError("Label is required for non-test samples!")
                messages.append({"role": "assistant", "content": sample['label']})

        
        user_prompt = process_sample_feature(
            image_paths=image_paths, 
            qwen=qwen,
            description=description,
            context=context,
        )
        messages.append({"role": "user", "content": user_prompt})
        if not demonstrations: messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens+10,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        output_dict = {}
        output_dict['output'] = response

        if save_history:
            messages.append({"role": "assistant", "content": output_dict['output']})
            output_dict['history'] = messages
    else:
        raise ValueError(f"Model type {qwen['type']} not found. Supported types: qwen2.5, qwen2-vl, qwen-vl")
        
    return output_dict