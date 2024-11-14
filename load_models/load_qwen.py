import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from configs import system_prompts
from helper import read_json, set_seed
from qwen_vl_utils import process_vision_info
import pdb

def load_qwen(
    model_name: str,
):  
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
            f"Qwen/{model_name}", 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")
        qwen = {
            'model': model,
            'processor': processor,
            'type': 'qwen2-vl',
        }
    elif 'qwen-vl' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}", device_map='auto', trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
        qwen = {
            'model': model,
            'tokenizer': tokenizer,
            'type': 'qwen-vl',
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
    if qwen['type'] in ['qwen-vl']:
        model, tokenizer = qwen['model'], qwen['tokenizer']  
        # make messages
        contents = []
        for i, image_path in enumerate(image_paths):
            if description:
                contents.append(process_text_qwen(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
            elif context:
                contents.append(process_text_qwen(f"Meme {i+1}: {read_json(image_path['description_path'])['description']}\n"))
                contents.append(process_image_qwen(image_path['image_path']))
            else:
                contents.append(process_image_qwen(image_path))
        contents.append(process_text_qwen(prompt))
        
        query = tokenizer.from_list_format(contents)
        output_dict = {}
        output_dict['output'], output_dict['history'] = model.chat(
            tokenizer, 
            query=query, 
            history=history,
            system = system_prompts['qwen'][system_prompt],
        )

        if not save_history: output_dict.pop('history')

    elif qwen['type'] in ['qwen2-vl']:
        model, processor = qwen['model'], qwen['processor']
        
        if history:
            messages = history
        else:
            messages = [{"role": "system", "content": system_prompts['qwen'][system_prompt]}]

        contents = []
        for i, image_path in enumerate(image_paths):
            if description:
                contents.append(process_text_qwen2(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
            else:
                contents.append(process_image_qwen2(image_path))
        contents.append(process_text_qwen2(prompt))
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
            for sample_idx, sample in enumerate(demonstrations):
                image_paths = sample['image_paths']

                if len(image_paths) > 1:
                    for image_idx, image_path in enumerate(image_paths):
                        user_prompt = f"Meme {image_idx+1}: {read_json(image_path)['description']}"
                else:
                    user_prompt = f"Meme: {read_json(image_paths[0])['description']}"

                messages.append({"role": "user", "content": user_prompt})

                if sample_idx < len(demonstrations) - 1:
                    # this is not the test sample
                    if not 'label' in sample:
                        raise ValueError("Label is required for non-test samples!")
                    messages.append({"role": "assistant", "content": sample['label']})
        else:
            user_prompt = ""
            for i, image_path in enumerate(image_paths):
                user_prompt += f"Meme {i+1}: {read_json(image_path)['description']}\n"
            user_prompt += prompt
            messages.append({"role": "user", "content": user_prompt})


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