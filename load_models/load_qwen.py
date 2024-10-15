# Update qwen/finetune.py

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from configs import system_prompts
from helper import read_json
from qwen_vl_utils import process_vision_info
import pdb
def load_qwen(
    model_name: str,
):  
    if 'qwen2' in model_name.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            f"Qwen/{model_name}", 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}")
        qwen = {
            'model': model,
            'processor': processor,
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}", device_map='auto', trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(f"Qwen/{model_name}", trust_remote_code=True)
        qwen = {
            'model': model,
            'tokenizer': tokenizer,
        }
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
    **kwargs,
):
    if 'tokenizer' in qwen:
        model, tokenizer = qwen['model'], qwen['tokenizer']  
        # make messages
        contents = []
        for i, image_path in enumerate(image_paths):
            if description:
                contents.append(process_text_qwen(f"Meme {i+1}: {read_json(image_path)['description']}\n"))
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

    else:
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_timestamp=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text = [text],
            images = image_inputs,
            videos = video_inputs,
            return_tensors = 'pt',
            padding = True,
        ).to(model.device)

        output_dict = {}
        
        generated_ids = model.generate(**inputs, max_new_tokens = max_new_tokens+4)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_dict['output'] = output_texts[0].replace('system\n', '').replace('assistant: ', '')

        if save_history:
            messages.append({"role": "assistant", "content": output_dict['output']})
            output_dict['history'] = messages
        
    return output_dict