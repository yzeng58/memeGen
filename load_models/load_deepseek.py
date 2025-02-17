import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import system_prompts
from helper import read_json, set_seed, retry_if_fail
from load_models.load_llama import call_llama
from load_models.load_qwen import call_qwen

def load_deepseek(
    model_path: str,
):
    model_name = model_path.split("/")[0]
    if model_path.endswith('/pretrained'):
        model_path = f"deepseek-ai/{model_name}"
    else:
        model_path = f"{root_dir}/models/{model_path}"
        
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{model_name}")
    return {
        'model': model,
        'tokenizer': tokenizer,
        'type': 'deepseek',
        "model_name": model_name,
    }


@retry_if_fail()
def call_deepseek(
    model_dict,
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    system_prompt = 'evaluator',
    description = '',
    max_new_tokens = 500,
    seed = 42,
    temperature = 0.6,
    context = "",
    demonstrations = None,
    **kwargs,
):
    set_seed(seed)
    if 'llama' in model_dict['model_name'].lower():
        model_dict['model_id'] = 'Llama-3.1'
        return call_llama(
            model_dict, 
            prompt, 
            image_paths, 
            max_new_tokens,
            history, 
            save_history, 
            system_prompt, 
            description, 
            seed, 
            context, 
            demonstrations, 
            **kwargs,
        )
    elif 'qwen' in model_dict['model_name'].lower():
        model_dict['type'] = 'qwen2.5'
        return call_qwen(
            model_dict, 
            prompt, 
            image_paths, 
            history, 
            save_history, 
            system_prompt, 
            description, 
            max_new_tokens, 
            seed, 
            temperature, 
            context, 
            demonstrations, 
            **kwargs,
        )
