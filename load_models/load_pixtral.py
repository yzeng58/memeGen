import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from helper import read_json, set_seed
from configs import system_prompts
import pdb

def load_pixtral(model_path):
    model_name = model_path.split("/")[0]
    if model_path.endswith('/pretrained'):
        model_path = f"mistral-community/{model_name}"
    else:
        model_path = f"{root_dir}/models/{model_path}"
    model_id = f"mistral-community/{model_name}"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map = "auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return {
        'model': model,
        'processor': processor,
        'model_name': model_name,
    }

def process_text_pixtral(text):
    return {"type": "text", "content": text}

def process_image_pixtral(image_path):
    return {"type": "image"}

def process_sample_feature(
    description,
    context,
    image_paths,
):
    content = []
    for i, image_path in enumerate(image_paths):
        idx_str = f" {i+1}" if len(image_paths) > 1 else ""
        if description:
            content.append(process_text_pixtral(f"Meme{idx_str}: {read_json(image_path)['description']['output']}"))
        elif context:
            content.append(process_text_pixtral(f"Meme{idx_str}: {read_json(image_path["description_path"])['description']['output']}"))
            content.append(process_image_pixtral(read_json(image_path["image_path"])['image_path']))
        else:
            content.append(process_image_pixtral(image_path))
    return content

def call_pixtral(
    pixtral,
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
    model, processor, model_name = pixtral['model'], pixtral['processor'], pixtral['model_name']

    if history:
        messages = history
    else:
        messages = [
            {"role": "system", "content": system_prompts[model_name][system_prompt]}
        ]

    images = []
    if demonstrations:
        for idx, sample in enumerate(demonstrations):
            contents = []
            if idx == 0:
                contents.append(process_text_pixtral(prompt))
            contents.extend(process_sample_feature(
                description=description,
                context=context,
                image_paths=sample['image_paths'],
            ))
            images.extend([Image.open(image_path) for image_path in sample['image_paths']])
            messages.append({"role": "user", "content": contents})

            if not 'label' in sample:
                raise ValueError("Label is required for non-test samples!")
            messages.append({"role": "assistant", "content": sample['label']})

    contents = process_sample_feature(
        description=description,
        context=context,
        image_paths=image_paths,
    )
    if not demonstrations: contents.append(process_text_pixtral(prompt))
    messages.append({"role": "user", "content": contents})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_timestamp=True)
    if image_paths:
        images.extend([Image.open(image_path) for image_path in image_paths])
    else:
        images = None
    inputs = processor(
        text = text,
        images = images,
        return_tensors = 'pt',
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )[0]
    output_dict = {}
    output_dict['output'] = output

    if save_history:
        messages.append({"role": "assistant", "content": output})
        output_dict['history'] = messages
    return output_dict
    