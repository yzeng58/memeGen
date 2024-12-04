import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import GEMINI_API_KEY

import google.generativeai as genai
import PIL.Image, pdb
from helper import read_json

def load_gemini(model_path, api_key):
    model_name = model_path.split('/')[0]
    if not model_path.endswith('/pretrained'):
        raise ValueError(f"Gemini doesn't support custom models!")
    genai.configure(api_key = GEMINI_API_KEY[api_key])
    model = genai.GenerativeModel(model_name)
    return model

def process_sample_feature(description, context, image_paths):
    contents = []
    for i, image_path in enumerate(image_paths):
        idx_str = f" {i+1}" if len(image_paths) > 1 else ""
        if description:
            contents.append(f"Meme{idx_str}: {read_json(image_path)['description']['output']}\n")
        elif context:
            contents.append(f"Meme{idx_str}: {read_json(image_path['description_path'])['description']['output']}\n")
            contents.append(PIL.Image.open(image_path['image_path']))
        else:
            contents.append(PIL.Image.open(image_path))
    return contents
    
def call_gemini(
    model,
    prompt,
    image_paths: list[str] = [],
    history = None,
    save_history = False,
    description = '',
    context = "",
    temperature = 0,
    max_new_tokens = 1000,
    demonstrations = [],
    **kwargs,
):

    contents = []
    
    if demonstrations:
        contents.append(prompt)
        for sample in demonstrations:
            contents.extend(process_sample_feature(
                description=description, 
                context=context, 
                image_paths=sample['image_paths'],
            ))

            if not 'label' in sample:
                raise ValueError("Label is required for non-test samples!")
            contents.append(sample['label'])
    
    contents.extend(process_sample_feature(
        description=description, 
        context=context, 
        image_paths=image_paths,
    ))
    if not demonstrations: contents.append(prompt)
    
    output_dict = {}
        
    if history is None: history = []
    response = model.generate_content(
        history + contents, 
        stream = False,
        generation_config=genai.types.GenerationConfig(
            temperature = temperature,
            max_output_tokens = max_new_tokens,
        ),
    )

    if image_paths and len(description) == 0: 
        response.resolve()

    try:
        output_dict['output'] = response.text
    except ValueError:
        output_dict['output'] = ''
    
    if save_history: output_dict['history'] = contents + [output_dict['output']]
    
    return output_dict