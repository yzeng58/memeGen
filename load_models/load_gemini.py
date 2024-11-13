import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from environment import GEMINI_API_KEY

import google.generativeai as genai
import PIL.Image, pdb
from helper import read_json

def load_gemini(model, api_key):
    genai.configure(api_key = GEMINI_API_KEY[api_key])
    model = genai.GenerativeModel(model)
    return model
    
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
    **kwargs,
):

    contents = []
    for i, image_path in enumerate(image_paths):
        if description:
            contents.append(f"Meme {i+1}: {read_json(image_path)['description']}\n")
        elif context:
            contents.append(f"Meme {i+1}: {read_json(image_path['description_path'])['description']}\n")
            contents.append(PIL.Image.open(image_path['image_path']))
        else:
            contents.append(PIL.Image.open(image_path))
    contents.append(prompt)
    
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
    print(output_dict)
    
    return output_dict